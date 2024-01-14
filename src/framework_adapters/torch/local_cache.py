import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from utils import XLOG

from DistEmb import DistEmbedding
from PsKvstore import get_kvstore, ShmKVStore
from utils import all2all_data_transfer, merge_op, kv_to_sparse_tensor, reduce_sparse_kv_tensor, all2all_sparse_tensor, sum_sparse_tensor, PerfCounter, Timer
from cache_common import AbsEmb, NVGPUCache

import recstore
from recstore import uva_cache_query_op, load_recstore_library


class LocalCachedEmbeddingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, keys, embedding_weight, emb_cache, fake_tensor):
        emb_dim = embedding_weight.shape[1]
        # search cached keys
        query_key_len = keys.shape[0]
        ret_value = torch.zeros((query_key_len, emb_dim)
                                ).cuda().requires_grad_()
        # cache query
        cache_query_result = emb_cache.Query(keys, ret_value)

        # search missing keys in shared DRAM table
        missing_value = F.embedding(cache_query_result.missing_keys.cpu(
        ), embedding_weight, sparse=True, padding_idx=None, scale_grad_by_freq=False,)
        missing_value = missing_value.cuda()

        # join together
        merge_op(ret_value, missing_value, cache_query_result.missing_index)

        ctx.save_for_backward(keys,)
        ctx.embedding_weight = embedding_weight
        ctx.emb_dim = emb_dim
        return ret_value

    @staticmethod
    def backward(ctx, grad_output):
        keys, = ctx.saved_tensors
        embedding_weight = ctx.embedding_weight
        emb_dim = ctx.emb_dim

        assert keys.shape[0] == grad_output.shape[0]
        assert emb_dim == grad_output.shape[1]

        # gather keys to rank 0
        grad = reduce_sparse_kv_tensor(
            keys, grad_output, embedding_weight.shape, dst_rank=0)
        if dist.get_rank() == 0:
            embedding_weight.grad = grad / dist.get_world_size()

        # if dist.get_rank() == 0:
        #     keys_gather_list = [torch.zeros_like(
        #         keys) for _ in range(dist.get_world_size())]
        # else:
        #     keys_gather_list = None
        # handle_1 = dist.gather(
        #     keys, dst=0, gather_list=keys_gather_list, async_op=True)

        # # gather grad_output to rank 0
        # if dist.get_rank() == 0:
        #     grad_gather_list = [torch.zeros_like(
        #         grad_output) for _ in range(dist.get_world_size())]
        # else:
        #     grad_gather_list = None

        # handle_2 = dist.gather(grad_output.contiguous(
        # ), dst=0, gather_list=grad_gather_list, async_op=True)

        # handle_1.wait()
        # handle_2.wait()

        # # reduce all ranks' grad_outputs in rank 0
        # if dist.get_rank() == 0:
        #     with torch.no_grad():
        #         grad = sum_sparse_tensor(
        #             keys_gather_list, grad_gather_list, embedding_weight.shape)
        #         embedding_weight.grad = grad / dist.get_world_size()

        return None, None, None, torch.randn(1, 1)


class LocalCachedEmbedding(AbsEmb):
    def __init__(self, emb, cache_ratio, ) -> None:
        self.fake_tensor = torch.randn(1, 1, requires_grad=True)
        self.emb = emb
        self.emb_dim = emb.shape[1]
        self.gpu_cache = NVGPUCache(
            int(emb.shape[0]*cache_ratio), self.emb_dim)

        raise NotImplementedError("TODO: update cache in backward ")

    def forward(self, input_keys):
        embed_value = LocalCachedEmbeddingFn.apply(
            input_keys, self.emb, self.gpu_cache, self.fake_tensor)
        assert embed_value.requires_grad
        return embed_value

    def reg_opt(self, opt):
        # TODO: Attenion!
        return
        if dist.get_rank() == 0:
            opt.add_param_group({"params": self.emb})


class KnownLocalCachedEmbeddingFn(torch.autograd.Function):
    class CacheConfig:
        # cached_range:
        # [ (start, end ),  # rank0
        #  (start, end),    # rank1
        #  (start, end),  ....
        #  (start, end),
        #  (start, end),    # rank7
        # ]
        # return: ([keys in rank0, keys in rank1.... ,], missing_keys, in_cache_mask, in_each_rank_cache_mask)
        @staticmethod
        def split_keys_to_shards(keys, cached_range):
            assert len(cached_range) == dist.get_world_size()
            # cached_keys = []
            in_each_rank_cache_mask = []
            for shard_no in range(len(cached_range)):
                start, end = cached_range[shard_no]
                in_this_rank = (keys >= start) & (keys < end)
                in_each_rank_cache_mask.append(in_this_rank)
            return in_each_rank_cache_mask

    @staticmethod
    def forward(ctx, keys, full_emb, emb_cache,
                fake_tensor, cached_range, ret_value,
                backward_grads, backward_grads_2
                ):
        rank, world_size = dist.get_rank(), dist.get_world_size()

        emb_dim = full_emb.shape[1]
        # keys 并不保序
        cached_start_key, cached_end_key = cached_range[rank][0], cached_range[rank][1]
        ctx.cached_start_key = cached_start_key
        ctx.cached_end_key = cached_end_key

        # 统计 keys 中有多少在本地缓存中
        # PERF_STATISTIC = True
        PERF_STATISTIC = False
        if PERF_STATISTIC:
            in_each_rank_cache_mask = KnownLocalCachedEmbeddingFn.CacheConfig.split_keys_to_shards(
                keys, cached_range)
            in_this_rank_cache_mask = in_each_rank_cache_mask[rank]
            hit_count = in_this_rank_cache_mask.sum()
            hit_ratio = hit_count / keys.shape[0]
            PerfCounter.Record("hit_ratio", hit_ratio)

        uva_cache_query_op(ret_value,
                           keys,
                           emb_cache,
                           full_emb.get_shm_tensor(),
                           cached_start_key,
                           cached_end_key)

        '''
        # 1.1 split keys into shards
        in_each_rank_cache_mask = KnownLocalCachedEmbeddingFn.CacheConfig.split_keys_to_shards(
            keys, cached_range)
        in_this_rank_cache_mask = in_each_rank_cache_mask[rank]
        not_in_this_rank_cache_mask = in_this_rank_cache_mask.logical_not()

        missing_keys = keys[not_in_this_rank_cache_mask]

        # 2. search local cache
        XLOG.debug("search local cache")
        cached_start_key, cached_end_key = cached_range[rank][0], cached_range[rank][1]
        ctx.cached_start_key = cached_start_key
        ctx.cached_end_key = cached_end_key

        # 3. merge into final result

        # 3.1 join missing keys
        XLOG.debug("join missing keys")

        if type(full_emb) is DistEmbedding:
            XLOG.debug(f'rank{rank}, before full_emb(missing_keys.cpu()')
            missing_value = full_emb(missing_keys.cpu(), record_trace=False)
            XLOG.debug(f'rank{rank}, after full_emb(missing_keys.cpu()')
        else:
            missing_value = F.embedding(missing_keys.cpu(
            ),  full_emb, sparse=True, padding_idx=None, scale_grad_by_freq=False,)

        ret_value[not_in_this_rank_cache_mask] = missing_value.cuda()

        # 3.2 join hit keys
        ret_value[in_this_rank_cache_mask] = emb_cache[keys[in_this_rank_cache_mask] - cached_start_key]
        '''
        ctx.save_for_backward(keys, )
        ctx.emb_dim = emb_dim
        ctx.embedding_weight = full_emb
        ctx.emb_cache = emb_cache
        ctx.cached_range = cached_range

        ctx.backward_grads = backward_grads
        ctx.backward_grads_2 = backward_grads_2
        return ret_value

    @staticmethod
    @torch.no_grad()
    def backward_py_sync(ctx, grad_output):
        rank = dist.get_rank()
        keys, = ctx.saved_tensors

        full_emb = ctx.embedding_weight
        emb_dim = ctx.emb_dim
        emb_cache = ctx.emb_cache
        in_each_rank_cache_mask = KnownLocalCachedEmbeddingFn.CacheConfig.split_keys_to_shards(
            keys, ctx.cached_range)

        # 下面的1.待优化
        assert keys.shape[0] == grad_output.shape[0]
        assert emb_dim == grad_output.shape[1]

        # 1. update local cache's grad
        XLOG.debug("backward: update local cache's grad")
        # 1.1 all to all keys's grad
        sharded_keys = [keys[each] for each in in_each_rank_cache_mask]
        sharded_grads = [grad_output[each] for each in in_each_rank_cache_mask]
        keys_in_this_rank, values_in_this_rank = all2all_sparse_tensor(
            sharded_keys, sharded_grads, tag=124)
        XLOG.debug("backward: all to all keys's grad done")

        # 1.2 update grad of local cache
        cached_start_key, cached_end_key = ctx.cached_start_key, ctx.cached_end_key
        cached_keys_in_this_rank = [
            each - cached_start_key for each in keys_in_this_rank]
        grad = sum_sparse_tensor(
            cached_keys_in_this_rank, values_in_this_rank, emb_cache.shape)
        grad = grad.cuda() / dist.get_world_size()
        emb_cache.grad = grad

        # 上面的1.待优化
        # 2 aggregate grad of in-dram keys
        reduced_dram_grads = reduce_sparse_kv_tensor(
            keys, grad_output, full_emb.shape, 0)
        XLOG.debug(f"rank{rank}: aggregate grad of in-dram keys done")

        mp.Barrier(dist.get_world_size())

        if dist.get_rank() == 0:
            reduced_dram_grads = reduced_dram_grads.coalesce()
            XLOG.debug(
                f"rank0: reduced_dram_grads {reduced_dram_grads._nnz()}")
            if type(full_emb) is DistEmbedding:
                full_emb.record_grad(
                    reduced_dram_grads.indices().squeeze(0), reduced_dram_grads.values())
            else:
                full_emb.grad = reduced_dram_grads / dist.get_world_size()
            XLOG.debug("rank0: set DRAM Emb's grad done")

        '''
        # 实际中发现上面的方式2.会使得rank0成为瓶颈，这里试一下直接每个rank自己设自己的，只测性能，正确性有问题
        if type(full_emb) is DistEmbedding:
            full_emb.record_grad(
               keys, grad_output)
            # if rank == 0:
            #     XLOG.warn(f"in local cache's back {keys.shape}, {grad_output}")
        else:
            dram_grads = kv_to_sparse_tensor(keys, grad_output, full_emb.shape)
            full_emb.grad = dram_grads / dist.get_world_size()
        '''
        return None, None, None, torch.randn(1, 1), None, None, None, None

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output):
        if KnownLocalCachedEmbeddingFn.backward_mode == "PySync":
            return KnownLocalCachedEmbeddingFn.backward_py_sync(ctx, grad_output)
        elif KnownLocalCachedEmbeddingFn.backward_mode == "CppSync" \
                or KnownLocalCachedEmbeddingFn.backward_mode == "CppAsync" \
                or KnownLocalCachedEmbeddingFn.backward_mode == "CppAsyncV2":
            rank = dist.get_rank()
            keys, = ctx.saved_tensors
            backward_grads = ctx.backward_grads
            backward_grads_2 = ctx.backward_grads_2

            if KnownLocalCachedEmbeddingFn.backgrad_init != "both":
                backward_grads.Copy_(
                    grad_output / dist.get_world_size(), non_blocking=False)
            else:
                backward_grads.Copy_(
                    grad_output / dist.get_world_size(), non_blocking=True)
                backward_grads_2.Copy_(
                    grad_output / dist.get_world_size(), non_blocking=True)
                torch.cuda.synchronize()
            return None, None, None, torch.randn(1, 1), None, None, None, None
        else:
            assert False


class KnownLocalCachedEmbedding(AbsEmb):
    def __init__(self, full_emb, cached_range, kForwardItersPerStep, backward_mode, backgrad_init) -> None:
        self.kForwardItersPerStep = kForwardItersPerStep

        KnownLocalCachedEmbeddingFn.backward_mode = backward_mode
        KnownLocalCachedEmbeddingFn.backgrad_init = backgrad_init

        self.fake_tensor = torch.randn(1, 1, requires_grad=True)
        self.emb_dim = full_emb.shape[1]
        rank = dist.get_rank()
        self.rank = rank

        start, end = cached_range[rank][0], cached_range[rank][1]
        cached_capacity = end - start

        # self.emb_cache = torch.zeros((cached_capacity, self.emb_dim)).cuda()
        self.emb_cache = recstore.IPCTensorFactory.NewIPCGPUTensor(
            f"embedding_cache_{rank}", [cached_capacity, self.emb_dim], torch.float32, rank)

        self.input_keys_shm = recstore.IPCTensorFactory.NewSlicedIPCTensor(
            f"input_keys_{rank}", [int(1e5),], torch.int64, )
        self.input_keys_neg_shm = recstore.IPCTensorFactory.NewSlicedIPCTensor(
            f"input_keys_neg_{rank}", [int(1e5),], torch.int64, )

        self.backgrad_init = backgrad_init

        if backgrad_init == 'cpu':
            self.backward_grads_shm = recstore.IPCTensorFactory.NewSlicedIPCTensor(
                f"backward_grads_{rank}", [int(1e5), self.emb_dim], torch.float, )
            self.backward_grads_neg_shm = recstore.IPCTensorFactory.NewSlicedIPCTensor(
                f"backward_grads_neg_{rank}", [int(1e5), self.emb_dim], torch.float, )
        elif backgrad_init == 'gpu':
            self.backward_grads_shm = recstore.IPCTensorFactory.NewSlicedIPCGPUTensor(
                f"backward_grads_{rank}", [int(1e5), self.emb_dim], torch.float, rank)
            self.backward_grads_neg_shm = recstore.IPCTensorFactory.NewSlicedIPCGPUTensor(
                f"backward_grads_neg_{rank}", [int(1e5), self.emb_dim], torch.float, rank)
        elif backgrad_init == 'both':
            self.backward_grads_shm_cpu = recstore.IPCTensorFactory.NewSlicedIPCTensor(
                f"backward_grads_{rank}", [int(1e5), self.emb_dim], torch.float, )
            self.backward_grads_neg_shm_cpu = recstore.IPCTensorFactory.NewSlicedIPCTensor(
                f"backward_grads_neg_{rank}", [int(1e5), self.emb_dim], torch.float, )

            self.backward_grads_shm_gpu = recstore.IPCTensorFactory.NewSlicedIPCGPUTensor(
                f"backward_grads_{rank}_gpu", [int(1e5), self.emb_dim], torch.float, rank)
            self.backward_grads_neg_shm_gpu = recstore.IPCTensorFactory.NewSlicedIPCGPUTensor(
                f"backward_grads_neg_{rank}_gpu", [int(1e5), self.emb_dim], torch.float, rank)
        else:
            assert False

        self.cached_range = cached_range
        self.full_emb = full_emb

        # NOTE: all processes need register the UVA region
        #  even though the region is exactly same.
        ShmKVStore.GetUVAMap(full_emb.get_shm_tensor())
        dist.barrier()
        XLOG.debug(f"{rank}: after UVAMap")
        self.emb_cache.copy_(self.full_emb.weight[start:end])
        dist.barrier()

        # TODO: 检查一下为啥直接这样分配会有问题
        self.ret_value = torch.zeros((int(1e5), self.emb_dim)).cuda()
        # self.ret_value = recstore.IPCTensorFactory.NewIPCGPUTensor(
        #     f"ret_value{rank}", [int(1e5), self.emb_dim], torch.float, rank)
        self.iter = 0
        XLOG.debug(f"{rank}: KnownLocalCachedEmbedding init done")
        print(f"{rank}: KnownLocalCachedEmbedding init done", flush=True)

    def GetCache(self):
        return self.emb_cache

    def forward(self, input_keys, trace=True):
        assert input_keys.is_cuda
        assert input_keys.shape[0] <= self.ret_value.shape[0]
        ret_value = torch.narrow(self.ret_value, 0, 0, input_keys.shape[0])
        if self.iter % self.kForwardItersPerStep == 0:
            self.input_keys_shm.Copy_(input_keys, non_blocking=False)
            assert self.input_keys_shm.GetSlicedTensor(
            ).shape[0] == input_keys.shape[0]
        else:
            self.input_keys_neg_shm.Copy_(input_keys, non_blocking=False)
            assert self.input_keys_neg_shm.GetSlicedTensor(
            ).shape[0] == input_keys.shape[0]

        if self.backgrad_init == 'cpu' or self.backgrad_init == 'gpu':
            if self.iter % self.kForwardItersPerStep == 0:
                embed_value = KnownLocalCachedEmbeddingFn.apply(
                    input_keys, self.full_emb, self.emb_cache, self.fake_tensor,
                    self.cached_range, ret_value, self.backward_grads_shm, None)
            else:
                embed_value = KnownLocalCachedEmbeddingFn.apply(
                    input_keys, self.full_emb, self.emb_cache, self.fake_tensor,
                    self.cached_range, ret_value, self.backward_grads_neg_shm, None)
        else:
            if self.iter % self.kForwardItersPerStep == 0:
                embed_value = KnownLocalCachedEmbeddingFn.apply(
                    input_keys, self.full_emb, self.emb_cache, self.fake_tensor,
                    self.cached_range, ret_value, self.backward_grads_shm_cpu, self.backward_grads_shm_gpu)
            else:
                embed_value = KnownLocalCachedEmbeddingFn.apply(
                    input_keys, self.full_emb, self.emb_cache, self.fake_tensor,
                    self.cached_range, ret_value, self.backward_grads_neg_shm_cpu, self.backward_grads_neg_shm_gpu)

        if trace:
            assert embed_value.requires_grad
        else:
            assert not embed_value.requires_grad

        self.iter += 1
        return embed_value

    def reg_opt(self, opt):
        # TODO: Attenion!
        opt.add_param_group({"params": self.emb_cache})

        # if dist.get_rank() == 0:
        #     opt.add_param_group({"params": self.full_emb.weight})
