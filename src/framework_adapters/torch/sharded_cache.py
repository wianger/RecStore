import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim

from DistEmb import DistEmbedding
from cache_common import AbsEmb, NVGPUCache
from utils import Timer, all2all_data_transfer, merge_op, reduce_sparse_kv_tensor, all2all_sparse_tensor, sum_sparse_tensor
from utils import XLOG


class ShardedCachedEmbeddingFn(torch.autograd.Function):
    # shard_range: [0, 4, xxx, 100] len= rank+1
    @staticmethod
    def split_keys_to_shards(keys, shard_range):
        sharded_keys = []
        for shard_no in range(len(shard_range)-1):
            start = shard_range[shard_no]
            end = shard_range[shard_no+1]
            shard_keys = keys[(keys >= start) & (keys < end)]
            sharded_keys.append(shard_keys)
        return sharded_keys

    @staticmethod
    def forward(ctx, keys, embedding_weight, emb_cache, fake_tensor, shard_range):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        emb_dim = embedding_weight.shape[1]

        # 1. all to all keys
        # 1.1 split keys into shards
        sharded_keys = ShardedCachedEmbeddingFn.split_keys_to_shards(
            keys, shard_range)

        # 1.2 all to all keys with shapes
        recv_keys = all2all_data_transfer(
            sharded_keys, None, tag=120, dtype=keys.dtype)
        # print(f"Rank{rank}: keys after a2a", recv_keys, flush=True)

        # 2. search local cache
        query_values = []
        for i in recv_keys:
            query_key_len = i.shape[0]
            # values = torch.zeros((query_key_len, emb_dim)).cuda().requires_grad_()
            temp_values = torch.zeros((query_key_len, emb_dim)).cuda()
            query_values.append(temp_values)

        cache_query_results = emb_cache.BatchQuery(recv_keys, query_values)

        # 3. all to all searched values
        if rank == 0:
            for i in cache_query_results:
                print("each\n", i)

        cache_query_values = [each.values for each in cache_query_results]
        missing_keys = [each.missing_keys for each in cache_query_results]
        missing_indexs = [each.missing_index for each in cache_query_results]

        XLOG.debug(f"{rank}: a2a cache_query_values",)
        cache_query_values_in_mine = all2all_data_transfer(
            cache_query_values, None, tag=121, dtype=cache_query_values[0].dtype)

        # 4. all to all missing keys
        XLOG.debug(f"{rank}: a2a missing_keys_in_mine",)
        missing_keys_in_mine = all2all_data_transfer(
            missing_keys, None, tag=122, dtype=missing_keys[0].dtype)

        XLOG.debug("a2a missing_indexs_in_mine ",)
        missing_indexs_in_mine = all2all_data_transfer(
            missing_indexs, None, tag=123, dtype=missing_indexs[0].dtype)

        # 5. search missing keys
        if rank == 0:
            print("missing_keys_in_mine", missing_keys_in_mine)
            print("missing_indexs_in_mine", missing_indexs_in_mine)

        for i in range(len(cache_query_values_in_mine)):
            cache_query_value = cache_query_values_in_mine[i]
            missing_keys = missing_keys_in_mine[i]
            missing_indexs = missing_indexs_in_mine[i]
            missing_value = F.embedding(missing_keys.cpu(
            ),  embedding_weight, sparse=True, padding_idx=None, scale_grad_by_freq=False,)

            missing_value = missing_value.cuda()
            # join together

            # if rank == 0 and i == 0:
            # XLOG.debug(cache_query_value.dtype)
            # XLOG.debug(missing_value.dtype)
            # XLOG.debug(missing_indexs.dtype)
            merge_op(cache_query_value, missing_value, missing_indexs)

        # 6. merge values
        # now: 里面都是按照shuffle后的顺序排的, len(cache_query_values_in_mine) = 8, 对应于sharded_keys的顺序

        ctx.save_for_backward(keys,)
        ctx.emb_dim = emb_dim
        ctx.embedding_weight = embedding_weight

        ret_values = torch.concat(cache_query_values_in_mine, dim=0)
        print("ret_values ", ret_values)
        return ret_values

    @staticmethod
    def backward(ctx, grad_output):
        # print("grad_output", grad_output, flush=True)
        keys, = ctx.saved_tensors
        embedding_weight = ctx.embedding_weight
        emb_dim = ctx.emb_dim

        assert keys.shape[0] == grad_output.shape[0]
        assert emb_dim == grad_output.shape[1]

        # gather keys to rank 0
        if dist.get_rank() == 0:
            keys_gather_list = [torch.zeros_like(
                keys) for _ in range(dist.get_world_size())]
        else:
            keys_gather_list = None
        handle_1 = dist.gather(
            keys, dst=0, gather_list=keys_gather_list, async_op=True)
        # gather grad_output to rank 0

        # grad_output = torch.zeros((100,100)).cuda()

        if dist.get_rank() == 0:
            grad_gather_list = [torch.zeros_like(
                grad_output) for _ in range(dist.get_world_size())]
        else:
            grad_gather_list = None

        handle_2 = dist.gather(grad_output.contiguous(
        ), dst=0, gather_list=grad_gather_list, async_op=True)

        handle_1.wait()
        handle_2.wait()

        if dist.get_rank() == 0:
            assert len(keys_gather_list) == len(grad_gather_list)

            with torch.no_grad():
                coo_list = []
                temp = torch.sparse_coo_tensor(
                    [[], []], [], size=embedding_weight.shape)

                for each in range(len(keys_gather_list)):
                    coo_list.append(
                        torch.sparse_coo_tensor(keys_gather_list[each].unsqueeze(0), grad_gather_list[each],
                                                size=embedding_weight.shape)
                    )
                    temp += coo_list[-1].cpu()

                embedding_weight.grad = temp / dist.get_world_size()

        return None, None, None, torch.randn(1, 1), None

# 这个类确保了cache是静态的，而且Key在缓存中是连续的，即数组缓存


class ShardedCachedEmbedding(AbsEmb):
    def __init__(self, emb, cache_capacity, ) -> None:
        self.fake_tensor = torch.randn(1, 1, requires_grad=True)
        self.emb_dim = emb.shape[1]
        self.gpu_cache = NVGPUCache(
            int(emb.shape[0]*cache_capacity), self.emb_dim)

        rank, world_size = dist.get_rank(), dist.get_world_size()
        import numpy as np
        self.shard_range = np.linspace(
            0, emb.shape[0], num=world_size+1, dtype=int)

    def forward(self, input_keys):
        embed_value = ShardedCachedEmbeddingFn.apply(
            input_keys, self.emb, self.gpu_cache, self.fake_tensor, self.shard_range)
        assert embed_value.requires_grad
        return embed_value


class KnownShardedCachedEmbeddingFn(torch.autograd.Function):
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
            cached_keys = []
            in_cache_mask = torch.tensor([False] * len(keys), device="cuda")
            in_each_rank_cache_mask = []
            for shard_no in range(len(cached_range)):
                start, end = cached_range[shard_no]
                in_this_rank = (keys >= start) & (keys < end)
                in_cache_mask = in_cache_mask | in_this_rank

                in_each_rank_cache_mask.append(in_this_rank)
                shard_keys = keys[in_this_rank]
                cached_keys.append(shard_keys)
            return cached_keys, keys[in_cache_mask.logical_not()], in_cache_mask, in_each_rank_cache_mask

    @staticmethod
    def forward(ctx, keys, full_emb, emb_cache, fake_tensor, cached_range, ret_value):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        emb_dim = full_emb.shape[1]

        # 1. all to all keys
        # 1.1 split keys into shards

        ctx.timer_BarrierTimeBeforeRank0 = Timer("bucket keys")
        cached_keys, missing_keys, in_cache_mask, in_each_rank_cache_mask = KnownShardedCachedEmbeddingFn.CacheConfig.split_keys_to_shards(
            keys, cached_range)
        ctx.timer_BarrierTimeBeforeRank0.stop()

        # 1.2 all to all keys with shapes
        ctx.timer_a2akeys = Timer("forward: a2a keys")
        recv_keys = all2all_data_transfer(
            cached_keys, None, tag=120, dtype=keys.dtype)
        ctx.timer_a2akeys.stop()

        # 2. search local cache
        ctx.timer_searchcache = Timer("forward: search cache")
        cached_start_key, cached_end_key = cached_range[rank][0], cached_range[rank][1]
        ctx.cached_start_key = cached_start_key
        ctx.cached_end_key = cached_end_key
        cache_query_values = []
        for each_shard_recv_keys in recv_keys:
            # print("each_shard_recv_keys - cached_start_key",
            #       each_shard_recv_keys - cached_start_key)
            cache_query_values.append(
                emb_cache[each_shard_recv_keys - cached_start_key])
        ctx.timer_searchcache.stop()

        # 3. all to all searched values
        ctx.timer_a2avalues = Timer("forward: a2a values")
        XLOG.debug(f"{rank}: a2a cache_query_values",)
        cache_query_values_in_mine = all2all_data_transfer(
            cache_query_values, None, tag=121, dtype=cache_query_values[0].dtype)
        XLOG.debug(f"{rank}: a2a cache_query_values done",)
        ctx.timer_a2avalues.stop()

        # 5. merge into final result
        # ret_value = torch.zeros((keys.shape[0], emb_dim)).cuda()

        # 5.1 join missing keys
        ctx.timer_join = Timer("forward: join")
        if missing_keys.shape[0] > 0:
            if type(full_emb) is DistEmbedding:
                missing_value = full_emb(
                    missing_keys.cpu(), record_trace=False)
                # F.embedding(missing_keys.cpu(
                # ),  full_emb, sparse=True, padding_idx=None, scale_grad_by_freq=False,)

            else:
                missing_value = F.embedding(missing_keys.cpu(
                ),  full_emb, sparse=True, padding_idx=None, scale_grad_by_freq=False,)

            ret_value[in_cache_mask.logical_not()] = missing_value.cuda()
        else:
            pass

        # 5.2 join in-cache keys
        for cache_query_value, mask in zip(cache_query_values_in_mine, in_each_rank_cache_mask):
            ret_value[mask] = cache_query_value

        ctx.timer_join.stop()

        ctx.save_for_backward(keys, in_cache_mask, )
        ctx.emb_dim = emb_dim
        ctx.embedding_weight = full_emb
        ctx.emb_cache = emb_cache
        ctx.in_each_rank_cache_mask = in_each_rank_cache_mask

        return ret_value

    @staticmethod
    def backward(ctx, grad_output):
        keys, in_cache_mask, = ctx.saved_tensors
        in_each_rank_cache_mask = ctx.in_each_rank_cache_mask
        embedding_weight = ctx.embedding_weight
        emb_dim = ctx.emb_dim
        emb_cache = ctx.emb_cache

        assert keys.shape[0] == grad_output.shape[0]
        assert emb_dim == grad_output.shape[1]

        # 1. all to all keys's grad
        # aggregate in-dram keys
        ctx.timer_aggregate_dram = Timer("back: aggr dram keys")
        missing_keys = keys[in_cache_mask.logical_not()]
        missing_grads = grad_output[in_cache_mask.logical_not()]

        reduced_missing_grads = reduce_sparse_kv_tensor(
            missing_keys, missing_grads, embedding_weight.shape, 0)
        ctx.timer_aggregate_dram.stop()

        if dist.get_rank() == 0:
            ctx.timer_dram_grad = Timer("back: set dram grad")
            if type(embedding_weight) is DistEmbedding:
                reduced_missing_grads = reduced_missing_grads.coalesce()
                XLOG.debug(f"reduced_missing_grads = {reduced_missing_grads}")
                embedding_weight.record_grad(
                    reduced_missing_grads.indices().squeeze(0), reduced_missing_grads.values())
            else:
                embedding_weight.grad = reduced_missing_grads / dist.get_world_size()
            ctx.timer_dram_grad.stop()

        ctx.timer_chunk_grad = Timer("back: a2a grad")
        # split keys into shards
        sharded_keys = [keys[each] for each in in_each_rank_cache_mask]
        sharded_grads = [grad_output[each] for each in in_each_rank_cache_mask]
        keys_in_this_rank, values_in_this_rank = all2all_sparse_tensor(
            sharded_keys, sharded_grads, tag=124)
        ctx.timer_chunk_grad.stop()

        # all tensors minus cached_start_key in list(keys_in_this_rank)
        ctx.timer_cache_grad = Timer("back: grad sum & grad cache ")
        cached_start_key, cached_end_key = ctx.cached_start_key, ctx.cached_end_key
        cached_keys_in_this_rank = [
            each - cached_start_key for each in keys_in_this_rank]
        grad = sum_sparse_tensor(
            cached_keys_in_this_rank, values_in_this_rank, emb_cache.shape)

        grad = grad.cuda() / dist.get_world_size()
        emb_cache.grad = grad
        ctx.timer_cache_grad.stop()
        return None, None, None, torch.randn(1, 1), None, None


class KnownShardedCachedEmbedding(AbsEmb):
    def __init__(self, emb, cached_range, ) -> None:
        self.fake_tensor = torch.randn(1, 1, requires_grad=True)
        self.emb_dim = emb.shape[1]
        rank = dist.get_rank()

        start, end = cached_range[rank][0], cached_range[rank][1]
        cached_capacity = end - start

        self.emb_cache = torch.zeros((cached_capacity, self.emb_dim)).cuda()
        self.cached_range = cached_range
        self.emb = emb

        self.emb_cache.copy_(self.emb.weight[start:end])

        # self.ret_value = torch.zeros((int(1e5), self.emb_dim)).cuda()

    def forward(self, input_keys, trace=True):
        # ret_value = torch.narrow(self.ret_value, 0, 0, input_keys.shape[0])
        ret_value = torch.zeros((input_keys.shape[0], self.emb_dim)).cuda()

        assert input_keys.is_cuda
        embed_value = KnownShardedCachedEmbeddingFn.apply(
            input_keys, self.emb, self.emb_cache, self.fake_tensor, self.cached_range, ret_value)
        if trace:
            assert embed_value.requires_grad
        else:
            assert not embed_value.requires_grad
        return embed_value

    def reg_opt(self, opt):
        # TODO: Attenion!
        opt.add_param_group({"params": self.emb_cache})
        return
        if dist.get_rank() == 0:
            opt.add_param_group({"params": self.emb})
