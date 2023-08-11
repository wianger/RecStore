import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import logging
from utils import all2all_data_transfer, merge_op, reduce_sparse_kv_tensor, all2all_sparse_tensor, sum_sparse_tensor
from cache_common import AbsEmb, NVGPUCache


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
    def forward(ctx, keys, embedding_weight, emb_cache, fake_tensor, cached_range):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        emb_dim = embedding_weight.shape[1]
        ret_value = torch.zeros((keys.shape[0], emb_dim)).cuda()

        # 1.1 split keys into shards
        in_each_rank_cache_mask = KnownLocalCachedEmbeddingFn.CacheConfig.split_keys_to_shards(
            keys, cached_range)
        in_this_rank_cache_mask = in_each_rank_cache_mask[rank]
        not_in_this_rank_cache_mask = in_this_rank_cache_mask.logical_not()

        missing_keys = keys[not_in_this_rank_cache_mask]

        # 2. search local cache
        cached_start_key, cached_end_key = cached_range[rank][0], cached_range[rank][1]
        ctx.cached_start_key = cached_start_key
        ctx.cached_end_key = cached_end_key

        # 3. merge into final result

        # 3.1 join missing keys
        missing_value = F.embedding(missing_keys.cpu(
        ),  embedding_weight, sparse=True, padding_idx=None, scale_grad_by_freq=False,)
        ret_value[not_in_this_rank_cache_mask] = missing_value.cuda()

        # 3.2 join hit keys
        ret_value[in_this_rank_cache_mask] = emb_cache[keys[in_this_rank_cache_mask] - cached_start_key]

        ctx.save_for_backward(keys, )
        ctx.emb_dim = emb_dim
        ctx.embedding_weight = embedding_weight
        ctx.emb_cache = emb_cache
        ctx.in_each_rank_cache_mask = in_each_rank_cache_mask

        return ret_value

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output):
        keys, = ctx.saved_tensors

        embedding_weight = ctx.embedding_weight
        emb_dim = ctx.emb_dim
        emb_cache = ctx.emb_cache
        in_each_rank_cache_mask = ctx.in_each_rank_cache_mask

        assert keys.shape[0] == grad_output.shape[0]
        assert emb_dim == grad_output.shape[1]

        # 1. update local cache's grad
        # 1.1 all to all keys's grad
        sharded_keys = [keys[each] for each in in_each_rank_cache_mask]
        sharded_grads = [grad_output[each] for each in in_each_rank_cache_mask]
        keys_in_this_rank, values_in_this_rank = all2all_sparse_tensor(
            sharded_keys, sharded_grads, tag=124)

        # 1.2 update grad of local cache
        cached_start_key, cached_end_key = ctx.cached_start_key, ctx.cached_end_key
        cached_keys_in_this_rank = [
            each - cached_start_key for each in keys_in_this_rank]
        grad = sum_sparse_tensor(
            cached_keys_in_this_rank, values_in_this_rank, emb_cache.shape)
        grad = grad.cuda() / dist.get_world_size()
        emb_cache.grad = grad

        # 2 aggregate grad of in-dram keys
        reduced_dram_grads = reduce_sparse_kv_tensor(
            keys, grad_output, embedding_weight.shape, 0)
        if dist.get_rank() == 0:
            embedding_weight.grad = reduced_dram_grads
        return None, None, None, torch.randn(1, 1), None


class KnownLocalCachedEmbedding(AbsEmb):
    def __init__(self, emb, cached_range, ) -> None:
        self.fake_tensor = torch.randn(1, 1, requires_grad=True)
        self.emb_dim = emb.shape[1]
        rank = dist.get_rank()

        start, end = cached_range[rank][0], cached_range[rank][1]
        cached_capacity = end - start

        self.emb_cache = torch.zeros((cached_capacity, self.emb_dim)).cuda()
        self.cached_range = cached_range
        self.emb = emb
        self.emb_cache.copy_(self.emb[start:end, :])

    def forward(self, input_keys):
        embed_value = KnownLocalCachedEmbeddingFn.apply(
            input_keys, self.emb, self.emb_cache, self.fake_tensor, self.cached_range)
        assert embed_value.requires_grad
        return embed_value

    @staticmethod
    def generate_cached_range(emb, cache_ratio):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        capacity = emb.shape[0]

        per_shard_size = (capacity + world_size-1) // world_size
        cached_range = []
        for i in range(world_size):
            start = i * per_shard_size
            end = min((i+1) * per_shard_size, capacity)
            cached_range.append((start, end))
        return cached_range

    def reg_opt(self, opt):
        opt.add_param_group({"params": self.emb_cache})
        if dist.get_rank() == 0:
            opt.add_param_group({"params": self.emb})
