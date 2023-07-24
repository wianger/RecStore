import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import logging
from utils import all2all_data_transfer, merge_op


class ShardedCachedEmbedding(torch.autograd.Function):
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
        # embedding_weight = ShmTensorStore.GetTensor(emb_name)
        emb_dim = embedding_weight.shape[1]

        # 1. all to all keys
        # 1.1 split keys into shards
        sharded_keys = ShardedCachedEmbedding.split_keys_to_shards(
            keys, shard_range)

        # 1.2 all to all keys with shapes
        recv_keys = all2all_data_transfer(
            sharded_keys, None, tag=120, dtype=keys.dtype)
        print(f"Rank{rank}: keys after a2a", recv_keys, flush=True)

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

        logging.debug(f"{rank}: a2a cache_query_values",)
        cache_query_values_in_mine = all2all_data_transfer(
            cache_query_values, None, tag=121, dtype=cache_query_values[0].dtype)

        # 4. all to all missing keys
        logging.debug(f"{rank}: a2a missing_keys_in_mine",)
        missing_keys_in_mine = all2all_data_transfer(
            missing_keys, None, tag=122, dtype=missing_keys[0].dtype)

        logging.debug("a2a missing_indexs_in_mine ",)
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
            # logging.debug(cache_query_value.dtype)
            # logging.debug(missing_value.dtype)
            # logging.debug(missing_indexs.dtype)
            merge_op(cache_query_value, missing_value, missing_indexs)

        # 6. merge values
        # now: 里面都是按照shuffle后的顺序排的, len(cache_query_values_in_mine) = 8, 对应于sharded_keys的顺序

        # ctx.save_for_backward(keys,)
        # ctx.emb_name = emb_name
        # ctx.emb_dim = emb_dim

        ret_values = torch.concat(cache_query_values_in_mine, dim=0)
        print("ret_values ", ret_values)
        return ret_values

    @staticmethod
    def backward(ctx, grad_output):
        print("grad_output", grad_output, flush=True)
        keys, = ctx.saved_tensors
        emb_name = ctx.emb_name
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
                embedding_weight = ShmTensorStore.GetTensor(emb_name)
                coo_list = []
                temp = torch.sparse_coo_tensor(
                    [[], []], [], size=embedding_weight.shape)

                for each in range(len(keys_gather_list)):
                    coo_list.append(
                        torch.sparse_coo_tensor(keys_gather_list[each].unsqueeze(0), grad_gather_list[each],
                                                size=embedding_weight.shape)
                    )
                    temp += coo_list[-1].cpu()

                embedding_weight.grad = temp

        return None, None, None, torch.randn(1, 1)
