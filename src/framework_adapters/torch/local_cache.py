import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import logging
from utils import all2all_data_transfer, merge_op


class LocalCachedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, keys, embedding_weight, emb_cache, fake_tensor):
        emb_dim = embedding_weight.shape[1]
        # search cached keys
        query_key_len = keys.shape[0]
        values = torch.zeros((query_key_len, emb_dim)).cuda().requires_grad_()
        # cache query
        cache_query_result = emb_cache.Query(keys, values)

        # search missing keys in shared DRAM table
        missing_value = F.embedding(cache_query_result.missing_keys.cpu(
        ), embedding_weight, sparse=True, padding_idx=None, scale_grad_by_freq=False,)
        missing_value = missing_value.cuda()

        # join together
        merge_op(values, missing_value, cache_query_result.missing_index)

        ctx.save_for_backward(keys,)
        ctx.embedding_weight = embedding_weight
        ctx.emb_dim = emb_dim
        return values

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
        if dist.get_rank() == 0:
            grad_gather_list = [torch.zeros_like(
                grad_output) for _ in range(dist.get_world_size())]
        else:
            grad_gather_list = None

        handle_2 = dist.gather(grad_output.contiguous(
        ), dst=0, gather_list=grad_gather_list, async_op=True)

        handle_1.wait()
        handle_2.wait()

        # reduce all ranks' grad_outputs in rank 0
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

                embedding_weight.grad = temp

        return None, None, None, torch.randn(1, 1)
