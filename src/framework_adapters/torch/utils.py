import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim

import logging


torch.ops.load_library(
    "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")
merge_op = torch.ops.librecstore_pytorch.merge_op

_send_cpu, _recv_cpu = {}, {}

# data: [rank0_data, rank1_data, ...]


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


@torch.no_grad()
def all2all_data_transfer(data, recv_shape, tag, dtype=torch.float, verbose=False):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if verbose:
        logging.debug(f'{rank}, a2a, input={data}')

    if recv_shape is None:
        # first
        # shard_keys_sizes: [shard0_size, shard1_size, ...]
        shard_data_shapes = [torch.tensor(
            [each.shape], device=torch.device("cuda")).long() for each in data]

        for each in data[1:]:
            assert len(data[0].shape) == len(each.shape)

        per_shard_shapes = list(torch.empty(
            [world_size * len(each.shape)], dtype=torch.int64, device=torch.device("cuda")).chunk(world_size))

        dist.all_to_all(per_shard_shapes, shard_data_shapes,)
        # per_shard_size: [rank0_shape_in_mine, rank1_shape_in_mine, ....]
        recv_shape = per_shard_shapes

    if verbose:
        logging.debug(f'{rank}, recv_shape={recv_shape}')

    msg, res = [None] * world_size, [None] * world_size
    for i in range(1, world_size):
        idx = (rank + i) % world_size
        key = 'dst%d_tag%d' % (idx, tag)
        if True or key not in _recv_cpu:
            _send_cpu[key] = torch.zeros_like(
                data[idx], dtype=dtype, device='cpu', pin_memory=True)
            _recv_cpu[key] = torch.zeros(
                recv_shape[idx].tolist(), dtype=dtype, pin_memory=True)
        msg[idx] = _send_cpu[key]
        res[idx] = _recv_cpu[key]

    for i in range(1, world_size):
        left = (rank - i + world_size) % world_size
        right = (rank + i) % world_size
        msg[right].copy_(data[right])
        if verbose:
            logging.debug(f"{rank}, data[right]={data[right]}")
            logging.debug(f"{rank}, msg[right]={msg[right]}")

        if msg[right].nelement() != 0:
            req = dist.isend(msg[right], dst=right, tag=tag)
            if verbose:
                logging.debug(f"{rank}->{right}, dist.isend, {msg[right]}")

        # logging.debug(f"{rank}, {res[left]}")
        if res[left].nelement() != 0:
            # logging.debug(f"{rank}<-{left}, before dist.recv, {res[left]}")
            dist.recv(res[left], src=left, tag=tag)
            # logging.debug(f"{rank}<-{left}, after dist.recv, {res[left]}")
        res[left] = res[left].cuda(non_blocking=True)

        if msg[right].nelement() != 0:
            req.wait()

    res[rank] = data[rank]
    return res


# def all2all_data_transfer(data, recv_shape, tag, dtype=torch.float, verbose=False):
#     rank, world_size = dist.get_rank(), dist.get_world_size()
#     if verbose:
#         logging.debug(f'{rank}, a2a, input={data}')

#     if recv_shape is None:
#         # first
#         # shard_keys_sizes: [shard0_size, shard1_size, ...]
#         shard_data_shapes = [torch.tensor(
#             [each.shape], device=torch.device("cuda")).long() for each in data]

#         for each in data[1:]:
#             assert len(data[0].shape) == len(each.shape)

#         per_shard_shapes = list(torch.empty(
#             [world_size * len(each.shape)], dtype=torch.int64, device=torch.device("cuda")).chunk(world_size))

#         dist.all_to_all(per_shard_shapes, shard_data_shapes,)
#         # per_shard_size: [rank0_shape_in_mine, rank1_shape_in_mine, ....]
#         recv_shape = per_shard_shapes

#     if verbose:
#         logging.debug(f'{rank}, recv_shape={recv_shape}')

#     msg, res = [None] * world_size, [None] * world_size
#     for i in range(1, world_size):
#         idx = (rank + i) % world_size
#         key = 'dst%d_tag%d' % (idx, tag)
#         if True or key not in _recv_cpu:
#             _send_cpu[key] = torch.zeros_like(
#                 data[idx], dtype=dtype, device='cpu', pin_memory=True)
#             _recv_cpu[key] = torch.zeros(
#                 recv_shape[idx].tolist(), dtype=dtype, pin_memory=True)
#         msg[idx] = _send_cpu[key]
#         res[idx] = _recv_cpu[key]

#     for i in range(1, world_size):
#         left = (rank - i + world_size) % world_size
#         right = (rank + i) % world_size
#         msg[right].copy_(data[right])
#         if verbose:
#             logging.debug(f"{rank}, data[right]={data[right]}")
#             logging.debug(f"{rank}, msg[right]={msg[right]}")

#         if msg[right].nelement() != 0:
#             req = dist.isend(msg[right], dst=right, tag=tag)
#             if verbose:
#                 logging.debug(f"{rank}->{right}, dist.isend, {msg[right]}")

#         # logging.debug(f"{rank}, {res[left]}")
#         if res[left].nelement() != 0:
#             # logging.debug(f"{rank}<-{left}, before dist.recv, {res[left]}")
#             dist.recv(res[left], src=left, tag=tag)
#             # logging.debug(f"{rank}<-{left}, after dist.recv, {res[left]}")
#         res[left] = res[left].cuda(non_blocking=True)

#         if msg[right].nelement() != 0:
#             req.wait()

#     res[rank] = data[rank]
#     return res


@torch.no_grad()
def reduce_sparse_tensor(sparse_tensor, dst_rank=0):
    if not sparse_tensor.is_coalesced():
        sparse_tensor = sparse_tensor.coalesce()
        
    keys = sparse_tensor.indices()
    values = sparse_tensor.values()
    shape = sparse_tensor.shape

    if dist.get_rank() == dst_rank:
        keys_gather_list = [torch.zeros_like(
            keys) for _ in range(dist.get_world_size())]
    else:
        keys_gather_list = None
    handle_1 = dist.gather(
        keys, dst=dst_rank, gather_list=keys_gather_list, async_op=True)
    # gather grad_output to rank 0

    if dist.get_rank() == dst_rank:
        value_list = [torch.zeros_like(
            values) for _ in range(dist.get_world_size())]
    else:
        value_list = None

    handle_2 = dist.gather(values.contiguous(
    ), dst=dst_rank, gather_list=value_list, async_op=True)

    handle_1.wait()
    handle_2.wait()

    if dist.get_rank() == dst_rank:
        res = sum_sparse_tensor(keys_gather_list, value_list, shape)
        res = res / dist.get_world_size()
    else:
        res = None

    return res


@torch.no_grad()
def reduce_sparse_kv_tensor(keys, values, shape, dst_rank=0):
    # print_rank0(f'shape = {shape}')
    # print_rank0(f'keys = {keys}')
    # print_rank0(f'values = {values}')

    if keys.dim() != 2:
        temp = keys.unsqueeze(0)
        assert temp.dim() == 2
    return reduce_sparse_tensor(torch.sparse_coo_tensor(temp, values, size=shape), dst_rank)


def all2all_sparse_tensor(keys, values, tag, verbose=False):
    a2a_keys = all2all_data_transfer(keys, None, tag=tag,
                                     dtype=torch.int64, verbose=verbose)

    a2a_values = all2all_data_transfer(values, None, tag=tag,
                                       dtype=torch.float32, verbose=verbose)

    return a2a_keys, a2a_values


@torch.no_grad()
def sum_sparse_tensor(keys_list, values_list, shape):
    assert len(keys_list) == len(values_list)

    coo_list = []
    res = torch.sparse_coo_tensor(
        [[], []], [], size=shape)
    for each in range(len(keys_list)):
        if keys_list[each].nelement() == 0:
            continue

        temp = keys_list[each]
        # map keys_list[each] to [[k0, k1, k2, ...]]
        if len(keys_list[each].shape) != 2:
            temp = keys_list[each].unsqueeze(0)
            assert len(temp.shape) == 2

        coo_list.append(
            torch.sparse_coo_tensor(temp, values_list[each],
                                    size=shape)
        )
        res += coo_list[-1].cpu()
    return res
