import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim

import logging

from recstore import merge_op

_send_cpu, _recv_cpu = {}, {}

# data: [rank0_data, rank1_data, ...]


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


@torch.no_grad()
def all2all_data_transfer(data, recv_shape, tag, dtype=torch.float, verbose=False):
    logging.debug("before all2all_data_transfer")
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
            [world_size * len(data[0].shape)], dtype=torch.int64, device=torch.device("cuda")).chunk(world_size))

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

    logging.debug("after all2all_data_transfer")
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
def gather_variable_shape_tensor(tensor, dst_rank=0):
    # assert tensor.ndim in all ranks are same

    rank, world_size = dist.get_rank(), dist.get_world_size()

    # gather the shape of each rank into rank0
    if rank == dst_rank:
        shape_list = torch.empty(
            [world_size * tensor.ndim], dtype=torch.int64, device=torch.device("cuda")).chunk(world_size)
        shape_list = list(shape_list)
    else:
        shape_list = None

    shape = torch.tensor(tensor.shape, device="cuda", dtype=torch.int64)
    dist.gather(shape, gather_list=shape_list,
                dst=dst_rank)

    # gather the tensor
    tag = 100
    if rank == dst_rank:
        res = [torch.empty(each.tolist(), dtype=tensor.dtype,
                           device=torch.device("cuda")) for each in shape_list]

        req_list = []
        for i in range(1, world_size):
            src_rank = (rank + i) % world_size
            req = dist.irecv(res[src_rank], src=src_rank, tag=tag)
            req_list.append(req)
        res[dst_rank].copy_(tensor)
        [each.wait() for each in req_list]
    else:
        dist.send(tensor, dst=dst_rank, tag=tag)
        res = None

    return res


@torch.no_grad()
def reduce_sparse_tensor(sparse_tensor, dst_rank=0):
    rank, world_size = dist.get_rank(), dist.get_world_size()

    if not sparse_tensor.is_coalesced():
        sparse_tensor = sparse_tensor.coalesce()

    keys = sparse_tensor.indices()
    values = sparse_tensor.values()
    shape = sparse_tensor.shape

    keys_gather_list = gather_variable_shape_tensor(keys, dst_rank=dst_rank)
    assert len(values.shape) == 2

    values_list = gather_variable_shape_tensor(
        values.contiguous(), dst_rank=dst_rank)

    # logging.info(f"rank{dist.get_rank()}: gather keys and values done")
    if dist.get_rank() == dst_rank:
        logging.debug(f"rank{dist.get_rank()}: before sum sparse tensors")
        # logging.debug(f"rank{dist.get_rank()}: keys_gather_list {keys_gather_list }")
        # logging.debug(f"rank{dist.get_rank()}: values_list {values_list}")
        res = sum_sparse_tensor(keys_gather_list, values_list, shape)
        logging.debug(f"rank{dist.get_rank()}: after sum sparse tensors")
        # logging.info(f"rank{dist.get_rank()}: {res}")
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

    '''
    # Create an empty sparse tensor with the following invariants:
    #   1. sparse_dim + dense_dim = len(SparseTensor.shape)
    #   2. SparseTensor._indices().shape = (sparse_dim, nnz)
    #   3. SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])
    #
    # For instance, to create an empty sparse tensor with nnz = 0, dense_dim = 0 and
    # sparse_dim = 1 (hence indices is a 2D tensor of shape = (1, 0))
    >>> S = torch.sparse_coo_tensor(torch.empty([1, 0]), [], [1])
    tensor(indices=tensor([], size=(1, 0)),
        values=tensor([], size=(0,)),
        size=(1,), nnz=0, layout=torch.sparse_coo)
    '''
    coo_list = []
    #  here, sparse_dim = 1, dense_dim = shape[1:], 
    res = torch.sparse_coo_tensor(torch.empty([1, 0]), torch.empty([0, *shape[1:]]), size=shape)

    for each in range(len(keys_list)):
        if keys_list[each].nelement() == 0:
            continue

        temp = keys_list[each]
        # map keys_list[each] to [[k0, k1, k2, ...]]
        if keys_list[each].dim() != 2:
            temp = keys_list[each].unsqueeze(0)
        assert temp.dim() == 2
        assert values_list[each].shape[1:] == shape[1:]

        coo_list.append(
            torch.sparse_coo_tensor(temp, values_list[each],
                                    size=shape)
        )
        res += coo_list[-1].cpu()
    return res
