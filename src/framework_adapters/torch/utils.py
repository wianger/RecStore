import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim

import logging
import time

from recstore import merge_op

_send_cpu, _recv_cpu = {}, {}

# data: [rank0_data, rank1_data, ...]

RECSTORE_UTIL_DEBUG = False

if RECSTORE_UTIL_DEBUG:
    logging.basicConfig(format='%(levelname)-2s [%(process)d %(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d:%H:%M:%S', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(levelname)-2s [%(process)d %(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d:%H:%M:%S', level=logging.INFO)


XLOG = logging


# class TimeFactory:
#     all_timers = {}

#     @classmethod
#     def beautifyNs(cls, s):
#         ns = s * 1e9
#         if (int(ns / 1000) == 0):
#             return f'{ns:.3f} ns'
#         ns /= 1000
#         if (int(ns / 1000) == 0):
#             return f'{ns:.3f} us'
#         ns /= 1000
#         if (int(ns / 1000) == 0):
#             return f'{ns:.3f} ms'
#         ns /= 1000
#         return f'{ns:.3f} s'

#     @classmethod
#     def AddToTimer(cls, timer):
#         if timer.name not in cls.all_timers:
#             cls.all_timers[timer.name] = timer

#     @classmethod
#     def AddToPerfCounter(cls, timer):
#         if timer.name not in cls.all_timers:
#             cls.all_timers[timer.name] = timer

#     @classmethod
#     def Report(cls):
#         print_str = f"Timer: Rank{dist.get_rank()}\n"
#         for name, timer in cls.all_timers.items():
#             print_str += f"{name}: {cls.beautifyNs(timer.average_time())}\n"

#         XLOG.error(print_str)


# class PerfCounter:
#     @classmethod
#     def Record(cls, name, value):
#         cls.all_counters[name] = value


# class Timer:
#     def __init__(self, name):
#         self.start_time = 0
#         self.end_time = 0
#         self.elapsed_time = 0
#         self.runs = 0
#         self.total_time = 0
#         self.name = name

#         TimeFactory.AddToTimer(self)

#     def start(self):
#         self.start_time = time.time()

#     def stop(self):
#         self.end_time = time.time()
#         self.elapsed_time = self.end_time - self.start_time
#         self.total_time += self.elapsed_time
#         self.runs += 1

#     def average_time(self):
#         if self.runs == 0:
#             return 0
#         return self.total_time / self.runs

import sys  # nopep8
sys.path.append("/home/xieminhui/RecStore/build/lib")  # nopep8
import timer_module  # nopep8


class Timer:
    @classmethod
    def StartReportThread(cls):
        timer_module.Reporter.StartReportThread(5000)

    @classmethod
    def StopReportThread(cls):
        timer_module.Reporter.StopReportThread()

    def __init__(self, name):
        self._c_timer = timer_module.Timer(name, 1)

    def start(self):
        self._c_timer.Start()

    def stop(self):
        self._c_timer.End()


class GPUTimer:
    @classmethod
    def StartReportThread(cls):
        timer_module.Reporter.StartReportThread(5000)

    def __init__(self, name):
        self.name = name
        self.start()

    def start(self):
        self.tick = torch.cuda.Event(enable_timing=True)
        self.tick.record()

    def stop(self):
        self.tock = torch.cuda.Event(enable_timing=True)
        self.tock.record()
        self.tock.synchronize()
        elapsed_time_ms = self.tick.elapsed_time(self.tock)
        timer_module.Timer.ManualRecordNs(self.name, elapsed_time_ms*1e3*1e3)


class PerfCounter:
    @classmethod
    def Record(cls, name, value):
        timer_module.PerfCounter.Record(name, value)


def XLOG_debug(str):
    rank = None
    try:
        rank = dist.get_rank()
    except:
        pass

    if rank == 0:
        color = "red"
    elif rank is not None:
        color = "green"
    else:
        color = "red"

    if color == "red":
        if rank is None:
            XLOG.debug(f'\033[31m{str}\033[0m')
        else:
            XLOG.debug(f'\033[31m{rank}:{str}\033[0m')
    elif color == "green":
        if rank is None:
            XLOG.debug(f'\033[32m{str}\033[0m')
        else:
            XLOG.debug(f'\033[32m{rank}:{str}\033[0m')
    else:
        if rank is None:
            XLOG.debug(f'{str}')
        else:
            XLOG.debug(f'{rank}:{str}')


XLOG.cdebug = XLOG_debug


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


@torch.no_grad()
def all2all_data_transfer(data, recv_shape, tag, dtype=torch.float, verbose=False):
    XLOG.debug("before all2all_data_transfer")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if verbose:
        XLOG.debug(f'{rank}, a2a, input={data}')

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
        XLOG.debug(f'{rank}, recv_shape={recv_shape}')

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
            XLOG.debug(f"{rank}, data[right]={data[right]}")
            XLOG.debug(f"{rank}, msg[right]={msg[right]}")

        if msg[right].nelement() != 0:
            req = dist.isend(msg[right], dst=right, tag=tag)
            if verbose:
                XLOG.debug(f"{rank}->{right}, dist.isend, {msg[right]}")

        # XLOG.debug(f"{rank}, {res[left]}")
        if res[left].nelement() != 0:
            # XLOG.debug(f"{rank}<-{left}, before dist.recv, {res[left]}")
            dist.recv(res[left], src=left, tag=tag)
            # XLOG.debug(f"{rank}<-{left}, after dist.recv, {res[left]}")
        res[left] = res[left].cuda(non_blocking=True)

        if msg[right].nelement() != 0:
            req.wait()

    res[rank] = data[rank]

    XLOG.debug("after all2all_data_transfer")
    return res


# def all2all_data_transfer(data, recv_shape, tag, dtype=torch.float, verbose=False):
#     rank, world_size = dist.get_rank(), dist.get_world_size()
#     if verbose:
#         XLOG.debug(f'{rank}, a2a, input={data}')

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
#         XLOG.debug(f'{rank}, recv_shape={recv_shape}')

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
#             XLOG.debug(f"{rank}, data[right]={data[right]}")
#             XLOG.debug(f"{rank}, msg[right]={msg[right]}")

#         if msg[right].nelement() != 0:
#             req = dist.isend(msg[right], dst=right, tag=tag)
#             if verbose:
#                 XLOG.debug(f"{rank}->{right}, dist.isend, {msg[right]}")

#         # XLOG.debug(f"{rank}, {res[left]}")
#         if res[left].nelement() != 0:
#             # XLOG.debug(f"{rank}<-{left}, before dist.recv, {res[left]}")
#             dist.recv(res[left], src=left, tag=tag)
#             # XLOG.debug(f"{rank}<-{left}, after dist.recv, {res[left]}")
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

    # XLOG.info(f"rank{dist.get_rank()}: gather keys and values done")
    if dist.get_rank() == dst_rank:
        XLOG.debug(f"rank{dist.get_rank()}: before sum sparse tensors")
        # XLOG.debug(f"rank{dist.get_rank()}: keys_gather_list {keys_gather_list }")
        # XLOG.debug(f"rank{dist.get_rank()}: values_list {values_list}")
        res = sum_sparse_tensor(keys_gather_list, values_list, shape)
        XLOG.debug(f"rank{dist.get_rank()}: after sum sparse tensors")
        # XLOG.info(f"rank{dist.get_rank()}: {res}")
    else:
        res = None

    return res


@torch.no_grad()
def kv_to_sparse_tensor(keys, values, shape):
    if keys.dim() != 2:
        temp = keys.unsqueeze(0)
        assert temp.dim() == 2
    return torch.sparse_coo_tensor(temp, values, size=shape)


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
    res = torch.sparse_coo_tensor(torch.empty(
        [1, 0]), torch.empty([0, *shape[1:]]), size=shape)

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
