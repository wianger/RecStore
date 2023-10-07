import numpy as np
import unittest
import datetime
import argparse
import debugpy
import tqdm

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from cache_common import ShmTensorStore, TorchNativeStdEmb, CacheShardingPolicy
from sharded_cache import KnownShardedCachedEmbedding, ShardedCachedEmbedding
from local_cache import KnownLocalCachedEmbedding, LocalCachedEmbedding
from DistEmb import DistEmbedding
from PsKvstore import kvinit
from DistOpt import SparseSGD, SparseAdagrad, SparseRowWiseAdaGrad
from utils import XLOG
import time

import random
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)


from pyinstrument import Profiler


from contextlib import contextmanager
@contextmanager
def xmh_nvtx_range(msg, condition=True):
    """
    Context manager / decorator that pushes an NVTX range at the beginning
    of its scope, and pops it at the end. If extra arguments are given,
    they are passed as arguments to msg.format().

    Args:
        msg (str): message to associate with the range
    """
    if condition:
        th.cuda.nvtx.range_push(msg)
        yield
        th.cuda.nvtx.range_pop()
    else:
        yield
        

def get_run_config():
    def parse_args(default_run_config):
        argparser = argparse.ArgumentParser("Training")
        argparser.add_argument('--num_workers', type=int,
                               default=4)
        argparser.add_argument('--num_embs', type=int,
                            #    default=100*1e6)
                               default=10*1e6)
        argparser.add_argument('--emb_dim', type=int,
                               default=32)
        argparser.add_argument('--with_perf', type=bool,
                               default=False)
        argparser.add_argument('--batch_size', type=int,
                                  default=1024*26)
                            #    default=1024)
        argparser.add_argument('--cache_ratio', type=float,
                               default=0.1)
        argparser.add_argument('--log_interval', type=int,
                               default=1000)
        argparser.add_argument('--run_steps', type=int,
                               default=1000)
        argparser.add_argument('--emb_choice', choices=["TorchNativeStdEmb", "KnownShardedCachedEmbedding", "KnownLocalCachedEmbedding"],
                               default="KnownShardedCachedEmbedding"
                               )
        
        return vars(argparser.parse_args())

    run_config = {}
    run_config.update(parse_args(run_config))
    return run_config


def init_emb_tensor(emb, worker_id, num_workers):
    return 
    import numpy as np
    linspace = np.linspace(0, emb.shape[0], num_workers+1, dtype=int)
    if worker_id == 0:
        print(f"rank {worker_id} start initing emb")
        for i in tqdm.trange(linspace[worker_id], linspace[worker_id + 1]):
            emb.weight[i] = torch.ones(emb.shape[1]) * i
    else:
        for i in range(linspace[worker_id], linspace[worker_id + 1]):
            emb.weight[i] = torch.ones(emb.shape[1]) * i
    print(f"rank {worker_id} reached barrier")
    dist.barrier()
    print(f"rank {worker_id} escaped barrier")


def main_routine(ARGS, routine):
    # wrap rountine with dist_init
    def worker_main(routine, worker_id, ARGS):
        torch.cuda.set_device(worker_id)
        torch.manual_seed(worker_id)
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12335')
        world_size = ARGS['num_workers']
        torch.distributed.init_process_group(backend=None,
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=100))
        routine(worker_id, ARGS)


    kvinit()
    XLOG.warn("Before init DistEmbedding")
    emb = DistEmbedding(int(ARGS['num_embs']),
                        int(ARGS['emb_dim']), name="emb",)
    XLOG.warn("After init DistEmbedding")

    # dummy LR, only register the tensor state of OSP
    opt = SparseSGD([emb], lr=100)
    dist_opt = SparseRowWiseAdaGrad([emb], lr=1)

    print(f"========== Running Perf with routine {routine}==========")
    workers = []
    for worker_id in range(ARGS['num_workers']):
        p = mp.Process(target=worker_main, args=(
            routine, worker_id, ARGS))
        p.start()
        print(f"Worker {worker_id} pid={p.pid}")
        workers.append(p)

    for each in workers:
        each.join()
        assert each.exitcode == 0


def routine_local_cache_helper(worker_id, ARGS):
    USE_SGD = True
    # USE_SGD = False
    rank = dist.get_rank()

    
    emb = DistEmbedding(int(ARGS['num_embs']),
                        int(ARGS['emb_dim']), name="emb",)

    init_emb_tensor(emb, worker_id, ARGS['num_workers'])

    fake_tensor = torch.Tensor([0])

    # if USE_SGD:
    #     sparse_opt = optim.SGD(
    #         [fake_tensor], lr=1,)
    #     dist_opt = SparseSGD([emb], lr=1/dist.get_world_size())
    # else:
    #     sparse_opt = optim.Adam([fake_tensor], lr=1)
    #     dist_opt = SparseAdagrad([emb], lr=1/dist.get_world_size())

    sparse_opt = optim.SGD(
        [fake_tensor], lr=1,)
    dist_opt = SparseSGD([emb], lr=1/dist.get_world_size())

    # dist_opt = SparseRowWiseAdaGrad([emb], lr=1/dist.get_world_size())

    abs_emb = None

    emb_name = ARGS["emb_choice"]

    if emb_name == "KnownShardedCachedEmbedding":
        cached_range = CacheShardingPolicy.generate_cached_range(
            emb, ARGS['cache_ratio'])
        abs_emb = KnownShardedCachedEmbedding(emb, cached_range)
    elif emb_name == "LocalCachedEmbedding":
        abs_emb = LocalCachedEmbedding(emb, cache_ratio=ARGS['cache_ratio'],)
    elif emb_name == "TorchNativeStdEmb":
        abs_emb = TorchNativeStdEmb(emb, device='cpu')
    elif emb_name == "KnownLocalCachedEmbedding":
        cached_range = CacheShardingPolicy.generate_cached_range(
            emb, ARGS['cache_ratio'])
        abs_emb = KnownLocalCachedEmbedding(emb, cached_range=cached_range)
    else:
        assert False
    abs_emb.reg_opt(sparse_opt)
    # Generate our embedding done

    std_emb = TorchNativeStdEmb(emb, device='cpu')

    # forward
    start = time.time()
    start_step = 0

    warmup_iters = 100

    profiler = Profiler()
    
    if ARGS['with_perf']:
        for_range = tqdm.trange(ARGS['run_steps'])
    else:
        for_range = range(ARGS['run_steps'])
    
    for _ in for_range:
        if _ == warmup_iters and rank ==0 and ARGS['with_perf']:
            profiler.start()
            print("cudaProfilerStart")
            torch.cuda.cudart().cudaProfilerStart()
        if _ == warmup_iters + 10 and rank ==0 and ARGS['with_perf']:
            profiler.stop()
            profiler.print()
            print("cudaProfilerStop")
            torch.cuda.cudart().cudaProfilerStop()
            break

        sparse_opt.zero_grad()
        dist_opt.zero_grad()
        
        input_keys = torch.randint(emb.shape[0], size=(
            ARGS['batch_size'],)).long().cuda()

        with xmh_nvtx_range(f"Step{_}:forward", condition=rank == 0 and _ >= warmup_iters and ARGS['with_perf']):
            embed_value = abs_emb.forward(input_keys)

        # embed_value = std_emb.forward(input_keys)
        loss = embed_value.sum(-1).sum(-1)

        loss.backward()
        sparse_opt.step()
        dist_opt.step()

        if (_ % ARGS['log_interval']) == (ARGS['log_interval']-1):
            end = time.time()
            print(f"Step{_}:rank{rank}, time: {end-start:.3f}, per_step: {(end-start)/(_-start_step+1):.6f}",flush=True)
            start = time.time()
            start_step = _


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(5678)
    # print("wait debugpy connect", flush=True)
    # debugpy.wait_for_client()

    ARGS = get_run_config()
    main_routine(ARGS, routine_local_cache_helper)

    print("Successfully xmh")
