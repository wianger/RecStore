import numpy as np
import unittest
import datetime
import logging
import argparse
import debugpy
import tqdm

import torch
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
from DistOpt import SparseSGD, SparseAdagrad

import random
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)


torch.classes.load_library(
    "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")

logging.basicConfig(format='%(levelname)-2s [%(filename)s:%(lineno)d] %(message)s',
                    # datefmt='%m-%d:%H:%M:%S', level=logging.DEBUG)
                    datefmt='%m-%d:%H:%M:%S', level=logging.INFO)


def get_run_config():
    def parse_args(default_run_config):
        argparser = argparse.ArgumentParser("Training")
        argparser.add_argument('--num_workers', type=int,
                               default=8)
        argparser.add_argument('--num_embs', type=int,
                               default=100*1e6)
        #    default=100000)
        argparser.add_argument('--emb_dim', type=int,
                               default=32)
        argparser.add_argument('--batch_size', type=int,
                               #    default=1024*26)
                               default=1024)
        argparser.add_argument('--cache_ratio', type=float,
                               default=0.1)
        argparser.add_argument('--log_interval', type=float,
                               default=1000)
        argparser.add_argument('--run_steps', type=float,
                               default=1000)
        argparser.add_argument('--emb_choice', choices=["KnownShardedCachedEmbedding", "KnownLocalCachedEmbedding"]
                               )
        
        return vars(argparser.parse_args())

    run_config = {}
    run_config.update(parse_args(run_config))
    return run_config


def init_emb_tensor(emb, worker_id, num_workers):
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

    print(f"========== Running Perf with routine {routine}==========")

    kvinit()
    emb = DistEmbedding(int(ARGS['num_embs']),
                        int(ARGS['emb_dim']), name="emb",)
    # dummy LR, only register the tensor state of OSP
    opt = SparseSGD([emb], lr=100)

    workers = []
    for worker_id in range(ARGS['num_workers']):
        p = mp.Process(target=worker_main, args=(
            routine, worker_id, ARGS))
        p.start()
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

    if USE_SGD:
        sparse_opt = optim.SGD(
            [fake_tensor], lr=1,)
        dist_opt = SparseSGD([emb], lr=1/dist.get_world_size())
    else:
        sparse_opt = optim.Adam([fake_tensor], lr=1)
        dist_opt = SparseAdagrad([emb], lr=1/dist.get_world_size())

    abs_emb = None

    # emb_name = "TorchNativeStdEmb"
    # emb_name = "KnownShardedCachedEmbedding"  # 07:20<
    # emb_name = "KnownLocalCachedEmbedding" #05:21 1000iter

    emb_name = ARGS["emb_choice"]

    if emb_name == "KnownShardedCachedEmbedding":
        cached_range = CacheShardingPolicy.generate_cached_range(
            emb, ARGS['cache_ratio'])
        abs_emb = KnownShardedCachedEmbedding(emb, cached_range)
    elif emb_name == "LocalCachedEmbedding":
        abs_emb = LocalCachedEmbedding(emb, cache_ratio=ARGS['cache_ratio'],)
    elif emb_name == "TorchNativeStdEmb":
        abs_emb = TorchNativeStdEmb(emb, device='cuda')
    elif emb_name == "KnownLocalCachedEmbedding":
        cached_range = CacheShardingPolicy.generate_cached_range(
            emb, ARGS['cache_ratio'])
        abs_emb = KnownLocalCachedEmbedding(emb, cached_range=cached_range)
    else:
        assert False
    abs_emb.reg_opt(sparse_opt)
    # Generate our embedding done

    # forward
    start = datetime.datetime.now()
    start_step = 0
    for _ in tqdm.trange(ARGS['run_steps']):
        sparse_opt.zero_grad()
        dist_opt.zero_grad()

        # print(f"========== Step {_} ========== ")
        input_keys = torch.randint(emb.shape[0], size=(
            ARGS['batch_size'],)).long().cuda()

        embed_value = abs_emb.forward(input_keys)
        loss = embed_value.sum(-1).sum(-1)
        loss.backward()

        sparse_opt.step()
        dist_opt.step()

        if _ % ARGS['log_interval'] == (ARGS['log_interval']-1):
            end = datetime.datetime.now()
            print(f"Step{_}:rank{rank}, time {(end-start)/(_-start_step+1)}")
            start = datetime.datetime.now()
            start_step = _


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(5678)
    # print("wait debugpy connect", flush=True)
    # debugpy.wait_for_client()

    ARGS = get_run_config()
    main_routine(ARGS, routine_local_cache_helper)

    print("Successfully xmh")
