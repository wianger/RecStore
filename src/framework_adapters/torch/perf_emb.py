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
from local_cache import LocalCachedEmbedding



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
                               default=2)
        argparser.add_argument('--num_embs', type=int,
                               #    default=0.01*1e6)
                               default=1000)
        argparser.add_argument('--emb_dim', type=int,
                               default=32)
        argparser.add_argument('--batch_size', type=int,
                               #    default=1024*26)
                               default=1024)
        return vars(argparser.parse_args())

    run_config = {}
    run_config.update(parse_args(run_config))
    return run_config


def init_emb_tensor(emb, worker_id, num_workers):
    if worker_id == 0:
        print(worker_id, emb)
        for i in range(emb.shape[0]):
            emb[i, :] = torch.ones(emb.shape[1]) * i
    mp.Barrier(num_workers)


def main_routine(ARGS, routine):
    # wrap rountine with dist_init
    def worker_main(routine, worker_id, ARGS):
        torch.cuda.set_device(worker_id)
        torch.manual_seed(worker_id)
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = ARGS['num_workers']
        torch.distributed.init_process_group(backend=None,
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=4))
        routine(worker_id, ARGS)

    print(f"========== Running Perf with routine {routine}==========")

    shm_tensor_store = ShmTensorStore(
        [("emb", (int(ARGS['num_embs']), int(ARGS['emb_dim'])))])

    workers = []
    for worker_id in range(ARGS['num_workers']):
        p = mp.Process(target=worker_main, args=(
            routine, worker_id, ARGS))
        p.start()
        workers.append(p)

    for each in workers:
        each.join()


def routine_local_cache_helper(worker_id, ARGS):
    USE_SGD = True
    # USE_SGD = False
    rank = dist.get_rank()

    emb = ShmTensorStore.GetTensor("emb")
    init_emb_tensor(emb, worker_id, ARGS['num_workers'])

    fake_tensor = torch.Tensor([0])
    sparse_opt = optim.SGD(
        [fake_tensor], lr=1,) if USE_SGD else optim.SparseAdam([fake_tensor], lr=1)

    # std_emb = TorchNativeStdEmb(emb.clone())
    # std_emb.reg_opt(sparse_opt)

    abs_emb = None
    emb_name = "KnownShardedCachedEmbedding"
    emb_name = "TorchNativeStdEmb"

    if emb_name == "KnownShardedCachedEmbedding":
        cached_range = CacheShardingPolicy.generate_cached_range(
            emb)
        abs_emb = KnownShardedCachedEmbedding(emb, cached_range)
    elif emb_name == "LocalCachedEmbedding":
        abs_emb = LocalCachedEmbedding(emb, cache_ratio=0.1,)
    elif emb_name == "TorchNativeStdEmb":
        abs_emb = TorchNativeStdEmb(emb, device='cuda')
    else:
        assert False
    abs_emb.reg_opt(sparse_opt)
    # Generate our embedding done

    # forward
    for _ in tqdm.trange(1000):
        print(f"========== Step {_} ========== ")
        input_keys = torch.randint(emb.shape[0], size=(
            ARGS['batch_size'],)).long()

        # std_embed_value = std_emb.forward(input_keys)
        # std_loss = std_embed_value.sum(-1).sum(-1)
        # std_loss.backward()
        # logging.debug(f"{rank}:std_embed_value {std_embed_value}")

        embed_value = abs_emb.forward(input_keys)
        loss = embed_value.sum(-1).sum(-1)
        loss.backward()
        logging.debug(f"{rank}:embed_value {embed_value}")

        # sparse_opt.step()
        # sparse_opt.zero_grad()

        mp.Barrier(ARGS['num_workers'])


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(5678)
    # print("wait debugpy connect", flush=True)
    # debugpy.wait_for_client()

    ARGS = get_run_config()
    main_routine(ARGS, routine_local_cache_helper)
