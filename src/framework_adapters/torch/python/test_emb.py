import random
import numpy as np
import datetime
import argparse
import debugpy
import tqdm
import pytest
import os
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


import sys
sys.path.append("/home/xieminhui/RecStore/src/framework_adapters/torch")  # nopep8
from recstore import IPCTensorFactory, KGCacheController, load_recstore_library, Mfence
from PsKvstore import ShmKVStore

from controller_process import KGCacheControllerWrapper, TestPerfSampler
from cache_common import ShmTensorStore, TorchNativeStdEmbDDP
from utils import print_rank0, XLOG
from DistEmb import DistEmbedding
from PsKvstore import kvinit
import DistOpt

from cache_emb_factory import CacheEmbFactory

random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)


EmbContext = namedtuple('EmbContext', ["emb_name", 'sparse_opt', 'dist_opt'])


USE_SGD = True
# USE_SGD = False
LR = 1


def worker_main(routine, worker_id, num_workers, emb_context, args):
    torch.cuda.set_device(worker_id)
    torch.manual_seed(worker_id)
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12545')
    world_size = num_workers
    torch.distributed.init_process_group(backend=None,
                                         init_method=dist_init_method,
                                         world_size=world_size,
                                         rank=worker_id,
                                         timeout=datetime.timedelta(seconds=100))
    routine(worker_id, num_workers, emb_context, args)


class TestShardedCache:
    num_workers = 2

    EMB_DIM = 3
    EMB_LEN = 20

    # EMB_DIM = 32
    # EMB_LEN = 20000

    # BATCH_SIZE=10
    # EMB_LEN = int(1 * 1e6)
    BATCH_SIZE = 1024

    def main_routine(self, routine, args=None):
        # wrap rountine with dist_init
        print(
            f"========== Running Test with routine {routine} {args}==========")

        kvinit()
        emb = DistEmbedding(TestShardedCache.EMB_LEN,
                            TestShardedCache.EMB_DIM, name="full_emb",)

        fake_tensor = torch.Tensor([0])
        if USE_SGD:
            sparse_opt = optim.SGD(
                [fake_tensor], lr=LR,)
            dist_opt = DistOpt.SparseSGD(
                [emb], lr=LR)
        else:
            sparse_opt = optim.Adam([fake_tensor], lr=LR)
            dist_opt = DistOpt.SparseAdagrad(
                [emb], lr=LR)

        # import copy
        # deep_copy_dist_opt = copy.deepcopy(dist_opt)
        emb_context = EmbContext(
            emb_name=emb.name, sparse_opt=sparse_opt, dist_opt=None)

        XLOG.cdebug(
            f"ShmKVStore.tensor_store {hex(ShmKVStore.tensor_store['full_emb'].data_ptr())}")

        workers = []
        for worker_id in range(1, TestShardedCache.num_workers):
            p = mp.Process(target=worker_main, args=(
                routine, worker_id, TestShardedCache.num_workers, emb_context, args))
            p.start()
            XLOG.info(f"Worker {worker_id} pid={p.pid}")
            workers.append(p)
        worker_main(
            routine, 0, TestShardedCache.num_workers, emb_context, args)

        for each in workers:
            each.join()
            # assert each.exitcode == 0

        print("join all processes done")

    def init_emb_tensor(self, emb, rank, num_workers):
        dist.barrier()
        XLOG.info(f"emb.data_ptr={hex(emb.get_shm_tensor().data_ptr())}")
        linspace = np.linspace(0, emb.shape[0], num_workers+1, dtype=int)

        assert rank == dist.get_rank()
        if rank == 0:
            print(f"rank {rank} start initing emb")
            for i in tqdm.trange(linspace[rank], linspace[rank + 1]):
                emb.weight[i] = torch.ones(emb.shape[1]) * i
        else:
            for i in range(linspace[rank], linspace[rank + 1]):
                emb.weight[i] = torch.ones(emb.shape[1]) * i
        dist.barrier()
        Mfence.mfence()
        for i in range(1000):
            idx = random.randint(0, emb.shape[0]-1)
            # idx = i
            if not (torch.allclose(
                    emb.weight[idx], torch.ones(emb.shape[1]) * idx)):
                XLOG.error(
                    f"init failed, idx={idx}, emb[idx]={emb.weight[idx]}")
                assert False
        dist.barrier()

    def routine_cache_helper(self, worker_id, num_workers, emb_context, args):
        rank = dist.get_rank()
        XLOG.debug(f"rank{rank}: pid={os.getpid()}")
        kvinit()
        emb = DistEmbedding(TestShardedCache.EMB_LEN,
                            TestShardedCache.EMB_DIM, name=emb_context.emb_name)
        dist.barrier()

        sparse_opt = emb_context.sparse_opt
        # dist_opt = emb_context.dist_opt

        dist_opt = DistOpt.SparseSGD(
            [emb], lr=LR)

        XLOG.debug(f'dist_opt._params = {dist_opt._params}')

        XLOG.debug(
            f"in rank{rank}, full_emb.data_ptr={hex(emb.get_shm_tensor().data_ptr())}")

        self.init_emb_tensor(emb, worker_id, num_workers)

        json_str = r'''{{
            "num_gpus": {num_workers},
            "L": {L},
            "kForwardItersPerStep": {kForwardItersPerStep},
            "clr": {lr},
            "BackwardMode": "{BackwardMode}",
            "nr_background_threads": {nr_background_threads}
        }}'''.format(num_workers=num_workers,
                     kForwardItersPerStep=args['kForwardItersPerStep'],
                     L=args['L'],
                     lr=dist_opt.lr,
                     BackwardMode=args['BackwardMode'],
                     nr_background_threads=args['nr_background_threads'],
                     )

        if rank == 0:
            print("------------json------------")
            print(json_str)

        # Generate standard embedding done
        std_emb = TorchNativeStdEmbDDP(emb, device='cuda')
        std_emb.reg_opt(sparse_opt)
        # Generate standard embedding done

        # Generate our embedding
        cached_emb_type = args['test_cache']
        abs_emb = CacheEmbFactory.New(cached_emb_type, emb, args)

        abs_emb.reg_opt(sparse_opt)

        test_perf_sampler = TestPerfSampler(rank=rank,
                                            L=args['L'],
                                            num_ids_per_step=TestShardedCache.BATCH_SIZE,
                                            full_emb_capacity=emb.shape[0])
        kg_cache_controller = KGCacheControllerWrapper(
            json_str, emb, args
        )
        kg_cache_controller.init()

        # Generate our embedding done

        # forward
        for _ in tqdm.trange(100):
            sparse_opt.zero_grad()
            dist_opt.zero_grad()

            print(f"========== Step {_} ========== ", flush=True)
            input_keys = next(test_perf_sampler)

            XLOG.debug(f"{rank}:step{_}, input_keys {input_keys}")
            std_embed_value = std_emb.forward(input_keys)
            std_loss = std_embed_value.sum(-1).sum(-1)
            std_loss.backward()

            XLOG.cdebug(
                f"{rank}:std_embed_value {std_embed_value}")

            XLOG.cdebug(
                f"{rank}: emb_cache {abs_emb.emb_cache}")

            XLOG.cdebug(
                f"{rank}: full_emb {abs_emb.full_emb.get_shm_tensor()}")

            embed_value = abs_emb.forward(input_keys)
            XLOG.cdebug(f"{rank}:embed_value {embed_value}")

            loss = embed_value.sum(-1).sum(-1)
            loss.backward()

            XLOG.debug(f'full_emb.grad = {list(abs_emb.full_emb.get_grad())}')

            assert (torch.allclose(
                embed_value, std_embed_value)), "forward is error"
            assert (torch.allclose(
                loss, std_loss))

            sparse_opt.step()
            dist_opt.step()

            kg_cache_controller.AfterBackward()

    def test_known_sharded_cache(self,):
        # for test_cache in ["KnownShardedCachedEmbedding", "KnownLocalCachedEmbedding"]:
        for test_cache in ["KnownLocalCachedEmbedding"]:
            # for cache_ratio in [0.1, 0.3, 0.5]:
            for cache_ratio in [0.1,]:
                IPCTensorFactory.ClearIPCMemory()
                args = {"test_cache": test_cache,
                        "cache_ratio": cache_ratio,
                        "kForwardItersPerStep": 1,
                        # "BackwardMode": "PySync",
                        "BackwardMode": "CppSync",
                        "L": 10,
                        "nr_background_threads": 4,
                        }
                print("xmh: ", args)
                self.main_routine(self.routine_cache_helper, args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    test = TestShardedCache()
    test.test_known_sharded_cache()
