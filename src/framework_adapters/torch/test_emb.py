import numpy as np
import datetime
import logging
import argparse
import debugpy
import tqdm
import pytest
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from cache_common import ShmTensorStore, TorchNativeStdEmb, CacheShardingPolicy
from sharded_cache import KnownShardedCachedEmbedding, ShardedCachedEmbedding
from local_cache import LocalCachedEmbedding, KnownLocalCachedEmbedding
from utils import print_rank0


import random
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)


torch.classes.load_library(
    "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")

logging.basicConfig(format='%(levelname)-2s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d:%H:%M:%S', level=logging.DEBUG)
# datefmt='%m-%d:%H:%M:%S', level=logging.INFO)


class TestShardedCache:
    num_workers = 2
    EMB_DIM = 3
    # EMB_LEN = 1000
    EMB_LEN = 3

    def main_routine(self, routine, args=None):
        # wrap rountine with dist_init
        def worker_main(routine, worker_id, num_workers, args):
            torch.cuda.set_device(worker_id)
            torch.manual_seed(worker_id)
            dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
                master_ip='127.0.0.1', master_port='12545')
            world_size = num_workers
            torch.distributed.init_process_group(backend=None,
                                                 init_method=dist_init_method,
                                                 world_size=world_size,
                                                 rank=worker_id,
                                                 timeout=datetime.timedelta(seconds=4))
            routine(worker_id, num_workers, args)

        print(
            f"========== Running Test with routine {routine} {args}==========")

        if ShmTensorStore.GetTensor("emb") is None:
            ShmTensorStore.RegTensor(
                "emb", (TestShardedCache.EMB_LEN, TestShardedCache.EMB_DIM))
        else:
            shm_tensor = ShmTensorStore.GetTensor("emb")
            assert shm_tensor.shape == (
                TestShardedCache.EMB_LEN, TestShardedCache.EMB_DIM)

        workers = []
        for worker_id in range(TestShardedCache.num_workers):
            p = mp.Process(target=worker_main, args=(
                routine, worker_id, TestShardedCache.num_workers, args))
            p.start()
            workers.append(p)

        for each in workers:
            each.join()

        print("join all processes done")

    def routine_shm_tensor(self, worker_id, num_workers, args):
        emb = ShmTensorStore.GetTensor("emb")
        if worker_id == 0:
            # print(worker_id, emb)
            for i in range(emb.shape[0]):
                emb[i, :] = torch.ones(emb.shape[1]) * i
        else:
            # time.sleep(2)
            pass
        mp.Barrier(num_workers)
        for i in range(emb.shape[0]):
            assert (torch.allclose(
                emb[i, :], torch.ones(emb.shape[1]) * i))

    @pytest.mark.skip(reason="simple")
    def test_shm_tensor(self):
        assert False
        self.main_routine(self.routine_shm_tensor)

    def init_emb_tensor(self, emb, worker_id, num_workers):
        if worker_id == 0:
            # print("pre", worker_id, emb)
            for i in range(emb.shape[0]):
                emb[i, :] = torch.ones(emb.shape[1]) * i
        # print(f"rank{worker_id} of {num_workers} arrived at barrier")
        # mp.Barrier(num_workers)
        dist.barrier()
        # print(f"rank{worker_id} of {num_workers} arrived after barrier")

        # print("post", worker_id, emb, flush=True)
        for i in range(emb.shape[0]):
            assert (torch.allclose(
                emb[i, :], torch.ones(emb.shape[1]) * i))

    def routine_cache_helper(self, worker_id, num_workers, args):
        USE_SGD = True
        # USE_SGD = False
        rank = dist.get_rank()

        emb = ShmTensorStore.GetTensor("emb")
        self.init_emb_tensor(emb, worker_id, num_workers)

        fake_tensor = torch.Tensor([0])
        sparse_opt = optim.SGD(
            [fake_tensor], lr=1,) if USE_SGD else optim.SparseAdam([], lr=1)

        # Generate standard embedding
        # std_emb = StdEmb(emb.clone())
        std_emb = TorchNativeStdEmb(emb, device='cuda')
        std_emb.reg_opt(sparse_opt)
        # Generate standard embedding done

        # Generate our embedding
        abs_emb = None
        emb_name = args['test_cache']

        if emb_name == "KnownShardedCachedEmbedding":
            cached_range = CacheShardingPolicy.generate_cached_range(
                emb, args['cache_ratio'])
            print_rank0(f"cache_range is {cached_range}")
            abs_emb = KnownShardedCachedEmbedding(
                emb, cached_range=cached_range)
        elif emb_name == "LocalCachedEmbedding":
            abs_emb = LocalCachedEmbedding(emb, cache_ratio=1,)
        elif emb_name == "KnownLocalCachedEmbedding":
            cached_range = CacheShardingPolicy.generate_cached_range(
                emb, args['cache_ratio'])
            print_rank0(f"cache_range is {cached_range}")
            abs_emb = KnownLocalCachedEmbedding(emb, cached_range=cached_range)
        else:
            assert False

        abs_emb.reg_opt(sparse_opt)
        # Generate our embedding done

        # forward

        for _ in tqdm.trange(20):
            # for _ in tqdm.trange(2):
            print(f"========== Step {_} ========== ")
            input_keys = torch.randint(emb.shape[0], size=(100,)).long().cuda()
            # if worker_id == 0:
            #     input_keys = torch.tensor([1, 2,],).long().cuda()
            # else:
            #     input_keys = torch.tensor([0, 2,],).long().cuda()
            logging.debug(f"{rank}:input_keys {input_keys}")

            std_embed_value = std_emb.forward(input_keys)
            std_loss = std_embed_value.sum(-1).sum(-1)
            std_loss.backward()
            logging.debug(f"{rank}:std_embed_value {std_embed_value}")

            embed_value = abs_emb.forward(input_keys)
            loss = embed_value.sum(-1).sum(-1)
            loss.backward()
            logging.debug(f"{rank}:embed_value {embed_value}")

            assert (torch.allclose(
                embed_value, std_embed_value)), "forward is error"
            assert (torch.allclose(
                loss, std_loss))

            # if worker_id == 0:
            #     assert (torch.allclose(
            #         emb.grad.to_dense().cpu(), std_emb.weight.grad.cpu())), "backward is error"

            sparse_opt.step()
            sparse_opt.zero_grad()

            mp.Barrier(num_workers)
            torch.cuda.synchronize()

            # for i in tqdm.trange(emb.shape[0]):
            #     assert (torch.allclose(
            #         emb[i, :], std_emb.weight[i, :].cpu(), atol=1e-6)), "opt is error"

    # @pytest.mark.parametrize("test_cache", ["KnownShardedCachedEmbedding", "KnownLocalCachedEmbedding"])
    # @pytest.mark.parametrize("cache_ratio", [0.1, 0.3, 0.5])
    # def test_known_sharded_cache(self, test_cache, cache_ratio):

    def test_known_sharded_cache(self,):

        for test_cache in ["KnownShardedCachedEmbedding", "KnownLocalCachedEmbedding"]:
            for cache_ratio in [0.1, 0.3, 0.5]:
                args = {"test_cache": test_cache, "cache_ratio": cache_ratio}
                print("xmh: ", args)
                self.main_routine(self.routine_cache_helper, args)

    @pytest.mark.skip(reason="now we use known local cache")
    def test_local_cache(self):
        args = {"test_cache": "LocalCachedEmbedding"}
        self.main_routine(self.routine_cache_helper, args)
