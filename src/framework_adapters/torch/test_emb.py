import random
import numpy as np
import datetime
import logging
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

from cache_common import ShmTensorStore, TorchNativeStdEmbDDP, CacheShardingPolicy
from sharded_cache import KnownShardedCachedEmbedding, ShardedCachedEmbedding
from local_cache import LocalCachedEmbedding, KnownLocalCachedEmbedding
from utils import print_rank0
from DistEmb import DistEmbedding
from PsKvstore import kvinit
from DistOpt import SparseSGD, SparseAdagrad

from cache_emb_factory import CacheEmbFactory

random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

logging.basicConfig(format='%(levelname)-2s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d:%H:%M:%S', level=logging.DEBUG)
# datefmt='%m-%d:%H:%M:%S', level=logging.INFO)


EmbContext = namedtuple('EmbContext', ['emb', 'sparse_opt', 'dist_opt'])
    

USE_SGD = True
# USE_SGD = False

class TestShardedCache:
    num_workers = 8
    EMB_DIM = 32

    EMB_LEN = 1000
    BATCH_SIZE=10
    # EMB_LEN = int(100* 1e6)
    # BATCH_SIZE=1024

    def main_routine(self, routine, args=None):
        # wrap rountine with dist_init
        def worker_main(routine, worker_id, num_workers, emb_context, args):
            '''
            if worker_id == 0:
                time.sleep(1000)
            print(f"yyyyrank {worker_id} reached barrier", flush=True)
            mp.Barrier(num_workers)
            print(f"yyyyrank {worker_id} escaped barrier", flush=True)
            '''

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

        print(
            f"========== Running Test with routine {routine} {args}==========")

        kvinit()
        emb = DistEmbedding(TestShardedCache.EMB_LEN,
                            TestShardedCache.EMB_DIM, name="emb",)
        
        fake_tensor = torch.Tensor([0])
        if USE_SGD:
            sparse_opt = optim.SGD(
                [fake_tensor], lr=1,)
            dist_opt = SparseSGD([emb], lr=1/TestShardedCache.num_workers)
        else:
            sparse_opt = optim.Adam([fake_tensor], lr=1)
            dist_opt = SparseAdagrad([emb], lr=1/TestShardedCache.num_workers)
        
        
        emb_context = EmbContext(emb=emb, sparse_opt=sparse_opt, dist_opt=dist_opt)

        workers = []
        for worker_id in range(TestShardedCache.num_workers):
            p = mp.Process(target=worker_main, args=(
                routine, worker_id, TestShardedCache.num_workers, emb_context, args))
            p.start()
            workers.append(p)

        for each in workers:
            each.join()
            assert each.exitcode == 0

        print("join all processes done")

    def routine_shm_tensor(self, worker_id, num_workers, args):
        # emb = ShmTensorStore.GetTensor("emb")
        emb = DistEmbedding(TestShardedCache.EMB_LEN,
                            TestShardedCache.EMB_DIM, name="emb",)
        self.init_emb_tensor(emb, worker_id, num_workers)

    @pytest.mark.skip(reason="simple")
    def test_shm_tensor(self):
        assert False
        self.main_routine(self.routine_shm_tensor)

    def init_emb_tensor(self, emb, worker_id, num_workers):
        import numpy as np    
        linspace = np.linspace(0, emb.shape[0], num_workers+1, dtype=int)
        if worker_id == 0:
            print(f"rank {worker_id} start initing emb")
            for i in tqdm.trange(linspace[worker_id], linspace[worker_id + 1]):
                emb.weight[i] = torch.ones(emb.shape[1]) * i
        else:
            for i in range(linspace[worker_id], linspace[worker_id + 1]):
                emb.weight[i] = torch.ones(emb.shape[1]) * i
        dist.barrier()
        for i in range(100):
            import random
            idx = random.randint(0, emb.shape[0]-1)
            assert (torch.allclose(
                emb.weight[idx], torch.ones(emb.shape[1]) * idx))
        dist.barrier()

    def routine_cache_helper(self, worker_id, num_workers, emb_context, args):
        rank = dist.get_rank()
        logging.debug(f"rank{rank}: pid={os.getpid()}")
        
        # emb = ShmTensorStore.GetTensor("emb")
        # emb = DistEmbedding(TestShardedCache.EMB_LEN,
        #                     TestShardedCache.EMB_DIM, name="emb",)
        emb = emb_context.emb
        sparse_opt = emb_context.sparse_opt 
        dist_opt = emb_context.dist_opt 

        self.init_emb_tensor(emb, worker_id, num_workers)

        # fake_tensor = torch.Tensor([0])
        # if USE_SGD:
        #     sparse_opt = optim.SGD(
        #         [fake_tensor], lr=1,)
        #     dist_opt = SparseSGD([emb], lr=1/dist.get_world_size())
        # else:
        #     sparse_opt = optim.Adam([fake_tensor], lr=1)
        #     dist_opt = SparseAdagrad([emb], lr=1/dist.get_world_size())

        # Generate standard embedding
        # std_emb = StdEmb(emb.clone())
        std_emb = TorchNativeStdEmbDDP(emb, device='cuda')
        std_emb.reg_opt(sparse_opt)
        # Generate standard embedding done

        # Generate our embedding
        cached_emb_type = args['test_cache']
        abs_emb = CacheEmbFactory.New(cached_emb_type, emb, args)
        abs_emb.reg_opt(sparse_opt)
        # Generate our embedding done

        # forward

        torch.set_grad_enabled(True)
        for _ in tqdm.trange(20):
            sparse_opt.zero_grad()
            dist_opt.zero_grad()

            print(f"========== Step {_} ========== ", flush=True)
            input_keys = torch.randint(emb.shape[0], size=(TestShardedCache.BATCH_SIZE,)).long().cuda()
            # if worker_id == 0:
            #     input_keys = torch.tensor([1, 2,],).long().cuda()
            #     # input_keys = torch.tensor([0, 1,],).long().cuda()
            # else:
            #     input_keys = torch.tensor([0, 2,],).long().cuda()
            #     # input_keys = torch.tensor([2, 3,],).long().cuda()

            logging.debug(f"{rank}:step{_}, input_keys {input_keys}")
            std_embed_value = std_emb.forward(input_keys)
            std_loss = std_embed_value.sum(-1).sum(-1)
            std_loss.backward()
            logging.debug(f"{rank}:std_embed_value {std_embed_value}")

            embed_value = abs_emb.forward(input_keys)
            logging.debug(f"{rank}:embed_value {embed_value}")
            loss = embed_value.sum(-1).sum(-1)
            
            loss.backward()

            assert (torch.allclose(
                embed_value, std_embed_value)), "forward is error"
            assert (torch.allclose(
                loss, std_loss))

            sparse_opt.step()
            dist_opt.step()

            dist.barrier()
            torch.cuda.synchronize()

            # if _ < 3:
            #     std_emb_249 = std_emb.forward(torch.tensor([249]).long().cuda())
            #     emb_249 = abs_emb.forward(torch.tensor([249]).long().cuda())
            #     if not torch.allclose(
            #         emb_249 , std_emb_249):
            #         logging.debug(f"step: {_}")
            #         logging.debug(f"std_emb_249: {std_emb_249}")
            #         logging.debug(f"emb_249 : {emb_249}")
            #         assert (torch.allclose(
            #             emb_249 , std_emb_249)), "forward is error"

    # @pytest.mark.parametrize("test_cache", ["KnownShardedCachedEmbedding", "KnownLocalCachedEmbedding"])
    # @pytest.mark.parametrize("cache_ratio", [0.1, 0.3, 0.5])
    # def test_known_sharded_cache(self, test_cache, cache_ratio):

    def test_known_sharded_cache(self,):
        # for test_cache in ["KnownShardedCachedEmbedding", "KnownLocalCachedEmbedding"]:
        for test_cache in ["KnownLocalCachedEmbedding"]:
            for cache_ratio in [0.1, 0.3, 0.5]:
                args = {"test_cache": test_cache, "cache_ratio": cache_ratio}
                print("xmh: ", args)
                self.main_routine(self.routine_cache_helper, args)

    @pytest.mark.skip(reason="now we use known local cache")
    def test_local_cache(self):
        args = {"test_cache": "LocalCachedEmbedding"}
        self.main_routine(self.routine_cache_helper, args)


if __name__ == "__main__":
    test = TestShardedCache()
    test.test_known_sharded_cache()
