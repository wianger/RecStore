from sharded_cache import ShardedCachedEmbedding
from local_cache import LocalCachedEmbedding
import unittest
import torch
import datetime
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import argparse

import debugpy

torch.classes.load_library(
    "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")

logging.basicConfig(format='%(levelname)-2s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d:%H:%M:%S', level=logging.DEBUG)


class ShmTensorStore:
    _tensor_store = {}

    def __init__(self, name_shape_list):
        for name, shape in name_shape_list:
            self._tensor_store[name] = torch.zeros(shape).share_memory_()
            # self._tensor_store[name] = torch.zeros(shape).share_memory_()

    @classmethod
    def GetTensor(cls, name):
        return cls._tensor_store[name]


class NVGPUCache:
    def __init__(self, capacity, feature_dim) -> None:
        self.gpu_cache = torch.classes.librecstore_pytorch.GpuCache(
            capacity, feature_dim)
        self.feature_dim = feature_dim

    def Query(self, keys, values):
        return self.gpu_cache.Query(keys, values)

    def BatchQuery(self, list_of_keys, list_of_values):
        ret = []
        for each_keys, each_values in zip(list_of_keys, list_of_values):
            ret.append(self.Query(each_keys, each_values))
        return ret

    def Replace(self, keys, values):
        assert values.shape[1] == self.feature_dim
        assert keys.shape[0] == values.shape[0]
        self.gpu_cache.Replace(keys, values)


def worker_main(worker_id, args_config):
    num_workers = args_config['num_workers']
    torch.cuda.set_device(worker_id)

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = num_workers
    torch.distributed.init_process_group(backend=None,
                                         init_method=dist_init_method,
                                         world_size=world_size,
                                         rank=worker_id,
                                         timeout=datetime.timedelta(seconds=4))

    gpu_cache = NVGPUCache(100, 3)

    emb = ShmTensorStore.GetTensor("emb")
    # test shm
    if worker_id == 0:
        print(worker_id, emb)
        for i in range(emb.shape[0]):
            emb[i, :] = torch.ones(emb.shape[1]) * i
    mp.Barrier(num_workers)
    print(worker_id, emb)

    if worker_id == 0:
        # sparse_opt = optim.SparseAdam([emb], lr=1)
        sparse_opt = optim.SGD([emb], lr=1)

    if worker_id == 0:
        input_keys = torch.tensor([0, 1,],).long().cuda()
    else:
        input_keys = torch.tensor([0, 2,],).long().cuda()

    fake_tensor = torch.randn(1, 1, requires_grad=True)

    shard_range = [0, 2, 4]

    # forward

    for _ in range(1):
        # emb_value = LocalCachedEmbedding.apply(
        #     input_keys, "emb", gpu_cache, fake_tensor)

        emb_value = ShardedCachedEmbedding.apply(
            input_keys, "emb", gpu_cache, fake_tensor, shard_range)

        assert emb_value.requires_grad

        loss = emb_value.sum(-1).sum(-1) * 100
        loss.backward()

        if worker_id == 0:
            print("emb grad is ", emb.grad)
            sparse_opt.step()
            sparse_opt.zero_grad()
            print("emb is ", emb)


class TestShardedCache(unittest.TestCase):
    num_workers = 2

    def main_routine(self, routine):
        # wrap rountine with dist_init
        def worker_main(routine, worker_id, num_workers, args):
            torch.cuda.set_device(worker_id)
            dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
                master_ip='127.0.0.1', master_port='12345')
            world_size = num_workers
            torch.distributed.init_process_group(backend=None,
                                                 init_method=dist_init_method,
                                                 world_size=world_size,
                                                 rank=worker_id,
                                                 timeout=datetime.timedelta(seconds=4))
            routine(worker_id, num_workers, None)

        print(f"========== Running Test with routine {routine}==========")

        shm_tensor_store = ShmTensorStore([("emb", (3, 3))])
        workers = []
        for worker_id in range(TestShardedCache.num_workers):
            p = mp.Process(target=worker_main, args=(
                routine, worker_id, TestShardedCache.num_workers, None))
            p.start()
            workers.append(p)

        for each in workers:
            each.join()

    def routine_shm_tensor(self, worker_id, num_workers, args):
        emb = ShmTensorStore.GetTensor("emb")
        if worker_id == 0:
            print(worker_id, emb)
            for i in range(emb.shape[0]):
                emb[i, :] = torch.ones(emb.shape[1]) * i
        mp.Barrier(num_workers)
        for i in range(emb.shape[0]):
            self.assertTrue(torch.allclose(
                emb[i, :], torch.ones(emb.shape[1]) * i))

    @unittest.skip("simple")
    def test_shm_tensor(self):
        self.main_routine(self.routine_shm_tensor)

    def routine_local_cache_helper(self, worker_id, num_workers, args):
        gpu_cache = NVGPUCache(100, 3)
        emb = ShmTensorStore.GetTensor("emb")
        fake_tensor = torch.randn(1, 1, requires_grad=True)
        if worker_id == 0:
            input_keys = torch.tensor([0, 1,],).long().cuda()
        else:
            input_keys = torch.tensor([0, 2,],).long().cuda()

        if worker_id == 0:
            # sparse_opt = optim.SparseAdam([emb], lr=1)
            sparse_opt = optim.SGD([emb], lr=1)


        std_emb = torch.embedding()

        # forward
        for _ in range(1):
            embed_value = LocalCachedEmbedding.apply(
                input_keys, emb, gpu_cache, fake_tensor)
            assert embed_value.requires_grad
            loss = embed_value.sum(-1).sum(-1) * 100
            loss.backward()

            if worker_id == 0:
                print("emb grad is ", emb.grad)
                sparse_opt.step()
                sparse_opt.zero_grad()
                print("emb is ", emb)

    def test_local_cache(self):
        self.main_routine(self.routine_local_cache)


if __name__ == '__main__':
    unittest.main()
    # debugpy.listen(5678)
    # print("wait debugpy connect", flush=True)
    # debugpy.wait_for_client()

    # def get_run_config():
    #     def parse_args(default_run_config):
    #         argparser = argparse.ArgumentParser("Training")
    #         argparser.add_argument('--num_workers', type=int,
    #                                default=2)

    #         return vars(argparser.parse_args())
    #     run_config = {}
    #     run_config.update(parse_args(run_config))
    #     return run_config

    # args = get_run_config()
    # shm_tensor_store = ShmTensorStore([("emb", (3, 3))])

    # workers = []
    # for worker_id in range(args['num_workers']):
    #     p = mp.Process(target=worker_main, args=(worker_id, args))
    #     p.start()
    #     workers.append(p)

    # for each in workers:
    #     each.join()
