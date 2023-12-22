from queue import Queue
from dglke.dataloader.sampler import TrainDataset
from dglke.dataloader import KGDataset, TrainDataset, NewBidirectionalOneShotIterator

import torch as th
import torch.multiprocessing as mp
import torch.distributed as dist
import queue
from threading import Thread

import sys
from cache_emb_factory import CacheEmbFactory
sys.path.append("/home/xieminhui/RecStore/src/framework_adapters/torch")  # nopep8
import recstore


class ProxySubGraph:
    def __init__(self, ) -> None:
        pass


class CircleBuffer:
    def __init__(self, L, rank) -> None:
        self.L = L

        self.buffer = []
        for i in range(L):
            sliced_id_tensor = recstore.IPCTensorFactory.NewSlicedIPCTensor(
                f"cached_sampler_r{rank}_{i}", (int(1e5), ), th.int64, )
            self.buffer.append(sliced_id_tensor)

        self.step_tensor = recstore.IPCTensorFactory.NewIPCTensor(
            f"step_r{rank}", (int(L), ), th.int64, )

        self.circle_buffer_end = recstore.IPCTensorFactory.NewIPCTensor(
            f"circle_buffer_end_r{rank}", (int(1), ), th.int64, )
        # [start, end)
        self.start = 0
        self.end = 0

        self.circle_buffer_end[0] = 0

    def push(self, step, item):
        assert item.ndim == 1
        # self.buffer[self.end].Copy_(item, non_blocking=True)
        self.buffer[self.end].Copy_(item, non_blocking=False)
        self.step_tensor[self.end] = step

        self.end = (self.end + 1) % self.L
        if self.end == self.start:
            self.start = (self.start + 1) % self.L

    def pop(self):
        if self.start == self.end:
            return None
        ret = self.buffer[self.start]

        self.start = (self.start + 1) % self.L
        return ret

    def __len__(self):
        return (self.end - self.start + self.L) % self.L


class TestPerfSampler:
    def __init__(self, rank, L, num_ids_per_step, full_emb_capacity) -> None:
        self.rank = rank
        self.L = L
        self.ids_circle_buffer = CircleBuffer(L, rank)
        self.sampler_iter_num = 0
        self.num_ids_per_step = num_ids_per_step
        self.full_emb_capacity = full_emb_capacity

        self.samples_queue = []

        for _ in range(L):
            entity_id = self.gen_next_sample()
            self.samples_queue.append(
                (self.sampler_iter_num, entity_id))
            self.ids_circle_buffer.push(self.sampler_iter_num, entity_id)
            self.sampler_iter_num += 1

    def gen_next_sample(self):
        from test_emb import XMH_DEBUG
        if XMH_DEBUG:
            if self.rank == 0:
                input_keys = th.tensor([1, 2,],).long().cuda()
                # input_keys = torch.tensor([0, 1,],).long().cuda()
            else:
                input_keys = th.tensor([0, 2,],).long().cuda()
                # input_keys = torch.tensor([2, 3,],).long().cuda()
            return input_keys
        else:
            entity_id = th.randint(self.full_emb_capacity, size=(
                self.num_ids_per_step,)).long().cuda()
            return entity_id

    def __next__(self):
        entity_id = self.gen_next_sample()

        self.samples_queue.append(
            (self.sampler_iter_num, entity_id))
        self.ids_circle_buffer.push(self.sampler_iter_num, entity_id)
        self.sampler_iter_num += 1

        _, entity_id = self.samples_queue.pop(0)
        return entity_id


class PerfSampler:
    def __init__(self, rank, L, num_ids_per_step, full_emb_capacity) -> None:
        self.rank = rank
        self.L = L
        self.ids_circle_buffer = CircleBuffer(L, rank)
        self.sampler_iter_num = 0
        self.num_ids_per_step = num_ids_per_step
        self.full_emb_capacity = full_emb_capacity

        self.samples_queue = []

        for _ in range(L):
            entity_id = self.gen_next_sample()
            self.samples_queue.append(
                (self.sampler_iter_num, entity_id))
            self.ids_circle_buffer.push(self.sampler_iter_num, entity_id)
            self.sampler_iter_num += 1

    def gen_next_sample(self):
        entity_id = th.randint(self.full_emb_capacity, size=(
            self.num_ids_per_step,)).long().cuda()
        return entity_id
        if self.rank == 0:
            input_keys = th.tensor([1, 2,],).long().cuda()
        else:
            input_keys = th.tensor([0, 2,],).long().cuda()
        return input_keys

    def __next__(self):
        entity_id = self.gen_next_sample()

        self.samples_queue.append(
            (self.sampler_iter_num, entity_id))
        self.ids_circle_buffer.push(self.sampler_iter_num, entity_id)
        self.sampler_iter_num += 1

        _, entity_id = self.samples_queue.pop(0)
        return entity_id



class CachedSampler:
    @staticmethod
    def BatchCreateCachedSamplers(L, samplers):
        ret = []
        for i in range(len(samplers)):
            ret.append(CachedSampler(i, L, samplers[i],))
        return ret

    def __init__(self, rank, L, dgl_sampler) -> None:
        self.rank = rank
        self.L = L
        self.sampler = dgl_sampler
        self.sampler_iter_num = 0

        self.graph_samples_queue = []

        self.ids_circle_buffer = CircleBuffer(L, rank)

        # Prefill L samples
        for _ in range(L):
            pos_g, neg_g = next(self.sampler)
            self.graph_samples_queue.append(
                (self.sampler_iter_num, pos_g, neg_g))
            self.CopyID(self.sampler_iter_num, pos_g, neg_g)
            self.sampler_iter_num += 1

            if neg_g.neg_head:
                neg_nids = neg_g.ndata['id'][neg_g.head_nid]
            else:
                neg_nids = neg_g.ndata['id'][neg_g.tail_nid]

            if rank == 0:
                print(f"-------Step {_}-------")
                print(pos_g.ndata['id'][:10], neg_nids[:10])

        # self.fetching_thread = mp.Process(target=self.FetchingThread, args=())
        # self.fetching_thread = Thread(target=self.FetchingThread, args=())
        # self.fetching_thread.start()

    # def FetchingThread(self):
    #     while True:
    #         pos_g, neg_g = next(self.sampler)
    #         try:
    #             # print("FetchingThread Put sample")
    #             self.graph_samples_queue.append(
    #                 (self.sampler_iter_num, pos_g, neg_g))
    #             print("FetchingThread Put sample done")
    #             self.CopyID(self.sampler_iter_num, pos_g, neg_g)
    #             self.sampler_iter_num += 1
    #         except queue.Full:
    #             pass

    def __next__(self):
        # pos_g, neg_g = next(self.sampler)

        try:
            pos_g, neg_g = next(self.sampler)
            self.graph_samples_queue.append(
                (self.sampler_iter_num, pos_g, neg_g))
            self.CopyID(self.sampler_iter_num, pos_g, neg_g)
            self.sampler_iter_num += 1
        except StopIteration as e:
            pass
        _, pos_g, neg_g = self.graph_samples_queue.pop(0)
        return pos_g, neg_g

        # while True:
        #     try:
        #         # print("Consumer: get from queue")
        #         iter_num, pos_g, neg_g = self.sample_q.get_nowait()
        #         print(f"Consumer {self.rank}: get from queue done")
        #         break
        #     except queue.Empty:
        #         pass
        # return pos_g, neg_g

    def CopyID(self, step, pos_g, neg_g):
        if neg_g.neg_head:
            neg_nids = neg_g.ndata['id'][neg_g.head_nid]
        else:
            neg_nids = neg_g.ndata['id'][neg_g.tail_nid]

        entity_id = th.cat([pos_g.ndata['id'], neg_nids], dim=0)
        self.ids_circle_buffer.push(step, entity_id)


class KGCacheControllerWrapper:
    def __init__(self, json_str, emb, args) -> None:
        dist.barrier()
        self.rank = dist.get_rank()
        self.args = args
        if (self.args['BackwardMode'] == "CppSync"
                or self.args['BackwardMode'] == "CppAsync"
                ) and self.rank == 0:
            cache_range = CacheEmbFactory.ReturnCachedRange(emb, args)
            self.controller = recstore.KGCacheController.Init(
                json_str, cache_range)
        dist.barrier()

    def init(self):
        import time
        time.sleep(5)

        if (self.args['BackwardMode'] == "CppSync"
                or self.args['BackwardMode'] == "CppAsync"
                ) and self.rank == 0:
            self.controller.RegTensorsPerProcess()
        self.step = 0
        dist.barrier()


    def OnNextStep(self,):
        if (self.args['BackwardMode'] == "CppSync"
                or self.args['BackwardMode'] == "CppAsync"
                )  and self.rank == 0:
            self.controller.BlockToStepN(self.step)
        dist.barrier()
        self.step += 1

    def AfterBackward(self,):
        dist.barrier()
        if self.args['BackwardMode'] == "CppSync" and self.rank == 0:
            self.controller.ProcessOneStep()
        dist.barrier()
