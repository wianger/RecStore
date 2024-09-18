import os
from queue import Queue
import queue
from threading import Thread
import json
import time


import torch as th
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.rpc as rpc

import recstore
from .cache import CacheEmbFactory
from .utils import XLOG, GPUTimer, Timer


os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'


class CircleBuffer:
    def __init__(self, L, rank, backmode) -> None:
        self.L = L
        self.rank = rank

        self.buffer = []
        for i in range(L):
            sliced_id_tensor = recstore.IPCTensorFactory.NewSlicedIPCTensor(
                f"cached_sampler_r{rank}_{i}", (int(1e6), ), th.int64, )
            self.buffer.append(sliced_id_tensor)

        self.step_tensor = recstore.IPCTensorFactory.NewIPCTensor(
            f"step_r{rank}", (int(L), ), th.int64, )

        self.circle_buffer_end = recstore.IPCTensorFactory.NewIPCTensor(
            f"circle_buffer_end_r{rank}", (int(1), ), th.int64, )

        self.circle_buffer_old_end = recstore.IPCTensorFactory.NewIPCTensor(
            f"circle_buffer_end_cppseen_r{rank}", (int(1), ), th.int64, )

        # [start, end)
        self.start = 0
        self.end = 0

        self.circle_buffer_end[0] = 0
        self.circle_buffer_old_end[0] = 0

        self.backmode = backmode

        self.async_process = self.backmode == "CppAsync" or self.backmode == "CppAsyncV2"

    def push(self, step, item, sync=False):
        assert item.ndim == 1
        assert item.shape[0] < 1e6

        # self.buffer[self.end].Copy_(item, non_blocking=True)
        self.buffer[self.end].Copy_(item, non_blocking=False)
        self.step_tensor[self.end] = step

        self.end = (self.end + 1) % self.L
        self.circle_buffer_end[0] = self.end

        if self.async_process and sync:
            # NOTE: 没有确保cpp端此时一定会消费到， 但确保CPP端不会漏
            # DetectNewSamplesCome
            debug_count = 0
            while (self.circle_buffer_end[0] != self.circle_buffer_old_end[0]):
                debug_count += 1
                if debug_count % 100000 == 0:
                    XLOG.debug("polling")
        else:
            pass

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


class BasePerfSampler:
    def __init__(self, rank, L, num_ids_per_step, backmode) -> None:
        self.rank = rank
        self.L = L
        self.ids_circle_buffer = CircleBuffer(L, rank, backmode)
        self.sampler_iter_num = 0
        self.num_ids_per_step = num_ids_per_step
        self.samples_queue = []
        self.backmode = backmode

    def gen_next_sample(self):
        raise NotImplementedError

    def Prefill(self):
        for _ in range(self.L):
            entity_id = self.gen_next_sample()
            self.samples_queue.append(
                (self.sampler_iter_num, entity_id))
            self.ids_circle_buffer.push(
                self.sampler_iter_num, entity_id, sync=True)
            self.sampler_iter_num += 1

    def __next__(self):
        entity_id = self.gen_next_sample()
        self.samples_queue.append(
            (self.sampler_iter_num, entity_id))

        timer_CircleBuffer = Timer("GenInput:CircleBuffer")

        # self.ids_circle_buffer.push(
        #     self.sampler_iter_num, entity_id, sync=True)

        # BUG may occur if we don't sync here
        self.ids_circle_buffer.push(
            self.sampler_iter_num, entity_id, sync=False)

        timer_CircleBuffer.stop()

        self.sampler_iter_num += 1
        _, entity_id = self.samples_queue.pop(0)
        return entity_id


class TestPerfSampler(BasePerfSampler):
    def __init__(self, rank, L, num_ids_per_step, full_emb_capacity, backmode) -> None:
        super().__init__(rank, L, num_ids_per_step, backmode)
        self.full_emb_capacity = full_emb_capacity

    def gen_next_sample(self):
        from test_emb import XMH_DEBUG
        if XMH_DEBUG:
            # if self.rank == 0:
            #     input_keys = th.tensor([0, 1,],).long().cuda()
            #     # input_keys = th.tensor([0, 1, 2],).long().cuda()
            # else:
            #     input_keys = th.tensor([1, 2,],).long().cuda()
            #     # input_keys = th.tensor([3, 4, 5],).long().cuda()
            # return input_keys

            entity_id = th.randint(self.full_emb_capacity, size=(
                self.num_ids_per_step,)).long().cuda()
            return entity_id
        else:
            entity_id = th.randint(self.full_emb_capacity, size=(
                self.num_ids_per_step,)).long().cuda()
            return entity_id


class GraphCachedSampler:
    @staticmethod
    def BatchCreateCachedSamplers(L, samplers, backmode):
        ret = []
        for i in range(len(samplers)):
            ret.append(GraphCachedSampler(i, L, samplers[i], backmode))
        return ret

    def __init__(self, rank, L, dgl_sampler, backmode) -> None:
        self.rank = rank
        self.L = L
        self.sampler = dgl_sampler
        self.sampler_iter_num = 0

        self.graph_samples_queue = []

        self.ids_circle_buffer = CircleBuffer(L, rank, backmode)

    def Prefill(self):
        # Prefill L samples
        for _ in range(self.L):
            pos_g, neg_g = next(self.sampler)
            self.graph_samples_queue.append(
                (self.sampler_iter_num, pos_g, neg_g))
            self.CopyID(self.sampler_iter_num, pos_g, neg_g)
            self.sampler_iter_num += 1

            if neg_g.neg_head:
                neg_nids = neg_g.ndata['id'][neg_g.head_nid]
            else:
                neg_nids = neg_g.ndata['id'][neg_g.tail_nid]

            if self.rank == 0:
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


def GetKGCacheControllerWrapper():
    assert KGCacheControllerWrapper.instance is not None
    return KGCacheControllerWrapper.instance


class KGCacheControllerWrapperBase:
    @classmethod
    def BeforeDDPInit(cls):
        th.set_num_threads(8)
        recstore.IPCTensorFactory.ClearIPCMemory()
        recstore.MultiProcessBarrierFactory.ClearIPCMemory()

    @classmethod
    def FactoryNew(cls, name, json_str):
        if name == "Dummy":
            return KGCacheControllerWrapperDummy()
        elif name == "RecStore":
            return KGCacheControllerWrapper(json_str)
        else:
            raise ValueError(f"Invalid KGCacheControllerWrapper name: {name}")

    def init(self):
        raise NotImplementedError

    def StopThreads(self):
        raise NotImplementedError

    def BlockToStepN(self,):
        raise NotImplementedError

    def AfterBackward(self,):
        raise NotImplementedError

    def _RegisterFolly(self):
        recstore.init_folly()


class KGCacheControllerWrapperDummy(KGCacheControllerWrapperBase):
    def init(self):
        self.rank = dist.get_rank()
        if self.rank == 0:
            Timer.StartReportThread()

        super()._RegisterFolly()

    def StopThreads(self):
        print("KGCacheControllerWrapperDummy.StopThreads")

    def AfterBackward(self,):
        dist.barrier()


class KGCacheControllerWrapper(KGCacheControllerWrapperBase):
    instance = None

    def __del__(self):
        Timer.StopReportThread()

    def __init__(self, json_str,) -> None:
        self.barrier = recstore.MultiProcessBarrierFactory.Create(
            "kgcachecontroller", dist.get_world_size())
        dist.barrier()

        self.rank = dist.get_rank()
        self.json_config = json.loads(json_str)
        full_emb_capacity = self.json_config['full_emb_capacity']
        print("full_emb_capacity  is ", full_emb_capacity)

        if self.rank == 0:
            Timer.StartReportThread()
            pass

        backmode = self.json_config['backwardMode']

        self.use_cpp_controller = (backmode == "CppSync"
                                   or backmode == "CppAsync"
                                   or backmode == "CppAsyncV2"
                                   )

        if self.use_cpp_controller and self.rank == 0:
            cache_range = CacheEmbFactory.ReturnCachedRange(
                full_emb_capacity, self.json_config)
            self.controller = recstore.KGCacheController.Init(
                json_str, cache_range, full_emb_capacity)
        dist.barrier()
        KGCacheControllerWrapper.instance = self

        self.timer_BlockToStepN = Timer("BlockToStepN")
        self.timer_AfterBackward = Timer("AfterBackward")
        self.timer_BarrierTimeBeforeRank0 = Timer("BarrierTimeBeforeRank0")

        self.__init_rpc()
        super()._RegisterFolly()

    def __init_rpc(self):
        rpc.init_rpc(name=f"worker{self.rank}",
                     rank=self.rank, world_size=dist.get_world_size())
        dist.barrier()

    def init(self):
        dist.barrier()
        if self.use_cpp_controller and self.rank == 0:
            self.controller.RegTensorsPerProcess()
        self.step = 0
        dist.barrier()

    @classmethod
    def StopThreads_cls(cls):
        KGCacheControllerWrapper.instance.StopThreads()

    def GraceFullyStopThreads(self):
        if self.rank == 0 and self.use_cpp_controller:
            print(
                f"On rank0, prepare to call self.controller.StopThreads(), self={self}")
            self.controller.StopThreads()

    def StopThreads(self):
        if self.rank == 0 and self.use_cpp_controller:
            print(
                f"On rank0, prepare to call self.controller.StopThreads(), self={self}")
            self.controller.StopThreads()
        else:
            XLOG.info("call rank0 to StopThreads")
            rpc.rpc_sync(
                "worker0", KGCacheControllerWrapper.StopThreads_cls, args=())
            XLOG.info("call rank0 to StopThreads done")

    def AfterBackward(self,):
        self.timer_BarrierTimeBeforeRank0.start()
        self.barrier.Wait()
        self.timer_BarrierTimeBeforeRank0.stop()

        if self.use_cpp_controller and self.rank == 0:
            self.timer_AfterBackward.start()
            # print(f"rank{self.rank}: before ProcessOneStep, {time.time()}")
            self.controller.ProcessOneStep(self.step)
            # print(f"rank{self.rank}: after ProcessOneStep, {time.time()}")
            self.timer_AfterBackward.stop()

        # self.barrier.Wait()

        self.step += 1

        self.timer_BlockToStepN.start()

        # print(f"rank{self.rank}: before BlockToStepN, {time.time()}")
        if self.use_cpp_controller and self.rank == 0:
            self.controller.BlockToStepN(self.step)
        # print(f"rank{self.rank}: after BlockToStepN, {time.time()}")

        self.barrier.Wait()
        self.timer_BlockToStepN.stop()
