from queue import Queue
from dglke.dataloader.sampler import TrainDataset
from dglke.dataloader import KGDataset, TrainDataset, NewBidirectionalOneShotIterator

import torch as th
import torch.multiprocessing as mp
import queue
from threading import Thread


class ProxySubGraph:
    def __init__(self, ) -> None:
        pass


class CachedSampler:
    def __init__(self, rank, L, sampler, message_q) -> None:
        # self.sample_q = Queue(L)
        self.sample_q = []
        self.rank = rank
        self.L = L
        self.sampler = sampler
        self.message_q = message_q
        self.sampler_iter_num = 0


        # Prefill L samples
        for _ in range(L):
            pos_g, neg_g = next(self.sampler)
            self.sample_q.append((self.sampler_iter_num, pos_g, neg_g))
            self.SendSample(self.sampler_iter_num, pos_g, neg_g)
            self.sampler_iter_num += 1



        # self.fetching_thread = mp.Process(target=self.FetchingThread, args=())
        # self.fetching_thread = Thread(target=self.FetchingThread, args=())
        # self.fetching_thread.start()

    def FetchingThread(self):
        while True:
            pos_g, neg_g = next(self.sampler)
            try:
                # print("FetchingThread Put sample")
                self.sample_q.put_nowait((self.sampler_iter_num, pos_g, neg_g))
                print("FetchingThread Put sample done")
                self.SendSample(self.sampler_iter_num, pos_g, neg_g)
                self.sampler_iter_num += 1
            except queue.Full:
                pass

    @staticmethod
    def BatchCreateCachedSamplers(L, samplers, message_qs):
        assert len(samplers) == len(message_qs)
        ret = []
        for i in range(len(samplers)):
            ret.append(CachedSampler(i, L, samplers[i], message_qs[i]))
        return ret

    def __next__(self):
        pos_g, neg_g = next(self.sampler)

        # try:
        #     pos_g, neg_g = next(self.sampler)
        #     self.sample_q.append((self.sampler_iter_num, pos_g, neg_g))
        #     self.SendSample(self.sampler_iter_num, pos_g, neg_g)
        #     self.sampler_iter_num += 1
        # except StopIteration as e:
        #     pass
        # _, pos_g, neg_g = self.sample_q.pop(0)
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

    def SendSample(self, step, pos_g, neg_g):
        entity_id = th.cat([pos_g.ndata['id'], neg_g.ndata['id']], dim=0)
        entity_id.share_memory_()
        self.message_q.put((step, entity_id))


class ControllerServer:
    def __init__(self, args, embedding_cache):
        self.args = args
        self.L = args.L
        self.manager = mp.Manager()
        self.manager.__enter__()
        
        # queues for sample
        self.async_sample_qs = self.manager.list()
        for i in range(args.nr_gpus):
            self.async_sample_qs.append(self.manager.Queue(100))

        # queues for grad
        self.async_grad_qs = self.manager.list()
        for i in range(args.nr_gpus):
            self.async_grad_qs.append(self.manager.Queue(200))

    def __del__(self):
        for i in range(self.args.nr_gpus):
            self.async_sample_qs[i].put(None)
        self.async_p.join()
        self.manager.__exit__(None, None, None)


    def CreateGradClients(self):
        ret = []
        for each_q in self.async_grad_qs:
            # ret.append(GradClient(each_q))
            ret.append(None)
        return ret


    def GetMessageQueues(self):
        return self.async_sample_qs

    def StartControlProcess(self,):
        self.async_p = mp.Process(
            target=self.ControlProcess, args=())
        self.async_p.start()

    def ControlProcess(self, ):
        assert len(self.async_sample_qs) == self.args.nr_gpus
        while True:
            for each_q in self.async_sample_qs:
                try:
                    ret = each_q.get_nowait()
                    if ret is None:
                        return
                except queue.Empty:
                    pass

    def ProcessGrad(self, name, keys, grads):

        pass


class GradClient:
    def __init__(self, async_q) -> None:
        self.async_q = async_q

    def push(self, name, id_tensor, data_tensor):
        id_tensor.share_memory_()
        data_tensor.share_memory_()
        self.async_q.put((name, id_tensor, data_tensor))
        
    def barrier(self):
        pass
        