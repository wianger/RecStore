from contextlib import contextmanager
from pyinstrument import Profiler
import numpy as np
import unittest
import datetime
import time
import argparse
import debugpy
import json
import tqdm

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity

import sys
sys.path.append("/home/xieminhui/RecStore/src/framework_adapters/torch")  # nopep8
import recstore
from recstore import Mfence

from rec_dataloader import RecDatasetCapacity
from cache_common import TorchNativeStdEmbDDP
from controller_process import KGCacheControllerWrapperBase, KGCacheControllerWrapperDummy, RecModelSampler, TestPerfSampler
from cache_emb_factory import CacheEmbFactory
from controller_process import KGCacheControllerWrapper
from sharded_cache import KnownShardedCachedEmbedding, ShardedCachedEmbedding
from local_cache import KnownLocalCachedEmbedding, LocalCachedEmbedding
from DistEmb import DistEmbedding
from PsKvstore import ShmKVStore, kvinit
from utils import XLOG, Timer, GPUTimer, xmh_nvtx_range
import time
import DistOpt


import random
random.seed(0)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)


LR = 0.01
# DIFF_TEST = True
DIFF_TEST = False

from rec_dataloader import RecDatasetLoader
from recstore import BasePerfSampler

class RecModelSampler(BasePerfSampler):
    def __init__(self, rank, L, batch_size, dataset_name, backmode,
                 ) -> None:

        if dataset_name == "criteo":
            nr_ids_one_sample = 26
            self.dataset = RecDatasetLoader(
                "/home/xieminhui/RecStoreDataset/criteo_binary", dist.get_world_size(), rank)
        elif dataset_name == "avazu":
            nr_ids_one_sample = 21
            self.dataset = RecDatasetLoader(
                "/home/xieminhui/RecStoreDataset/avazu_binary", dist.get_world_size(), rank)
        else:
            assert False

        super().__init__(rank, L, batch_size * nr_ids_one_sample, backmode)

        self.dataset_name = dataset_name
        self.rank = rank
        self.batch_size = batch_size

    def gen_next_sample(self):
        sample = self.dataset.get(self.batch_size)
        assert sample.ndim == 1
        assert sample.shape[0] < 1e6
        sample = sample.cuda()
        return sample



def get_run_config():
    def parse_args(default_run_config):
        argparser = argparse.ArgumentParser("Training")
        argparser.add_argument('--num_workers', type=int,
                               default=4)
        argparser.add_argument('--emb_dim', type=int,
                               default=32)
        argparser.add_argument('--L', type=int,
                               default=10)
        argparser.add_argument('--with_perf', type=bool,
                               default=False)
        argparser.add_argument('--batch_size', type=int,
                               default=32)
        argparser.add_argument('--cache_ratio', type=float,
                               default=0.1)
        argparser.add_argument('--log_interval', type=int,
                               default=50)
        argparser.add_argument('--run_steps', type=int,
                               default=500)
        argparser.add_argument('--emb_choice',
                               choices=CacheEmbFactory.SupportedCacheType(),
                               default="KnownLocalCachedEmbedding"
                               )
        argparser.add_argument('--backwardMode', choices=["CppSync",
                                                          "CppAsync",
                                                          "CppAsyncV2",
                                                          "PySync"],
                               default="PySync"
                               )
        argparser.add_argument('--backgrad_init', choices=["cpu",
                                                           "gpu",
                                                           "both",
                                                           ],
                               default="both"
                               )
        argparser.add_argument('--kForwardItersPerStep', type=int,
                               default=1)
        argparser.add_argument('--nr_background_threads', type=int,
                               default=16)

        argparser.add_argument('--dataset', choices=['criteo', 'avazu'],
                               default='criteo')

        argparser.add_argument('--with_nn', type=str, default="256,")
        return vars(argparser.parse_args())

    run_config = {}
    run_config.update(parse_args(run_config))
    run_config['num_embs'] = RecDatasetCapacity.Capacity(run_config['dataset'])

    run_config['with_nn'] = [int(item)
                             for item in run_config['with_nn'].split(',')]
    run_config['with_nn'].insert(0, run_config['emb_dim'])
    return run_config


def init_emb_tensor(emb, worker_id, num_workers):
    if not DIFF_TEST:
        return
    dist.barrier()
    linspace = np.linspace(0, emb.shape[0], num_workers+1, dtype=int)
    if worker_id == 0:
        print(f"rank {worker_id} start initing emb")
        for i in tqdm.trange(linspace[worker_id], linspace[worker_id + 1]):
            emb.weight[i] = torch.ones(emb.shape[1]) * i
    else:
        for i in range(linspace[worker_id], linspace[worker_id + 1]):
            emb.weight[i] = torch.ones(emb.shape[1]) * i
    dist.barrier()
    Mfence.mfence()
    for i in range(1000):
        idx = random.randint(0, emb.shape[0]-1)
        if not (torch.allclose(
                emb.weight[idx], torch.ones(emb.shape[1]) * idx)):
            XLOG.error(
                f"init failed, idx={idx}, emb[idx]={emb.weight[idx]}")
            assert False
    dist.barrier()


def main_routine(args, routine):
    # wrap rountine with dist_init
    def worker_main(routine, worker_id, args):
        torch.cuda.set_device(worker_id)
        torch.manual_seed(worker_id)
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12335')
        world_size = args['num_workers']
        torch.distributed.init_process_group(backend=None,
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=100))
        routine(worker_id, args)

    kvinit()
    emb = DistEmbedding(int(args['num_embs']),
                        int(args['emb_dim']), name="full_emb",)

    XLOG.warn("After init DistEmbedding")

    # dummy LR, only register the tensor state of OSP
    dist_opt = DistOpt.SparseSGD([emb], lr=LR)

    print(f"========== Running Perf with routine {routine}==========")
    workers = []
    for worker_id in range(1, args['num_workers']):
        p = mp.Process(target=worker_main, args=(
            routine, worker_id, args))
        p.start()
        print(f"Worker {worker_id} pid={p.pid}")
        workers.append(p)

    worker_main(routine, 0, args)

    for each in workers:
        each.join()
        assert each.exitcode == 0


class SimpleDLRM(nn.Module):
    def __init__(self, dim_list, sigmoid_layer=-1):
        super(SimpleDLRM, self).__init__()
        layers = nn.ModuleList()
        for i in range(len(dim_list)-1):
            n = dim_list[i]
            m = dim_list[i+1]
            LL = nn.Linear(int(n), int(m), bias=True)
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        self.list = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.list(x)


def routine_local_cache_helper(worker_id, args):
    ShmKVStore.tensor_store.clear()

    USE_SGD = True
    # USE_SGD = False
    rank = dist.get_rank()
    emb = DistEmbedding(int(args['num_embs']),
                        int(args['emb_dim']), name="full_emb",)
    XLOG.debug(
        f"in rank{rank}, full_emb.data_ptr={hex(emb.get_shm_tensor().data_ptr())}")
    dist.barrier()
    # print("begin full_emb.get_shm_tensor().zero_()", flush=True)
    # emb.get_shm_tensor().zero_()
    # print("end full_emb.get_shm_tensor().zero_()",  flush=True)

    init_emb_tensor(emb, worker_id, args['num_workers'])
    # Generate our embedding
    fake_tensor = torch.Tensor([0])
    sparse_opt = optim.SGD(
        [fake_tensor], lr=LR,)
    dist_opt = DistOpt.SparseSGD([emb], lr=LR)
    abs_emb = CacheEmbFactory.New(args["emb_choice"], emb, args)
    abs_emb.reg_opt(sparse_opt)
    dist.barrier()
    # Generate our embedding done

    if DIFF_TEST:
        # Generate standard embedding
        std_emb = TorchNativeStdEmbDDP(emb, device='cuda')
        std_emb.reg_opt(sparse_opt)
        # Generate standard embedding done

    json_str = r'''{{
        "num_gpus": {num_workers},
        "L": {L},
        "kForwardItersPerStep": {kForwardItersPerStep},
        "clr": {lr},
        "backwardMode": "{backwardMode}",
        "nr_background_threads": {nr_background_threads}, 
        "cache_ratio": {cache_ratio},
        "backgrad_init": "{backgrad_init}",
        "update_pq_use_omp": 2,
        "kUpdatePqWorkerNum": 8
    }}'''.format(num_workers=args['num_workers'],
                 kForwardItersPerStep=args['kForwardItersPerStep'],
                 L=args['L'],
                 lr=LR,
                 backwardMode=args['backwardMode'],
                 nr_background_threads=args['nr_background_threads'],
                 cache_ratio=args['cache_ratio'],
                 backgrad_init=args['backgrad_init'],
                 )

    # forward
    start = time.time()
    start_step = 0
    warmup_iters = 50
    # warmup_iters = 0

    if args['with_perf']:
        for_range = tqdm.trange(args['run_steps'])
        with_perf = True
    else:
        for_range = range(args['run_steps'])
        with_perf = False

    with_pyinstrucment = True
    with_cudaPerf = False
    with_torchPerf = False

    if with_perf and with_pyinstrucment:
        pyinstruct_profiler = Profiler()

    if with_perf and with_torchPerf:
        torch_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
        torch_profiler.start()

    perf_sampler = RecModelSampler(rank=rank,
                                   L=args['L'],
                                   batch_size=args['batch_size'],
                                   dataset_name=args['dataset'],
                                   backmode=args['backwardMode'],
                                   )
    print("Construct RecModelSampler done", flush=True)

    if args["emb_choice"] == "KnownLocalCachedEmbedding":
        kg_cache_controller = KGCacheControllerWrapper(
            json_str, emb.shape[0],
        )
    else:
        kg_cache_controller = KGCacheControllerWrapperDummy(
        )

    print("Construct KGCacheControllerWrapper done", flush=True)

    kg_cache_controller.init()

    perf_sampler.Prefill()

    # define NN
    nn_model = SimpleDLRM(args['with_nn'])
    nn_model = nn_model.cuda()
    nn_model = DDP(nn_model, device_ids=[rank])
    sparse_opt.add_param_group({'params': nn_model.parameters()})

    timer_geninput = Timer("GenInput")
    # timer_Forward = GPUTimer("Forward")
    # timer_Backward = GPUTimer("Backward")
    # timer_Optimize = GPUTimer("Optimize")
    # timer_NN = GPUTimer("NN")
    timer_Forward = Timer("Forward")
    timer_Backward = Timer("Backward")
    timer_Optimize = Timer("Optimize")
    timer_NN = Timer("NN")
    timer_onestep = Timer(f"OneStep")

    print("Before Training", flush=True)

    start_barrier = recstore.MultiProcessBarrierFactory.Create(
        "start_barrier", dist.get_world_size())
    start_barrier.Wait()

    loss_fn = torch.nn.MSELoss(reduction="mean")
    device = torch.device(f"cuda:{rank}")

    exp_all_start_time = time.time()
    for _ in for_range:
        if rank == 0 and _ % 10 == 0:
            exp_all_now = time.time()
            if exp_all_now - exp_all_start_time > 60:
                break

        timer_onestep.start()
        sparse_opt.zero_grad()
        dist_opt.zero_grad()

        if rank == 0:
            print(f"========== Step {_} ========== ", flush=True)

        if with_perf and _ == warmup_iters:
            if rank == 0:
                if with_pyinstrucment:
                    pyinstruct_profiler.start()

                if with_cudaPerf:
                    print("cudaProfilerStart")
                    th.cuda.cudart().cudaProfilerStart()

        if with_perf and _ == warmup_iters + 3:
            if rank == 0:
                if with_pyinstrucment:
                    pyinstruct_profiler.stop()
                    pyinstruct_profiler.print()
                if with_cudaPerf:
                    print("cudaProfilerStop")
                    th.cuda.cudart().cudaProfilerStop()
                if with_torchPerf:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace("trace.json")
            break

        timer_geninput.start()
        input_keys = next(perf_sampler)
        start_barrier.Wait()
        timer_geninput.stop()

        timer_Forward.start()
        with xmh_nvtx_range(f"Step{_}:forward", condition=rank == 0 and _ >= warmup_iters and args['with_perf']):
            embed_value = abs_emb.forward(input_keys)

        timer_NN.start()
        reshaped_emb = embed_value.view(
            args['batch_size'], -1, args['emb_dim'])

        sum_pooling = reshaped_emb.sum(1)

        logit = nn_model(sum_pooling)
        loss = loss_fn(logit, torch.randint(
            0, 2, logit.shape, dtype=torch.float32, device=device))
        timer_NN.stop()
        timer_Forward.stop()

        timer_Backward.start()
        loss.backward()
        timer_Backward.stop()

        # print("emb.grad=", abs_emb.std_emb.weight.grad)

        if DIFF_TEST:
            # diff test begin
            std_embed_value = std_emb.forward(input_keys)
            std_loss = std_embed_value.sum(-1).sum(-1)
            std_loss.backward()
            # XLOG.cdebug(
            #     f"{rank}:std_embed_value {std_embed_value}")
            # XLOG.cdebug(
            #     f"{rank}:embed_value {embed_value}")
            # XLOG.cdebug(
            #     f"{rank}:full_emb {abs_emb.full_emb.get_shm_tensor()}")
            assert (torch.allclose(
                embed_value.cpu(), std_embed_value.cpu())), "forward is error"
            assert torch.allclose(loss, std_loss)
            # diff done

        timer_Optimize.start()
        sparse_opt.step()
        dist_opt.step()
        timer_Optimize.stop()

        kg_cache_controller.AfterBackward()

        if (_ % args['log_interval']) == (args['log_interval']-1):
            end = time.time()
            print(
                f"Step{_}:rank{rank}, time: {end-start:.3f}, per_step: {(end-start)/(_-start_step+1):.6f}", flush=True)
            start = time.time()
            start_step = _

            if rank == 0:
                Timer.Report()

        timer_onestep.stop()

    if rank == 0:
        print("Successfully xmh", flush=True)
        kg_cache_controller.StopThreads()


if __name__ == "__main__":
    KGCacheControllerWrapperBase.BeforeDDPInit()

    args = get_run_config()
    main_routine(args, routine_local_cache_helper)

    print("Successfully xmh")
