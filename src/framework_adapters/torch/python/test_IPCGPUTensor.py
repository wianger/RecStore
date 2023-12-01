import torch
import torch as th

import sys
sys.path.append("/home/xieminhui/RecStore/src/framework_adapters/torch")  # nopep8

import recstore

import torch.multiprocessing as mp
import tqdm


def worker_main(worker_id, barrier, d):
    th.cuda.set_device(0)

    if worker_id == 0:
        shm_tensor = recstore.IPCTensorFactory.NewIPCGPUTensor(
            "gpu0_tensor", (1000,), th.int64, 0)
        shm_tensor.zero_()
        shm_tensor[:10] = 1
        barrier.wait()

    else:
        barrier.wait()
        shm_tensor = recstore.IPCTensorFactory.GetIPCTensorFromName(
            "gpu0_tensor")
        print(f"in worker{worker_id} process", shm_tensor[:10])
        shm_tensor[:10] = 100
        print(f"in worker{worker_id} process", shm_tensor[:10])
        return


workers = []
nr_workers = 2


barrier = mp.Barrier(nr_workers)



recstore.IPCTensorFactory.ClearIPCMemory()
with mp.Manager() as manager:
    d = manager.dict()

    for worker_id in range(nr_workers):
        p = mp.Process(target=worker_main,
                       args=(worker_id, barrier, d))
        p.start()
        print(f"Worker {worker_id} pid={p.pid}")
        workers.append(p)

    for each in workers:
        each.join()

# print("in main process", shm_tensor[:10])
