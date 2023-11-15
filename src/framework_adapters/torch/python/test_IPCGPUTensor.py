import torch
import torch as th

import sys
sys.path.append("/home/xieminhui/RecStore/src/framework_adapters/torch")  # nopep8

import recstore 

import multiprocessing as mp
import tqdm




def worker_main(worker_id, barrier, d):
    th.cuda.set_device(worker_id)

    if worker_id == 0:
        shm_tensor_handle= recstore.IPCTensorFactory.NewIPCGPUTensor(0, (1000,), "gpu0_tensor", th.int64)
        shm_tensor = recstore.IPCTensorFactory.GetIPCGPUTensorFromHandleLocal(shm_tensor_handle)

        serialized_str = shm_tensor_handle.__repr__()
        d['shm_tensor_handle'] = serialized_str
        shm_tensor.zero_()
        shm_tensor[:10] = 1
        barrier.wait()
    
    else:
        barrier.wait()
        print(d['shm_tensor_handle'])
        shm_tensor_handle = recstore.IPCGPUMemoryHandle.CreateFromString(d['shm_tensor_handle'])
        shm_tensor = recstore.IPCTensorFactory.GetIPCGPUTensorFromHandle(shm_tensor_handle)
        print(f"in worker{worker_id} process", shm_tensor[:10])
        shm_tensor[:10] = 100
        print(f"in worker{worker_id} process", shm_tensor[:10])
        return

    
workers = []    
nr_workers = 4


barrier = mp.Barrier(nr_workers)


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