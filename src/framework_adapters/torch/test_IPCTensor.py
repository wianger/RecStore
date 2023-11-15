import torch
import torch as th
import multiprocessing as mp
import tqdm


def worker_main(worker_id, shm_tensor):
    torch.cuda.set_device(worker_id)
    # a = torch.randn((100, 100)).cuda()
    # b = torch.randn((100, 100)).cuda()
    # c = a * b

    for _ in tqdm.trange(1000000):
        shm_tensor[0] = shm_tensor[0]+1
    return

    
workers = []    

nr_workers = 8
shm_tensor = th.tensor([0 for _ in range(nr_workers)], dtype=th.int64).share_memory_()

for worker_id in range(nr_workers):
    p = mp.Process(target=worker_main, 
        args=(worker_id, shm_tensor))
    p.start()
    print(f"Worker {worker_id} pid={p.pid}")
    workers.append(p)

for each in workers:
    each.join()

print(shm_tensor)