import torch.distributed as dist
import torch.multiprocessing as mp
import torch
import time


def f(rank):
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', world_size=4, rank=rank)
    t = torch.rand(1)
    gather_t = [torch.ones_like(t)] * dist.get_world_size()
    dist.all_gather(gather_t, t)
    print(rank, t, gather_t)

def test_send_recv(rank, world_size):
    dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:23456', world_size=world_size, rank=rank)
    send_tensor = torch.rand((3,3))
    recv_tensor = torch.rand((4,3))
    if rank == 0:
        dist.send(send_tensor, dst=1) 
    if rank == 1:
        dist.recv(recv_tensor, src=0)
    
    if rank == 1:
        time.sleep(2)
    print(rank, send_tensor, recv_tensor, flush=True)

if __name__ == '__main__':
    # mp.spawn(f, nprocs=4, args=())

    processes=[]
    for i in range(2):
        p = mp.Process(target=test_send_recv, args=(i, 2))
        p.start()
        processes.append(p)
    for each in processes:
        each.join()