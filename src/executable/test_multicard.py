import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import datetime




def main():
    def worker_main(worker_id):
        torch.cuda.set_device(worker_id)
        torch.manual_seed(worker_id)
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12335')
        world_size = 8
        rank = worker_id
        torch.distributed.init_process_group(backend=None,
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=100))
        tensor = torch.randn(10000, 64).cuda()
        print(f'Process {rank}/{world_size - 1} generated tensor with shape: {tensor.shape}')

    workers = []
    for worker_id in range(1, 8):
        p = mp.Process(target=worker_main, args=(
            worker_id, ))
        p.start()
        print(f"Worker {worker_id} pid={p.pid}")
        workers.append(p)

    worker_main(0, )

    for each in workers:
        each.join()
        assert each.exitcode == 0
 
 

if __name__ == "__main__":
    main()