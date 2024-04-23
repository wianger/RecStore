import torch
import torch.optim as optim
import datetime
import torch.multiprocessing as mp


def func(args=None):
    reduced_missing_grads = torch.load('/tmp/grad.pt')
    print(reduced_missing_grads)
    emb = torch.randn((10000, 3))
    sgd = optim.SGD([emb], lr=1)
    emb.grad = reduced_missing_grads
    print(emb[1043, :])
    sgd.step()
    print(emb[1043, :])


def main_routine(routine, args=None):
    # wrap rountine with dist_init
    def worker_main(routine, worker_id, num_workers, args):
        torch.cuda.set_device(worker_id)
        torch.manual_seed(worker_id)
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12545')
        world_size = num_workers
        torch.distributed.init_process_group(backend=None,
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=100000))
        routine()

    workers = []
    num_workers = 2
    for worker_id in range(num_workers):
        p = mp.Process(target=worker_main, args=(
            routine,  worker_id, num_workers, args))
        p.start()
        workers.append(p)

    for each in workers:
        each.join()


main_routine(func)
