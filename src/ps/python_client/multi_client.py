from load_client import client
import multiprocessing
import json

class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

if __name__ == '__main__':
    num_processes = 8  # 定义要启动的进程数
    processes = []
    args = []
    for i in range(num_processes):
        arg = Args()
        with open("./config/config" + str(0) + ".json") as f:
            config = json.load(f)
        arg.nepochs = config['nepochs']
        arg.avg_arrival_rate = config['avg_arrival_rate']
        arg.max_mini_batch_size = config['batch_size']
        arg.avg_mini_batch_size = config['batch_size']
        arg.var_mini_batch_size = config['batch_size']
        arg.sub_task_batch_size = config['sub_task_batch_size']
        arg.embedding_size = config['embedding_size']
        arg.machine = config['machine']
        arg.port = config['port']
        arg.dataset = config['dataset']
        arg.test = config['test']
        arg.batch_size_distribution = "fixed"
        args.append(arg)

    for i in range(num_processes):
        process = multiprocessing.Process(target=client, args=(args[i], ))
        processes.append(process)
        process.start()  # 启动进程

    for process in processes:
        process.join()  # 等待所有进程完成

    print("All processes have finished")