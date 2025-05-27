from loadGenerator import loadGenerator
from client import GRPCParameterClient as Client
from dataset import DatasetLoader
import argparse, json

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=0)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--max_mini_batch_size", type=int, default=1)
    parser.add_argument("--avg_mini_batch_size", type=float, default=1)
    parser.add_argument("--var_mini_batch_size", type=float, default=1)
    parser.add_argument("--batch_size_distribution", type=str, default="fixed") # synthetic or dataset
    parser.add_argument("--batch_dist_file", type=str, default="config/batch_distribution.txt") # synthetic or dataset
    parser.add_argument("--sub_task_batch_size", type=int, default=16)
    parser.add_argument("--avg_arrival_rate", type=float, default=10)
    parser.add_argument("--nepochs", type=int, default=1)

    args = parser.parse_args()
    return args  

def client(args):
    client = Client(args.machine, args.port, 0, args.embedding_size)
    dataset = DatasetLoader(args.dataset, args.test, args.table_size, args.batch_size)
    args.num_batches = (dataset.offsets.size()[0] - 1) // args.sub_task_batch_size
    loadGenerator(args, client, dataset)

if __name__ == '__main__':
    args = parse()
    with open(args.config) as f:
        config = json.load(f)
    args.nepochs = config['nepochs']
    args.avg_arrival_rate = config['avg_arrival_rate']
    args.batch_size = config['batch_size']
    args.sub_task_batch_size = config['sub_task_batch_size']
    args.embedding_size = config['embedding_size']
    args.machine = config['machine']
    args.port = config['port']
    args.dataset = config['dataset']
    args.test = config['test']
    args.table_size = config['table_size']
    client(args)
