import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse

torch.classes.load_library(
    "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")
torch.ops.load_library(
    "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")

merge_op = torch.ops.librecstore_pytorch.merge_op


class ShmTensorStore:
    _tensor_store = {}

    def __init__(self, name_shape_list):
        for name, shape in name_shape_list:
            self._tensor_store[name] = torch.zeros(shape).share_memory_()
            # self._tensor_store[name] = torch.zeros(shape).share_memory_()

    @classmethod
    def GetTensor(cls, name):
        return cls._tensor_store[name]


class NVGPUCache:
    def __init__(self, capacity, feature_dim) -> None:
        self.gpu_cache = torch.classes.librecstore_pytorch.GpuCache(
            capacity, feature_dim)
        self.feature_dim = feature_dim

    def Query(self, keys):
        return self.gpu_cache.Query(keys)

    def Replace(self, keys, values):
        assert values.shape[1] == self.feature_dim
        assert keys.shape[0] == values.shape[0]
        self.gpu_cache.Replace(keys, values)


class ShardedCachedEmbedding:
    def __init__(self, embedding, cache_ratio):
        self.embedding = embedding

    def forward(self, keys):
        # all to all keys
        pass

        # search cached

        # all to all cached values

        # search missing keys in shared DRAM table

        # join together

        # return value

    def __call__(self, x):
        return self.cache


class LocalCachedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, keys, emb_name, emb_cache, fake_tensor):
        embedding_weight = ShmTensorStore.GetTensor(emb_name)
        # embedding = nn.Embedding(*embedding_weight.shape, _weight=embedding_weight, sparse=True)

        emb_dim = embedding_weight.shape[1]
        # search cached keys
        query_key_len = keys.shape[0]

        values = torch.zeros((query_key_len, emb_dim)).cuda().requires_grad_()

        # cache query
        # print("self.cache.Query(keys, values)", keys, values)
        cache_query_result = emb_cache.Query(keys, values)

        # search missing keys in shared DRAM table
        # missing_value = embedding_weight[cache_query_result.missing_keys].cuda()
        missing_value = F.embedding(cache_query_result.missing_keys.cpu(
        ), embedding_weight, sparse=True, padding_idx=None, scale_grad_by_freq=False,)
        missing_value = missing_value.cuda()

        # join together
        merge_op(values, missing_value, cache_query_result.missing_index)

        ctx.save_for_backward(keys,)
        ctx.emb_name = emb_name
        ctx.emb_dim = emb_dim

        print("values", values)
        return values

    @staticmethod
    def backward(ctx, grad_output):
        print("grad_output", grad_output, flush=True)
        keys, = ctx.saved_tensors
        emb_name = ctx.emb_name
        emb_dim = ctx.emb_dim

        assert keys.shape[0] == grad_output.shape[0]
        assert emb_dim == grad_output.shape[1]

        # gather keys to rank 0
        if dist.get_rank() == 0:
            keys_gather_list = [torch.zeros_like(keys) for _ in range(dist.get_world_size())]
        else:
            keys_gather_list = None
        handle_1 = dist.gather(keys, dst=0, gather_list=keys_gather_list, async_op=True)
        # gather grad_output to rank 0
        
        
        # grad_output = torch.zeros((100,100)).cuda()
        
        if dist.get_rank() == 0:
            grad_gather_list = [torch.zeros_like(grad_output) for _ in range(dist.get_world_size())]
        else:
            grad_gather_list = None
        
        handle_2 = dist.gather(grad_output.contiguous(), dst=0, gather_list=grad_gather_list, async_op=True)

        handle_1.wait()
        handle_2.wait()

        if dist.get_rank() == 0:
            assert len(keys_gather_list) == len(grad_gather_list)

            with torch.no_grad():
                embedding_weight = ShmTensorStore.GetTensor(emb_name)
                coo_list = []
                temp = torch.sparse_coo_tensor([[], []], [], size=embedding_weight.shape)

                for each in range(len(keys_gather_list)):
                    coo_list.append(
                        torch.sparse_coo_tensor(keys_gather_list[each].unsqueeze(0), grad_gather_list[each],
                                                size=embedding_weight.shape)
                    )
                    temp += coo_list[-1].cpu()
                
                embedding_weight.grad = temp

        return None, None, None, torch.randn(1, 1)


def worker_main(worker_id, args_config):
    num_workers = args_config['num_workers']
    torch.cuda.set_device(worker_id)

    # if num_workers > 1:

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = num_workers
    torch.distributed.init_process_group(backend="nccl",
                                            init_method=dist_init_method,
                                            world_size=world_size,
                                            rank=worker_id,
                                            timeout=datetime.timedelta(seconds=4))

    gpu_cache = torch.classes.librecstore_pytorch.GpuCache(100, 3)

    emb = ShmTensorStore.GetTensor("emb")
    if worker_id == 0:
        print(worker_id, emb)
        emb.copy_(torch.ones(emb.shape))
    mp.Barrier(num_workers)
    print(worker_id, emb)

    if worker_id == 0:
        input_keys = torch.tensor([0, 1,],).long().cuda()
    else:
        input_keys = torch.tensor([0, 2,],).long().cuda()

    fake_tensor = torch.randn(1, 1, requires_grad=True)

    emb_value = LocalCachedEmbedding.apply(
        input_keys, "emb", gpu_cache, fake_tensor)
    assert emb_value.requires_grad

    loss = emb_value.sum(-1).sum(-1) * 100
    loss.backward()

    if worker_id == 0:
        print(emb.grad)


def get_run_config():
    def parse_args(default_run_config):
        argparser = argparse.ArgumentParser("Training")
        argparser.add_argument('--num_workers', type=int,
                               default=2)

        return vars(argparser.parse_args())
    run_config = {}
    run_config.update(parse_args(run_config))
    return run_config


if __name__ == '__main__':
    args = get_run_config()

    shm_tensor_store = ShmTensorStore([("emb", (3, 3))])

    workers = []
    for worker_id in range(args['num_workers']):
        p = mp.Process(target=worker_main, args=(worker_id, args))
        p.start()
        workers.append(p)

    for each in workers:
        each.join()
