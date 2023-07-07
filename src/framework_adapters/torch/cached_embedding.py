import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
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

    def Query(self, keys, values):
        return self.gpu_cache.Query(keys, values)

    def BatchQuery(self, list_of_keys, list_of_values):
        ret = []
        for each_keys, each_values in zip(list_of_keys, list_of_values):
            ret.append(self.Query(each_keys, each_values))
        return ret

    def Replace(self, keys, values):
        assert values.shape[1] == self.feature_dim
        assert keys.shape[0] == values.shape[0]
        self.gpu_cache.Replace(keys, values)


_send_cpu, _recv_cpu = {}, {}

# data: [rank0_data, rank1_data, ...]
# recv_max_shape


def all2all_data_transfer(data, recv_shape, tag, dtype=torch.float):
    rank, world_size = dist.get_rank(), dist.get_world_size()

    if recv_shape is None:
        # first
        # shard_keys_sizes: [shard0_size, shard1_size, ...]
        shard_data_sizes = [torch.tensor(
            [each.size()], device=torch.device("cuda")).long() for each in data]
        per_shard_size = list(torch.empty(
            [world_size], dtype=torch.int64, device=torch.device("cuda")).chunk(world_size))
        dist.all_to_all(per_shard_size, shard_data_sizes,)
        # per_shard_size: [rank0_keys_size_in_mine, rank1_keys_size_in_mine, ....]
        recv_shape = per_shard_size

    msg, res = [None] * world_size, [None] * world_size
    for i in range(1, world_size):
        idx = (rank + i) % world_size
        key = 'dst%d_tag%d' % (idx, tag)
        if key not in _recv_cpu:
            _send_cpu[key] = torch.zeros_like(
                data[idx], dtype=dtype, device='cpu', pin_memory=True)
            _recv_cpu[key] = torch.zeros(
                recv_shape[idx], dtype=dtype, pin_memory=True)
        msg[idx] = _send_cpu[key]
        res[idx] = _recv_cpu[key]

    for i in range(1, world_size):
        left = (rank - i + world_size) % world_size
        right = (rank + i) % world_size
        print("data[right]=", data[right])
        print("msg[right]=", msg[right])
        msg[right].copy_(data[right])
        req = dist.isend(msg[right], dst=right, tag=tag)
        dist.recv(res[left], src=left, tag=tag)
        res[left] = res[left].cuda(non_blocking=True)
        req.wait()

    res[rank] = data[rank]

    return res









class ShardedCachedEmbedding(torch.autograd.Function):
    # shard_range: [0, 4, xxx, 100] len= rank+1
    @staticmethod
    def forward(ctx, keys, emb_name, emb_cache, fake_tensor, shard_range):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        embedding_weight = ShmTensorStore.GetTensor(emb_name)

        emb_dim = embedding_weight.shape[1]

        # 1. all to all keys
        # 1.1 split keys into shards
        sharded_keys = []
        for shard_no in range(len(shard_range)-1):
            start = shard_range[shard_no]
            end = shard_range[shard_no+1]
            shard_keys = keys[(keys >= start) & (keys < end)]
            sharded_keys.append(shard_keys)

        # 1.3 all to all keys with shapes
        recv_keys = all2all_data_transfer(
            sharded_keys, None, tag=123, dtype=keys.dtype)
        print(f"Rank{rank}: keys after a2a", recv_keys, flush=True)

        # 2. search local cache
        query_values = []
        for each in recv_keys:
            query_key_len = each.shape[0]
            # values = torch.zeros((query_key_len, emb_dim)).cuda().requires_grad_()
            values = torch.zeros((query_key_len, emb_dim)).cuda()
            query_values.append(values)

        cache_query_results = emb_cache.BatchQuery(recv_keys, query_values)

        # 3. all to all searched values
        if rank == 0:
            for each in cache_query_results:
                print("each", each)
                print()

        missing_keys = [each.missing_keys for each in cache_query_results]
        missing_indexs = [each.missing_index for each in cache_query_results]

        # 4. all to all missing keys
        missing_keys_in_mine = all2all_data_transfer(
            missing_keys, None, tag=123, dtype=missing_keys[0].dtype)
        missing_indexs_in_mine = all2all_data_transfer(
            missing_indexs, None, tag=124, dtype=missing_indexs[0].dtype)

        # 5. search missing keys
        if rank == 0:
            print("missing_keys_in_mine", missing_keys_in_mine)
            print("missing_indexs_in_mine", missing_indexs_in_mine)

            

        # 6. merge values

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
            keys_gather_list = [torch.zeros_like(
                keys) for _ in range(dist.get_world_size())]
        else:
            keys_gather_list = None
        handle_1 = dist.gather(
            keys, dst=0, gather_list=keys_gather_list, async_op=True)
        # gather grad_output to rank 0

        # grad_output = torch.zeros((100,100)).cuda()

        if dist.get_rank() == 0:
            grad_gather_list = [torch.zeros_like(
                grad_output) for _ in range(dist.get_world_size())]
        else:
            grad_gather_list = None

        handle_2 = dist.gather(grad_output.contiguous(
        ), dst=0, gather_list=grad_gather_list, async_op=True)

        handle_1.wait()
        handle_2.wait()

        if dist.get_rank() == 0:
            assert len(keys_gather_list) == len(grad_gather_list)

            with torch.no_grad():
                embedding_weight = ShmTensorStore.GetTensor(emb_name)
                coo_list = []
                temp = torch.sparse_coo_tensor(
                    [[], []], [], size=embedding_weight.shape)

                for each in range(len(keys_gather_list)):
                    coo_list.append(
                        torch.sparse_coo_tensor(keys_gather_list[each].unsqueeze(0), grad_gather_list[each],
                                                size=embedding_weight.shape)
                    )
                    temp += coo_list[-1].cpu()

                embedding_weight.grad = temp

        return None, None, None, torch.randn(1, 1)


class LocalCachedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, keys, emb_name, emb_cache, fake_tensor):
        embedding_weight = ShmTensorStore.GetTensor(emb_name)

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
            keys_gather_list = [torch.zeros_like(
                keys) for _ in range(dist.get_world_size())]
        else:
            keys_gather_list = None
        handle_1 = dist.gather(
            keys, dst=0, gather_list=keys_gather_list, async_op=True)
        # gather grad_output to rank 0

        # grad_output = torch.zeros((100,100)).cuda()

        if dist.get_rank() == 0:
            grad_gather_list = [torch.zeros_like(
                grad_output) for _ in range(dist.get_world_size())]
        else:
            grad_gather_list = None

        handle_2 = dist.gather(grad_output.contiguous(
        ), dst=0, gather_list=grad_gather_list, async_op=True)

        handle_1.wait()
        handle_2.wait()

        if dist.get_rank() == 0:
            assert len(keys_gather_list) == len(grad_gather_list)

            with torch.no_grad():
                embedding_weight = ShmTensorStore.GetTensor(emb_name)
                coo_list = []
                temp = torch.sparse_coo_tensor(
                    [[], []], [], size=embedding_weight.shape)

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

    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    world_size = num_workers
    # torch.distributed.init_process_group(backend="nccl",
    torch.distributed.init_process_group(backend=None,
                                         init_method=dist_init_method,
                                         world_size=world_size,
                                         rank=worker_id,
                                         timeout=datetime.timedelta(seconds=4))

    # gpu_cache = torch.classes.librecstore_pytorch.GpuCache(100, 3)
    gpu_cache = NVGPUCache(100, 3)

    emb = ShmTensorStore.GetTensor("emb")
    # test shm
    if worker_id == 0:
        print(worker_id, emb)
        emb.copy_(torch.ones(emb.shape))
    mp.Barrier(num_workers)
    print(worker_id, emb)

    if worker_id == 0:
        # sparse_opt = optim.SparseAdam([emb], lr=1)
        sparse_opt = optim.SGD([emb], lr=1)

    if worker_id == 0:
        input_keys = torch.tensor([0, 1,],).long().cuda()
    else:
        input_keys = torch.tensor([0, 2,],).long().cuda()

    fake_tensor = torch.randn(1, 1, requires_grad=True)

    # forward

    for _ in range(3):
        # emb_value = LocalCachedEmbedding.apply(
        #     input_keys, "emb", gpu_cache, fake_tensor)

        emb_value = ShardedCachedEmbedding.apply(
            input_keys, "emb", gpu_cache, fake_tensor, [0, 2, 4])

        assert emb_value.requires_grad

        loss = emb_value.sum(-1).sum(-1) * 100
        loss.backward()
        # if worker_id == 0:
        #     print(emb.grad)
        if worker_id == 0:
            print("emb grad is ", emb.grad)
            sparse_opt.step()
            sparse_opt.zero_grad()
            print("emb is ", emb)


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
