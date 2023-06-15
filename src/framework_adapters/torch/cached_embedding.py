import torch
import torch.nn as nn

import torch.multiprocessing as mp
import argparse

torch.classes.load_library("/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")
torch.ops.load_library("/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")

merge_op = torch.ops.librecstore_pytorch.merge_op

class ShmTensorStore:
    _tensor_store = {}
    
    def __init__(self, name_shape_list):
        for name, shape in name_shape_list:
            self._tensor_store[name] = torch.zeros(shape).share_memory_()

    @classmethod
    def GetTensor(cls, name):
        return cls._tensor_store[name]
    
    
class NVGPUCache:
    def __init__(self, capacity, feature_dim) -> None:
        self.gpu_cache = torch.classes.librecstore_pytorch.GpuCache(capacity,feature_dim)
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


def recursive_output(function, module, inputs):
    if type(inputs) is tuple:
        touched_outputs = []
        for output in inputs:
            touched_output = recursive_output(function, module, output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(inputs) is torch.Tensor:
        return function.apply(module, inputs)


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, inputs):
        print(f"Model has attribute {hasattr(module, 'values')}, {inputs[0].requires_grad}") 
        ctx.module = module
        return inputs.detach()

    @staticmethod
    def backward(ctx, *args):
        for each in ctx.module.values.parameters():
            print("------")
            print(each, each.grad, flush=True)
        return (None, ) + args
    

class LocalCachedEmbedding(nn.Module):
    def __init__(self, name, cache_ratio):
        super(LocalCachedEmbedding, self).__init__() 
        
        # self.embedding = ShmTensorStore.GetTensor(name)
        # self.emb_dim = self.embedding.shape[1]

        # slot_num = int(self.embedding.shape[0] * cache_ratio)

        
        # slot_num  = 100

        # self.cache = NVGPUCache(slot_num, self.emb_dim)

        # # self.values = None
        

        self.emb_dim = 32
        
        self.values = None

    def forward(self, keys):

        # return keys
        
        # search cached keys
        query_key_len = keys.shape[0]

        # values = torch.zeros((query_key_len, self.emb_dim)).cuda()

        # values = torch.view

        # self.values= torch.zeros((query_key_len, self.emb_dim)).cuda()
        
        # self.values= torch.randn(query_key_len, self.emb_dim).cuda()
        self.values =  torch.nn.Linear(2,2).cuda()
        
        
        return self.values(keys)
        
        # cache query
        print("self.cache.Query(keys, values)", keys, values)
        cache_query_result = self.cache.Query(keys, values)

        # search missing keys in shared DRAM table
        missing_value = self.embedding[cache_query_result.missing_keys].cuda()

        # join together
        merge_op(values, missing_value, cache_query_result.missing_index)
        

        # values.grad
        return values


def worker_main(worker_id, args_config):

    emb = ShmTensorStore.GetTensor("emb")
    if worker_id == 0:
        print(worker_id, emb)
        emb.copy_(torch.ones(emb.shape))
    mp.Barrier(args_config['num_workers'])
    print(worker_id, emb)
    
    # cached_emb = LocalCachedEmbedding("emb", 0.1)
    cached_emb = torch.nn.Sequential(LocalCachedEmbedding("emb", 0.1)).cuda()
    
    ll = [] 
    def pre_forward_fun(module, inputs):
        print(module.values)
        rc = recursive_output(PostBackwardFunction, module, inputs)
        ll.append(rc)
        return rc

    for child in cached_emb.children():
        child.register_forward_pre_hook(pre_forward_fun)
    
    aaaa = cached_emb.forward(torch.tensor([0., 1.,], requires_grad=True).cuda())

    loss = aaaa.sum(-1).sum(-1) * 100

    print(loss)

    loss.backward()
        

    



def get_run_config():
    def parse_args(default_run_config):
        argparser = argparse.ArgumentParser("Training")
        argparser.add_argument('--num_workers', type=int,
                            default=4)

        return vars(argparser.parse_args())
    run_config = {}
    run_config.update(parse_args(run_config))
    return run_config



if __name__ == '__main__':
    shm_tensor_store = ShmTensorStore([("emb", (3, 3))])
    args = get_run_config()

    workers = []
    for worker_id in range(4):
        p = mp.Process(target=worker_main, args=(worker_id, args))
        p.start()
        workers.append(p)
    
    for each in workers:
        each.join()
    