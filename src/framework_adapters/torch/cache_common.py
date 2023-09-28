import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
import logging


from abc import ABC
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from DistEmb import DistEmbedding


class CacheShardingPolicy:
    @staticmethod
    def generate_cached_range(emb, cache_ratio):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        whole_capacity = emb.shape[0]
        cache_capacity = int(whole_capacity * cache_ratio)
        per_shard_cachesize = (cache_capacity + world_size-1) // world_size
        cached_range = []
        for i in range(world_size):
            start = i * per_shard_cachesize
            end = min((i+1) * per_shard_cachesize, cache_capacity)
            cached_range.append((start, end))
        return cached_range


class AbsEmb(ABC):
    def __init__(self):
        raise NotImplementedError

    def forward(self, input_keys, trace=True):
        raise NotImplementedError

    def reg_opt(self, opt):
        raise NotImplementedError


class TorchNativeStdEmbDDP(AbsEmb):
    def __init__(self, emb, device):
        # this standard embedding will clone (deep copy) the embedding variable <emb>
        worker_id = dist.get_rank()
        self.device = device

        if type(emb) is DistEmbedding:
            weight = emb.weight.to_dense_tensor()
        else:
            weight = emb

            
        print("weight.shape", weight.shape)
        
        if device == 'cuda':
            std_emb = nn.Embedding.from_pretrained(weight, freeze=False).cuda()
            self.std_emb_ddp = DDP(std_emb, device_ids=[worker_id],)
        elif device == 'cpu':
            std_emb = nn.Embedding.from_pretrained(weight, freeze=False)
            self.std_emb_ddp = DDP(std_emb, device_ids=None,)
        else:
            assert False

    def forward(self, input_keys):
        if self.device == 'cpu':
            return self.std_emb_ddp(input_keys.cpu())
        elif self.device == 'cuda':
            return self.std_emb_ddp(input_keys.cuda())
        else:
            assert False

    def reg_opt(self, opt):
        opt.add_param_group({"params": self.std_emb_ddp.parameters()})



class TorchNativeStdEmb(AbsEmb):
    def __init__(self, emb, device):
        # this standard embedding will clone (deep copy) the embedding variable <emb>
        worker_id = dist.get_rank()
        self.device = device

        if type(emb) is DistEmbedding:
            weight = emb.weight.to_dense_tensor()
        else:
            weight = emb

            
        print("weight.shape", weight.shape)
        
        if device == 'cuda':
            std_emb = nn.Embedding.from_pretrained(weight, freeze=False).cuda()
            self.std_emb = std_emb
        elif device == 'cpu':
            std_emb = nn.Embedding.from_pretrained(weight, freeze=False)
            self.std_emb = std_emb
        else:
            assert False

    def forward(self, input_keys):
        if self.device == 'cpu':
            return self.std_emb(input_keys.cpu()).cuda()
        elif self.device == 'cuda':
            return self.std_emb(input_keys.cuda()).cuda()
        else:
            assert False

    def reg_opt(self, opt):
        opt.add_param_group({"params": self.std_emb.parameters()})


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


class ShmTensorStore:
    _tensor_store = {}

    # def __init__(self, name_shape_list):
    #     for name, shape in name_shape_list:
    #         self._tensor_store[name] = torch.zeros(shape).share_memory_()

    @classmethod
    def GetTensor(cls, name):
        if name in cls._tensor_store.keys():
            return cls._tensor_store[name]
        else:
            return None

    @classmethod
    def RegTensor(cls, name, shape):
        cls._tensor_store[name] = torch.zeros(shape).share_memory_()