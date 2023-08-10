from abc import ABC
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class AbsEmb(ABC):
    def __init__(self):
        raise NotImplementedError

    def forward(self, input_keys):
        raise NotImplementedError

    def reg_opt(self, opt):
        raise NotImplementedError


class TorchNativeStdEmb(AbsEmb):
    def __init__(self, emb):
        # this standard embedding will clone (deep copy) the embedding variable <emb>
        std_emb = nn.Embedding.from_pretrained(emb, freeze=False).cuda()
        worker_id = dist.get_rank()
        self.std_emb_ddp = DDP(std_emb, device_ids=[worker_id],)

    def forward(self, input_keys):
        return self.std_emb_ddp(input_keys)

    def reg_opt(self, opt):
        opt.add_param_group({"params": self.std_emb_ddp.parameters()})


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

    def __init__(self, name_shape_list):
        for name, shape in name_shape_list:
            self._tensor_store[name] = torch.zeros(shape).share_memory_()
            # self._tensor_store[name] = torch.zeros(shape).share_memory_()

    @classmethod
    def GetTensor(cls, name):
        return cls._tensor_store[name]
