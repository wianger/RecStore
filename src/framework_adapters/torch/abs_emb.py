from abc import ABC


class AbsEmb(ABC):
    def __init__(self):
        raise NotImplementedError

    def forward(self, input_keys):
        raise NotImplementedError

        
    def reg_opt(self, opt):
        raise NotImplementedError

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

