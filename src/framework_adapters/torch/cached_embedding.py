import torch
import torch.nn as nn

torch.classes.load_library("/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")
torch.ops.load_library("/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")

merge_op = torch.ops.librecstore_pytorch.merge_op

class ShmTensorStore:
    _tensor_store = {}
    
    def __init__(self, name_shape_list):
        for name, shape in name_shape_list:
            self._tensor_store[name] = torch.zeros(shape).shared_memory_()

    @classmethod
    def GetTensor(cls, name):
        return cls.tensor_store[name]
    
    
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

class LocalCachedEmbedding:
    def __init__(self, name, cache_ratio):
        self.embedding = ShmTensorStore.GetTensor(name)
        slot_num = int(self.embedding.shape[0] * cache_ratio)
        self.emb_dim = self.embedding.shape[1]
        self.cache = NVGPUCache(slot_num, self.emb_dim)

    def forward(self, keys):
        # search cached keys
        query_key_len = keys.shape[0]
        values = torch.zeros((query_key_len, self.emb_dim)).cuda()
        
        cache_query_result = self.cache.Query(keys, values)

        

        cache_query_result.missing_keys
        cache_query_result.missing_index
        cache_query_result.values
        
        # search missing keys in shared DRAM table
        missing_value = self.embedding[cache_query_result.missing_keys].cuda()

        # join together

        
        # return value
        pass



if __name__ == '__main__':
    pass