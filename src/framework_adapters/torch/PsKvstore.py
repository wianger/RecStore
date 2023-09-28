from operator import itemgetter
import torch as th
from abc import ABC
import torch.nn.functional as F

class AbsKVStore(ABC):
    def __init__(self):
        raise NotImplementedError

    def init_data(self, name, shape, dtype, part_policy=None, init_func=None, ):
        raise NotImplementedError

    def data_name_list(self):
        raise NotImplementedError

    def get_data_meta(self, name):
        raise NotImplementedError

    def Get(self, name, id_tensor):
        raise NotImplementedError

    def Put(self, name, id_tensor, data_tensor):
        raise NotImplementedError

    def Delete(self, name):
        raise NotImplementedError


# called by client / Tensor
class ShmKVStore(AbsKVStore):
    def __init__(self):
        self.tensor_store = dict()

    # only can be called by the master process (before fork)
    def init_data(self, name, shape, dtype, part_policy=None, init_func=None, is_gdata=None):
        assert name not in self.tensor_store.keys()
        temp = init_func(
            shape=shape, dtype=dtype).share_memory_()
        # Don't use share_memory_().pinned_memory(). It will cause BUG!
        self.tensor_store[name] = temp


    def data_name_list(self):
        return self.tensor_store.keys()

    def get_data_meta(self, name):
        tensor = self.tensor_store[name]
        return tensor.dtype, tensor.shape, None

    def Get(self, name, id_tensor):
        if type(id_tensor) is list:
            id_tensor = th.tensor(id_tensor)
        
        if id_tensor.dtype == th.int64 or id_tensor.dtype == th.int32:
            pass
        else:
            assert False
        
        return F.embedding(id_tensor, self.tensor_store[name])

    def Put(self, name, id_tensor, data_tensor):
        self.tensor_store[name][id_tensor, :] = data_tensor

    def Delete(self, name):
        del self.tensor_store[name]

    def GetUVAMap(self, name):
        temp = self.tensor_store[name]
        cudart = th.cuda.cudart()
        r = cudart.cudaHostRegister(temp.data_ptr(), temp.numel() * temp.element_size(), 0)
        print(f"cudaHostRegister {r}, type(r)={type(r)}")
        return r

    def GetRowTensor(self, name):
        return self.tensor_store[name]


KVSTORE = None


def kvinit():
    global KVSTORE
    if KVSTORE is None:
        KVSTORE = ShmKVStore()


def get_kvstore():
    return KVSTORE
