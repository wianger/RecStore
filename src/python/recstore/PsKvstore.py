from operator import itemgetter
import torch as th
from abc import ABC
import torch.nn.functional as F

import recstore
from .utils import XLOG


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
        ShmKVStore.tensor_store = self.tensor_store
        if hasattr(ShmKVStore, "instance_inited"):
            assert not ShmKVStore.instance_inited
        ShmKVStore.instance_inited = True

    # only can be called by the master process (before fork)
    def init_data(self, name, shape, dtype, part_policy=None, init_func=None, is_gdata=None):
        assert name not in self.tensor_store.keys()

        temp = recstore.IPCTensorFactory.NewIPCTensor(
            name, shape, th.float32)
        if temp is None:
            temp = recstore.IPCTensorFactory.FindIPCTensorFromName(name)
            assert temp.shape == shape
            assert temp.dtype == th.float32
            assert temp.is_cpu
            XLOG.debug(
                f"rank{th.distributed.get_rank()}: NewIPCTensor failed, already exists, {hex(temp.data_ptr())}")

        if init_func is not None:
            init_func(temp, shape, th.float32)

        # temp = init_func(
        #     shape=shape, dtype=dtype).share_memory_()

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
        if name in self.tensor_store.keys():
            del self.tensor_store[name]
        else:
            XLOG.error("Delete tensor failed, not found")

    @staticmethod
    def GetUVAMap(tensor):
        temp = tensor
        cudart = th.cuda.cudart()
        r = cudart.cudaHostRegister(
            temp.data_ptr(), temp.numel() * temp.element_size(), 0)
        print(f"cudaHostRegister {hex(temp.data_ptr())}")
        return r

    def GetRowTensor(self, name):
        # XLOG.debug(
        #     f"get dict, {name}.data_ptr={hex(self.tensor_store[name].data_ptr())}")
        return self.tensor_store[name]


KVSTORE = None


def kvinit():
    global KVSTORE
    if KVSTORE is None:
        KVSTORE = ShmKVStore()


def get_kvstore():
    return KVSTORE
