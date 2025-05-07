from operator import itemgetter
import torch as th
from client import ParameterClient

class KvStore:
    def __init__(self):
        self.client = ParameterClient("127.0.0.1", 15000, 0, 32)
        self.num_servers = 1
        
    def init_data(self, name, shape, dtype, part_policy=None, init_func=None, is_gdata=False):
        nemb, emb_shape = shape
        if isinstance(emb_shape,int):
            emb_shape = (1,emb_shape)
        self.client.PutParameter(th.tensor([name + i for i in range(nemb)]), th.cat([init_func(emb_shape, dtype) for _ in range(nemb)], dim=0))
        # self._DTensors[name] = ([init_func(emb_shape, dtype) for _ in range(nemb)], shape, dtype, part_policy, init_func, is_gdata)

    def data_name_list(self):
        return self._DTensors.keys()

    def get_data_meta(self, name):
        return self._DTensors[name][1:4]
    
    def Get(self, name, id_tensor):
        return self.client.GetParameter(th.tensor([name + i for i in id_tensor]))
    
    def Put(self, name, id_tensor, data_tensor):
        assert(isinstance(id_tensor, list))
        self.client.PutParameter(th.tensor([name + i for i in id_tensor]), data_tensor)
    
    def Delete(self, name):
        # self._DTensors.pop(name)
        pass

KVSTORE = None

def kvinit():
    global KVSTORE
    KVSTORE = KvStore()
def get_kvstore():
    return KVSTORE
