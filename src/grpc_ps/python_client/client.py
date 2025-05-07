import os
import sys
import numpy as np
import torch

torch.classes.load_library('/home/frw/workdir/RecStore/build/lib/libgrpc_ps_client_python.so')
PClient = torch.classes.grpc_ps_client_python.PythonParameterClient

# from grpc_ps_client_python import PythonParameterClient as PClient

class ParameterClient:
    def __init__(self, host: str, port: int, shard: int, emb_dim: int) -> None:
        assert(type(host) == str)
        assert(type(port) == int)
        assert(type(emb_dim) == int)
        self.emb_dim = emb_dim
        self.client = PClient(host, port, shard, emb_dim)
    
    def GetParameter(self, keys):
        result = self.client.GetParameter(keys, True)
        return result

    def PutParameter(self, keys, values) -> bool:
        return self.client.PutParameter(keys, values)

    def LoadFakeData(self, key_size: int) -> bool:
        return self.client.LoadFakeData(key_size)