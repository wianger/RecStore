import os
import sys
import numpy as np
import torch

torch.classes.load_library(
    '/home/xieminhui/RecStore/build/lib/libgrpc_ps_client_python.so')
RecStoreGRPCPSClient = torch.classes.grpc_ps_client_python.GRPCParameterClientTorch


class GRPCParameterClient:
    def __init__(self, host: str, port: int, shard: int) -> None:
        assert (type(host) == str)
        assert (type(port) == int)
        self.client = RecStoreGRPCPSClient(host, port, shard)

    def GetParameter(self, keys, emb_dim) -> torch.Tensor:
        result = self.client.GetParameter(keys, emb_dim, True)
        return result

    def PutParameter(self, keys, values) -> bool:
        return self.client.PutParameter(keys, values)

    def LoadFakeData(self, key_size: int) -> bool:
        return self.client.LoadFakeData(key_size)
