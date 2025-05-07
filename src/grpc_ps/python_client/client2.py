import os
import sys
import numpy as np

class ParameterClient:
    def __init__(self, host: str, port: int, shard: int, emb_dim: int) -> None:
        assert(type(host) == str)
        assert(type(port) == int)
        assert(type(emb_dim) == int)
        
    def GetParameter(self, keys) -> list:
        # result = self.client.GetParameter(keys, True)
        # return result
        print("Get")
        return [1.1] * 8

    def PutParameter(self, keys, values) -> bool:
        # return self.client.PutParameter(keys, values)
        print("Put")
        return True
