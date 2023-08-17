import os
import sys
import numpy as np
from client import ParameterClient

if __name__ == "__main__":
    pc = ParameterClient("127.0.0.1", 15000, 0, 32)
    keys = np.array([1, 2, 3], dtype=np.uint64)
    right_vals = []
    for i in range(3):
        right_vals.append([])
        for j in range(32):
            right_vals[i].append(i * 32 + j)
    right_values = np.array(right_vals, dtype=np.float32)
    empty_values = np.array([[], [], []])
    values = pc.GetParameter(keys)
    print(values)
    pc.PutParameter(keys, right_values)
    values = pc.GetParameter(keys)
    print(values)
