import torch as th
import time
from client import ParameterClient
def main():
    keys_raw = []
    for i in range(100000):
        keys_raw.append(i)
    keys = th.tensor(keys_raw, dtype=th.int64)
    client = ParameterClient("127.0.0.1", 15000, 0, 32)
    times = 0.0
    cnt = 0
    while(1):
        t1 = time.time()
        values = client.GetParameter(keys)
        t2 = time.time()
        times += t2 - t1
        cnt += 1
        if(cnt == 100):
            print("Average time: ", times / cnt)
            cnt = 0
            times = 0.0

if __name__ == "__main__":
    main()