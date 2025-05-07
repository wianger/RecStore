import gzip
import torch
import numpy as np

class DatasetLoader:
    def __init__(self, filename: str, test: bool, table_size: int, batch_size: int) -> None:
        # self.offsets = 1024
        # return
        print("Dataset: ", filename)
        if(test):
            indices, offsets, lengths = torch.load(filename)
        else:
            indices, offsets = torch.load(filename)

        print("Dataset loaded")
        self.indices = indices
        self.offsets = offsets
        self.tot_batch_size = batch_size
        self.table_size = table_size
        self.index = 0

    def get(self, batch_size = 1):
        # return torch.tensor([1] * 10)
        index_begin = self.index
        index_end = min(self.index + batch_size, self.tot_batch_size)
        self.index = index_end
        self.index %= self.tot_batch_size
        idxs = [self.indices[self.offsets[i * self.tot_batch_size + index_begin]:self.offsets[i * self.tot_batch_size + index_end]] for i in range(self.table_size)]
        return torch.cat(idxs)

if __name__ == '__main__':
    loader = DatasetLoader('/dev/shm/2021/fbgemm_t856_bs65536_0.pt', False, 856, 65536)
    print(loader.get(128))
    print(loader.get(128))
    print(loader.get(128))
    print(loader.get(128))
    print(loader.get(128))
