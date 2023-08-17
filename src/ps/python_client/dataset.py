import gzip
import torch
import numpy as np

class DatasetLoader:
    def __init__(self, filename: str, test: bool) -> None:
        # self.offsets = 1024
        # return
        print("Dataset: ", filename)
        if(test):
            indices, offsets, lengths = torch.load(filename)
        else:
            with gzip.open(filename) as f:
                indices, offsets, lengths = torch.load(f)

        print("Dataset loaded")
        self.indices = indices
        self.offsets = offsets
        self.lengths = lengths
        self.tot_datasets = offsets.size()[0] - 1
        self.index = 0

    def get(self, batch_size = 1):
        # return torch.tensor([1] * 10)
        index_begin = self.index
        index_end = min(self.index + batch_size, self.tot_datasets)
        self.index = index_end
        self.index %= self.tot_datasets
        return self.indices[self.offsets[index_begin]:self.offsets[index_end]]

if __name__ == '__main__':
    loader = DatasetLoader('./fbgemm_t856_bs65536_0.pt.gz')
    print(loader.get())
    print(loader.get())
    print(loader.get())
    print(loader.get())
    print(loader.get(3))
