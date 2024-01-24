import gzip
import torch
import numpy as np

class DatasetLoader:
    def __init__(self, filename: str, full: bool, world_size: int, rank: int) -> None:
        print("Dataset: ", filename)
        indices = np.load(filename + ".npy")
        offsets = np.load(filename + "_offsets.npy")
        print("Dataset loaded")
        self.indices = indices
        self.offsets = offsets
        self.dataset_size = len(offsets) - 1
        self.index = 0
        avg_size = (self.dataset_size + world_size - 1) // world_size
        self.local_batch_size = min(avg_size, self.dataset_size - avg_size * rank)
        self.rank_offset = avg_size * rank
        self.rank = rank

    def get(self, batch_size = 1):
        # return torch.tensor([1] * 10)
        index_begin = self.index
        index_end = min(self.index + batch_size, self.local_batch_size)
        self.index = index_end
        self.index %= self.local_batch_size
        index_begin += self.rank_offset
        index_end += self.rank_offset
        idxs = self.indices[self.offsets[index_begin]:self.offsets[index_end]]
        return torch.tensor(idxs)

if __name__ == '__main__':
    loader = DatasetLoader('/dev/shm/criteo_binary', False, 2, 0)
    print(loader.get(1))
    loader2 = DatasetLoader('/dev/shm/criteo_binary', False, 2, 1)
    print(loader2.get(1))
