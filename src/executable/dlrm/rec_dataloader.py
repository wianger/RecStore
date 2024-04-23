import gzip
import torch
import numpy as np


class RecDatasetCapacity:
    @classmethod
    def Capacity(cls, filename: str) -> int:
        if filename == 'avazu':
            return 9449205 + 1
        elif filename == 'criteo':
            return 33762577 + 1
        else:
            assert False


class RecDatasetLoader:
    def __init__(self, filename: str, world_size: int, rank: int) -> None:
        print("Dataset: ", filename)
        indices = np.load(filename + ".npy")
        offsets = np.load(filename + "_offsets.npy")
        print("Dataset loaded")
        self.indices = indices
        self.offsets = offsets

        print("np.max = ", np.max(indices))
        print("np.min= ", np.min(indices))

        self.dataset_size = len(offsets) - 1
        self.index = 0
        avg_size = (self.dataset_size + world_size - 1) // world_size
        self.local_batch_size = min(
            avg_size, self.dataset_size - avg_size * rank)
        self.rank_offset = avg_size * rank
        self.rank = rank

    def get(self, batch_size=1):
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
    # loader = RecDatasetLoader('/dev/shm/criteo_binary', 2, 0)
    # print(loader.get(1))

    loader2 = RecDatasetLoader('/dev/shm/criteo_binary', 2, 1)
    print(loader2.get(1))

    # loader = RecDatasetLoader('/dev/shm/avazu_binary', 2, 0)
    # print(loader.get(1))
