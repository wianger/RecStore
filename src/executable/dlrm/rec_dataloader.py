import gzip
import torch
import numpy as np
import tensorflow as tf


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
    def __init__(self, filename: str, world_size: int, rank: int, local_batch_size:int) -> None:
        print("Dataset: ", filename)
        self.indices = np.load(filename + ".npy")
        self.offsets = np.load(filename + "_offsets.npy")
        print("Dataset loaded")
        print("id.max = ", np.max(self.indices))
        print("id.min= ", np.min(self.indices))

        self.dataset_size = len(self.offsets) - 1
        # round down to the nearest multiple of world_size * batch_size
        self.dataset_size = (self.dataset_size // (world_size * local_batch_size)) * world_size * local_batch_size

        self.local_batch_size = local_batch_size
        
        self.rank = rank
        self.world_size = world_size

        self.global_batch_size = self.local_batch_size * self.world_size
        self.index = self.rank * self.local_batch_size

    def get(self):
        index_begin = self.index
        index_end = min(self.index + self.local_batch_size, self.dataset_size)
        self.index = self.index+ self.global_batch_size
        self.index %= self.dataset_size
        idxs = self.indices[self.offsets[index_begin]:self.offsets[index_end]]
        return torch.tensor(idxs)



class RecDatasetLoaderTF:
    
    def __init__(self, filename: str, world_size: int, rank: int, local_batch_size:int) -> None:
        self.indices = np.load(filename + ".npy")
        self.offsets = np.load(filename + "_offsets.npy")
        # self.labels= np.load(filename + "_label.npy")
        self.labels = np.random.randint(0, 2, size=len(self.indices))

        print(f"Dataset{filename} loaded: id.max = ", np.max(self.indices))

        self.dataset_size = len(self.offsets) - 1
        # round down to the nearest multiple of world_size * batch_size
        self.dataset_size = (self.dataset_size // (world_size * local_batch_size)) * world_size * local_batch_size

        self.local_batch_size = local_batch_size
        
        self.rank = rank
        self.world_size = world_size

        self.global_batch_size = self.local_batch_size * self.world_size
        self.index = self.rank * self.local_batch_size


    def __len__(self):
        return self.dataset_size // self.global_batch_size


    def get(self):
        index_begin = self.index
        index_end = min(self.index + self.local_batch_size, self.dataset_size)
        self.index = self.index+ self.global_batch_size
        self.index %= self.dataset_size

        idxs = self.indices[self.offsets[index_begin]:self.offsets[index_end]]
        labels = self.labels[index_begin: index_end]
        
        idxs = idxs.reshape(self.local_batch_size, -1)
        idxs  = tf.convert_to_tensor(idxs, dtype=tf.int64)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        return (None, idxs), labels


if __name__ == '__main__':
    dataloader_t = RecDatasetLoaderTF
    # dataloader_t = RecDatasetLoader

    # loader = dataloader_t('/home/xieminhui/RecStore/datasets/criteo_binary', 1, 0, 1)
    loader = dataloader_t('/home/xieminhui/RecStore/datasets/criteo_binary_mini', 1, 0, 1)
    print(loader.get())
    print(loader.get())

    # loader = dataloader_t('/home/xieminhui/RecStore/datasets/criteo_binary', 2, 0, 2)
    # print(loader.get())

    # loader = dataloader_t('/home/xieminhui/RecStore/datasets/criteo_binary', 2, 1, 2)
    # print(loader.get())


    # loader = RecDatasetLoader('/dev/shm/avazu_binary', 2, 0)
    # print(loader.get(1))
