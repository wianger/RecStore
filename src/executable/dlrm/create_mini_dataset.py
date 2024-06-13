import gzip
import torch
import numpy as np





def dump_mini(filename):
	indices = np.load(filename + ".npy")
	offsets = np.load(filename + "_offsets.npy")
	labels = np.load(filename + "_label.npy")

	dataset_size = len(offsets) - 1

	mini_dataset_size = 100000

	offsets = offsets[:mini_dataset_size + 1]
	indices = indices[:offsets[-1]]
	labels = labels[:mini_dataset_size]

	np.save(filename + "_mini.npy", indices)
	np.save(filename + "_mini_offsets.npy", offsets)
	np.save(filename + "_mini_label.npy", labels)


filename= '/home/xieminhui/RecStore/datasets/criteo_binary'
dump_mini(filename)

filename= '/home/xieminhui/RecStore/datasets/avazu_binary'
dump_mini(filename)