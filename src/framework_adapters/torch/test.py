import torch
torch.classes.load_library("/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")

s = torch.classes.librecstore_pytorch.GpuCache(100,100)
print(s)