import torch
import sys
print("torch.cuda.is_available=", torch.cuda.is_available())
print("torch.version.cuda=", torch.version.cuda)
torch.cuda.set_device("cuda:0")

key = torch.Tensor([1,2,3]).long().cuda()
values = torch.range(1,9).float().reshape(3,3).cuda()

torch.classes.load_library("/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")


gpu_cache = torch.classes.librecstore_pytorch.GpuCache(100,3)

print("before query", flush=True)
print(gpu_cache.Query(key))

print("after query", flush=True)

gpu_cache.Replace(key, values)
print("after replace", flush=True)

print(gpu_cache.Query(key))
print("after query", flush=True)