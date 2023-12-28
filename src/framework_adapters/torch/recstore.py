import torch


def load_recstore_library():
    torch.ops.load_library(
        "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")
    torch.classes.load_library(
        "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")


load_recstore_library()

merge_op = torch.ops.librecstore_pytorch.merge_op
uva_cache_query_op = torch.ops.librecstore_pytorch.uva_cache_query_op
init_folly = torch.ops.librecstore_pytorch.init_folly 


GpuCache = torch.classes.librecstore_pytorch.GpuCache
IPCTensorFactory = torch.classes.librecstore_pytorch.IPCTensorFactory
KGCacheController = torch.classes.librecstore_pytorch.KGCacheController

Mfence = torch.classes.librecstore_pytorch.Mfence

