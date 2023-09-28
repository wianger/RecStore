import torch
torch.ops.load_library(
    "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")
merge_op = torch.ops.librecstore_pytorch.merge_op
uva_cache_query_op = torch.ops.librecstore_pytorch.uva_cache_query_op

# merge_op = None
# uva_cache_query_op = None


torch.classes.load_library(
    "/home/xieminhui/RecStore/build/lib/librecstore_pytorch.so")

GpuCache = torch.classes.librecstore_pytorch.GpuCache