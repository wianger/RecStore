import torch
import sys
print("torch.cuda.is_available=", torch.cuda.is_available())
print("torch.version.cuda=", torch.version.cuda)
torch.cuda.set_device("cuda:0")

key = torch.Tensor([1,2,3]).long().cuda()
values = torch.range(1,9).float().reshape(3,3).cuda()
empty_values = torch.zeros_like(values).cuda()

from recstore import GpuCache


def query_result_to_str(query_result):
    return f'values={query_result.values}\nmissing_index={query_result.missing_index}\nmissing_keys={query_result.missing_keys}' 



gpu_cache = GpuCache(100,3)

query_result = gpu_cache.Query(key, empty_values)

assert id(query_result.values) == id(empty_values)

# print(query_result_to_str(query_result))
assert not query_result.values.all()
assert torch.equal(query_result.missing_keys.sort().values, key.sort().values)
assert torch.equal(query_result.missing_index.sort().values, torch.range(0, key.shape[0]-1).cuda())




print("after replace", flush=True)
gpu_cache.Replace(key, values)

for step in range(10):
    with torch.cuda.nvtx.range(f"Step{step}:forward"):
        re_query = torch.Tensor([1, 5, 3]).long().cuda()
        query_result = gpu_cache.Query(re_query, empty_values)

print("after query", flush=True)
print(query_result_to_str(query_result))

# values=tensor([[1., 2., 3.],
#         [0., 0., 0.],
#         [7., 8., 9.]], device='cuda:0')
# missing_index=tensor([1], device='cuda:0')
# missing_keys=tensor([5], device='cuda:0')



from recstore import merge_op

retrived = torch.Tensor([[5, 5, 5]]).cuda()

print(merge_op)
merge_op(query_result.values, retrived, query_result.missing_index)
print(query_result.values)
