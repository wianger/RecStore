from Adagrad import SparseAdagrad
from DistEmb import DistEmbedding
from DistTensor import DistTensor
from EmbBag import myEmbeddingBag
from utils import init_prefix_embdim
from PsKvstore import kvinit

import torch as th

init_prefix_embdim(1, 32)
kvinit()

# layer = DistEmbedding(10,10,'test')


init = lambda shape, dtype: th.ones(shape, dtype=dtype)
arr = DistTensor((5, 32), th.float32, init_func=init)
print(arr[0:3])
"""
    tensor([[1, 1],
            [1, 1],
            [1, 1]], dtype=torch.int32)
"""
arr[0:3] = th.ones((3, 32), dtype=th.float32) * 2
print(arr[0:3])
"""
    tensor([[2, 2],
            [2, 2],
            [2, 2]], dtype=torch.int32)
"""
arr2 = DistTensor((5, 32), th.float32, init_func=init)
print(arr2[0:3])

embedding_sum = myEmbeddingBag(6, 32, mode='sum')
data = th.Tensor([[1] * 32,[2] * 32,[3] * 32,[4] * 32,[5] * 32,[6] * 32])
embedding_sum.weight.set_data(data)
input = th.tensor([0,2,3,5,0,2,3,5,0,2,3,5,0,2,3,5,0,2,3,5,0,2,3,5,0,2,3,5,0,2,3,5], dtype=th.long)
off = th.tensor([0,2], dtype=th.long)
wei = th.Tensor([0.1] * 32)
# offsets = th.tensor([0,4], dtype=th.long)
#>>> # xdoctest: +IGNORE_WANT("non-deterministic")
print(embedding_sum(input=input,offsets=off,per_sample_weights=wei))
"""
def initializer(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    #arr.uniform_(-1, 1)
    return arr
emb = DistEmbedding(5, 3, init_func=initializer)
optimizer = SparseAdagrad([emb], lr=0.01)
print(emb(th.arange(0,5)))
emb.reset_trace()
nids = th.LongTensor([1,3])
for blocks in range(1000):
    feats = emb(nids)
    loss = th.mean(th.pow(feats-1,2))
    loss.backward()
    optimizer.step()
print(emb(th.arange(0,5)))
emb.reset_trace()
"""