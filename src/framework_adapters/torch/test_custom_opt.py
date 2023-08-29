import DistOpt
from DistEmb import DistEmbedding
from DistTensor import DistTensor

from PsKvstore import kvinit

import torch as th

kvinit()

# layer = DistEmbedding(10,10,'test')


def init(shape, dtype): return th.ones(shape, dtype=dtype)


arr = DistTensor((5, 3), dtype=th.float32, name="arr",
                 init_func=init, persistent=True)
assert th.allclose(arr[0:2], th.ones((2, 3), dtype=th.float32))

arr[0:2] = th.ones((2, 3), dtype=th.float32) * 2

assert th.allclose(arr[0:2], th.ones((2, 3), dtype=th.float32)*2)


emb = DistEmbedding(5, 3, name="emb", init_func=init,)

std_emb = th.zeros((5, 3), dtype=th.float32)

for i in range(emb.shape[0]):
    emb.weight[i] = th.ones(emb.shape[1]) * i
    std_emb[i] = th.ones(emb.shape[1]) * i


optimizer = DistOpt.SparseSGD([emb], lr=1)

with th.no_grad():
    print(emb(th.arange(0, 5)))


nids = th.LongTensor([1, 3])
for _ in range(3):
    print(f"========== Step {_} ========== ")
    optimizer.zero_grad()

    feats = emb(nids)
    loss = feats.sum(-1).sum(-1)
    loss.backward()

    # print("_trace", emb._trace)
    optimizer.step()

    with th.no_grad():
        print(emb(th.arange(0, 5)))
