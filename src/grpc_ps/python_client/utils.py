import torch as th

global prefix
global emb_dim

def init_prefix_embdim(num: int, dim: int):
    global prefix
    global emb_dim
    prefix = num * (2 ** 30)
    emb_dim = dim

def get_role():
    global prefix
    return prefix

def get_emb_dim():
    global emb_dim
    return emb_dim

def toindex(idx):
    if isinstance(idx,list):
        return idx
    elif isinstance(idx,th.Tensor):
        assert(len(idx.shape) == 1)
        assert(idx.dtype == th.int64)
        return list(idx.detach().cpu().numpy())
    elif isinstance(idx,slice):
        return [i for i in range(idx.start,idx.stop)]
    else:
        return [idx]

def toTensor(x, device='cpu'):
    return th.LongTensor(x,device=device)

def attach_grad(x:th.Tensor):
    if x.grad is not None:
        x.grad.zero_()
        return x
    else:
        return x.requires_grad_()

def boolean_mask(input, mask):
    if "bool" not in str(mask.dtype):
        mask = th.as_tensor(mask, dtype=th.bool)
    return input[mask]