import os
import torch as th

def diff_tensor(new_tensor, name):
    file_name = f"/tmp/cached_tensor_{name}.pkl" 
    if os.path.exists(file_name):
        old_new_tensor  = th.load(file_name)
        assert (old_new_tensor == new_tensor).all()
        
    else:
        th.save(new_tensor, file_name)

# def diff_obj(new_tensor, name):
#     file_name = f"/tmp/cached_obj_{name}.pkl" 
#     if os.path.exists(file_name):
#         old_new_tensor  = th.load(file_name)
#         assert (old_new_tensor == new_tensor).all()
        
#     else:
#         th.save(new_tensor, file_name)