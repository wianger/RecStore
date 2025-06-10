import torch
import os
from typing import Optional

class RecstoreClient:
    _is_initialized = False

    def __init__(self, library_path: Optional[str] = None):
        if RecstoreClient._is_initialized:
            return
            
        if library_path is None:
            script_dir = os.path.dirname(__file__)
            default_lib_path = os.path.abspath(
                os.path.join(script_dir, '../../../../build/lib/librecstore_torch_ops.so')
            )
            if not os.path.exists(default_lib_path):
                 raise ImportError(
                    f"Could not find Recstore library at default path: {default_lib_path}\n"
                    "Please provide the correct path via the 'library_path' argument "
                    "or ensure your project is built correctly."
                )
            library_path = default_lib_path

        torch.ops.load_library(library_path)
        self.ops = torch.ops.recstore_ops
        RecstoreClient._is_initialized = True
        print(f"RecstoreClient initialized. Loaded library from: {library_path}")


    def emb_read(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")

        return self.ops.emb_read(keys)

    def emb_update(self, keys: torch.Tensor, grads: torch.Tensor) -> None:
        if keys.dtype != torch.int64:
            raise TypeError(f"keys tensor must be of dtype torch.int64, but got {keys.dtype}")
        if grads.dtype != torch.float32:
            raise TypeError(f"grads tensor must be of dtype torch.float32, but got {grads.dtype}")

        self.ops.emb_update(keys, grads)

