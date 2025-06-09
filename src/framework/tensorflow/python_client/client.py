import tensorflow as tf
import os
from typing import Optional

class RecstoreClient:
    _ops_module = None

    def __init__(self, library_path: Optional[str] = None):
        if RecstoreClient._ops_module:
            return
            
        if library_path is None:
            script_dir = os.path.dirname(__file__)
            default_lib_path = os.path.abspath(
                os.path.join(script_dir, '../../../../build/lib/librecstore_tf_ops.so')
            )
            if not os.path.exists(default_lib_path):
                 raise ImportError(
                    f"Could not find Recstore TF library at default path: {default_lib_path}\n"
                    "Please provide the correct path via the 'library_path' argument "
                    "or ensure your project is built correctly."
                )
            library_path = default_lib_path
        
        RecstoreClient._ops_module = tf.load_op_library(library_path)
        print(f"RecstoreClient (TensorFlow) initialized. Loaded library from: {library_path}")


    def emb_read(self, keys: tf.Tensor) -> tf.Tensor:
        if not isinstance(keys, tf.Tensor):
            keys = tf.convert_to_tensor(keys, dtype=tf.uint64)
        
        if keys.dtype != tf.uint64:
            raise TypeError(f"keys tensor must be of dtype tf.uint64, but got {keys.dtype}")

        return RecstoreClient._ops_module.recstore_emb_read(keys)

    def emb_update(self, keys: tf.Tensor, grads: tf.Tensor) -> tf.Operation:
        if not isinstance(keys, tf.Tensor):
            keys = tf.convert_to_tensor(keys, dtype=tf.uint64)
        if not isinstance(grads, tf.Tensor):
            grads = tf.convert_to_tensor(grads, dtype=tf.float32)

        if keys.dtype != tf.uint64:
            raise TypeError(f"keys tensor must be of dtype tf.uint64, but got {keys.dtype}")
        if grads.dtype != tf.float32:
            raise TypeError(f"grads tensor must be of dtype tf.float32, but got {grads.dtype}")

        return RecstoreClient._ops_module.recstore_emb_update(keys, grads)

