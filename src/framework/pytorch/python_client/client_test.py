import torch
from client import RecstoreClient

client = RecstoreClient()

keys_to_read = torch.tensor([10, 25, 101, 3, 42], dtype=torch.int64)
print(f"Reading embeddings for keys: {keys_to_read.tolist()}")

read_values = client.emb_read(keys_to_read)

print("Read successful. Shape of values:", read_values.shape)
print("First returned embedding vector:\n", read_values[0])

print("-" * 20)

keys_to_update = torch.tensor([15, 20], dtype=torch.int64)
grads_to_update = torch.randn(2, 128, dtype=torch.float32)
print(f"Updating embeddings for keys: {keys_to_update.tolist()}")

client.emb_update(keys_to_update, grads_to_update)
print("Update call successful.")