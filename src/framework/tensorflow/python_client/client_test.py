import tensorflow as tf
from client import RecstoreClient

client = RecstoreClient()

# --- 测试 EmbRead ---
keys_to_read = tf.constant([10, 25, 101, 3, 42], dtype=tf.uint64)
read_values = client.emb_read(keys_to_read)

print("Read call successful.")
print("Shape of returned values:", read_values.shape)
print("First returned embedding vector:\n", read_values[0].numpy())

print("-" * 20)

# --- 测试 EmbUpdate ---
keys_to_update = tf.constant([15, 20], dtype=tf.uint64)
grads_to_update = tf.random.normal(shape=(2, 128), dtype=tf.float32)

update_op = client.emb_update(keys_to_update, grads_to_update)

print("Update call successful.")