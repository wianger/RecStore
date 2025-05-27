import json

config = dict()


config['cache_ps'] = dict()
config['cache_ps']['max_batch_keys_size'] = 65536
config['cache_ps']['num_threads'] = 32


config['cache_ps']['kv_type'] = "KVEngineMap"
base_kv_config = dict()
base_kv_config['path'] = "/dev/shm/" + config['cache_ps']['kv_type']

base_kv_config['capacity'] = 1 * (10**6)

# base_kv_config['value_size'] = 32
# base_kv_config['value_pool_size'] = base_kv_config['capacity'] * \
#     base_kv_config['value_size']

base_kv_config['value_pool_size'] = 1 * 1024 * 1024 * 1024

base_kv_config['corotine_per_thread'] = 1


config['cache_ps']['base_kv_config'] = base_kv_config

with open("./grpc_ps_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
