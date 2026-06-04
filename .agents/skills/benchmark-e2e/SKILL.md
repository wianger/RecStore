---
name: benchmark-e2e
description: Use when setting up RecStore BRPC end-to-end DLRM/TorchRec benchmarks with explicit client placement, PS placement, sharding, runnable commands, and Chinese performance summaries.
---

# Benchmark E2E

## Workflow

Use this skill from a RecStore checkout. Do not run helper scripts from this
skill directory; call project scripts directly.

1. Confirm the current directory is the RecStore repo root, or pass `--repo`.
2. Prompt only for P0 inputs unless the user already provided them:
   - client list, each entry as `(ssh_host, repo_root, ip, gpu_id, node_rank, nproc_per_node)`;
     default = one local client, GPU 0, `node_rank=0`, `nproc_per_node=1`
   - PS server list, each entry as `(ssh_host, repo_root, ip, port, shard_id)`;
     default = one local PS, `127.0.0.1:15000`, shard 0
   - result output directory (default = `results/brpc_e2e_$(date +%m%d%H%M)`)
3. Apply P1 defaults without prompting, and record them in `summary.md`:
   - model = `dlrm`
   - client deployment = inferred from client list
   - PS deployment = inferred from PS server list
   - shard policy = one shard per PS; `distributed_client.num_shards` equals
     the server count
   - dataset path = `model_zoo/torchrec_dlrm/processed_day_0_data`
   - runtime directory = `<output_dir>/runtime`
   - batch size = `1024`
   - embedding dimension = `128`
   - num embeddings = `200000`
   - steps = `80`, warmup steps = `5`, repeat = `3`
   - read mode = `prefetch`, prefetch depth = `0`
   - RecStore index type = `DRAM_PET_HASH`
   - comparison lanes = RecStore BRPC + TorchRec-HBM
4. Override P1 defaults only when the user mentions them. Ask P2 questions only
   when required by the requested experiment:
   - additional TorchRec baseline lanes such as `uvm_caching`, or disabling
     TorchRec with `--no-torchrec`
   - multiple batch sizes, embedding dimensions, or cardinality sweeps
   - non-default index type, read mode, prefetch depth, or run length
   - custom dataset, runtime directory, or per-feature cardinalities
5. Run:
   - `cmake -S . -B build`
   - `cmake --build build --target ps_server -j`
   - `ctest -R 'brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client' --output-on-failure`
   - start one `ps_server` per PS entry
   - run `model_zoo/rs_demo/run_mock_stress.py` for each RecStore E2E client
   - run matched TorchRec-HBM client commands with the same workload
6. Save deployment, commands, logs, runtime config, CSV artifacts, and
   `summary.md` under the chosen result directory.
7. Write `summary.md` as exactly three report sections, with benchmark
   hyperparameters recorded as Chinese prose under `Workload 说明` before the
   first table:
   - Workload description
   - E2E throughput
   - E2E latency breakdown

## Command Template

Ask the user for P0 inputs only: `client_list`, `ps_server_list`, and
`output_dir`. Use P1 defaults for everything else unless the user explicitly
overrides them.

```bash
cmake -S . -B build
cmake --build build --target ps_server -j
ctest -R 'brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client' --output-on-failure
```

Create `<runtime_dir>/recstore_config.json` with BRPC and the requested shard
layout:

```json
{
  "cache_ps": {
    "ps_type": "BRPC",
    "num_shards": 1,
    "servers": [
      {"host": "127.0.0.1", "port": 15000, "shard": 0}
    ],
    "base_kv_config": {
      "capacity": 200000,
      "index": {"type": "DRAM_PET_HASH"},
      "value": {"type": "DRAM_VALUE_STORE"}
    }
  },
  "distributed_client": {
    "num_shards": 1,
    "hash_method": "city_hash",
    "max_keys_per_request": 65536,
    "servers": [
      {"host": "127.0.0.1", "port": 15000, "shard": 0}
    ]
  }
}
```

Start each PS server from its configured host:

```bash
cd <server_repo>
build/bin/ps_server --config_path <runtime_dir>/recstore_config.json
```

Run one training command per client entry:

```bash
cd <client_repo>
CUDA_VISIBLE_DEVICES=<gpu_id> \
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore \
  --ps-type BRPC \
  --recstore-index-type <index_type> \
  --ps-kv-backend recstore_dram \
  --batch-size <batch_size> \
  --embedding-dim <embedding_dim> \
  --num-embeddings <num_embeddings> \
  --steps <steps> \
  --warmup-steps <warmup_steps> \
  --read-mode <read_mode> \
  --prefetch-depth <prefetch_depth> \
  --data-dir <dataset_path> \
  --output-root <output_dir> \
  --run-id <run_id> \
  --recstore-runtime-dir <runtime_dir> \
  --no-start-server \
  --server-host <first_ps_ip> \
  --server-port0 <first_ps_port>
```

For distributed clients, add:

```bash
  --nnodes <client_node_count> \
  --node-rank <node_rank> \
  --nproc-per-node <nproc_per_node> \
  --master-addr <master_ip> \
  --master-port <master_port>
```

Run the matched TorchRec-HBM command by default with the same dataset, batch
size, embedding dimension, steps, warmup steps, client deployment, and GPU
placement. If the user requests `uvm_caching`, add a second TorchRec baseline;
if the user requests RecStore-only, pass `--no-torchrec`.

## Deployment Record

Write `<output_dir>/deployment.md` before running:

```text
模型: dlrm
传输: BRPC
client:
  - ssh_host=<ssh_host>, repo=<repo_root>, ip=<client_ip>, gpu=<gpu_id>, node_rank=<rank>, nproc_per_node=<nproc>
ps:
  - ssh_host=<ssh_host>, repo=<repo_root>, ip=<ps_ip>, port=<port>, shard=<shard_id>
client 部署: single-node | distributed
PS 部署: single-ps | sharded-ps
分片: <num_shards>, hash_method=city_hash, max_keys_per_request=<value>
dataset: <dataset_path>
runtime: <runtime_dir>
output: <output_dir>
```

## Summary Format

Generate `<output_dir>/summary.md` from `recstore_main.csv`,
`recstore_main_agg.csv`, or the matrix runner's `summary_e2e.csv` after the E2E
benchmark finishes. Keep only these three sections:

1. `Workload 说明`
2. `E2E 吞吐（samples/s，...）`
3. `E2E 延迟分解（ms，...）`

Under `Workload 说明`, before the workload table, record the benchmark
hyperparameters in Chinese prose. Include at least: model, transport, client
list, PS server list, client deployment, PS deployment, shard count, shard
mapping, dataset path, runtime directory, output directory, batch size,
embedding dimension, num embeddings, steps, warmup steps, repeat, read mode,
prefetch depth, RecStore index type, and GPU placement.

Use `M` for values >= 1,000,000 and `K` for values >= 1,000. Include repeat
mean and CV columns only when repeat >= 3.

## Reporting Rules

- Do not claim tests pass unless the commands completed successfully.
- If a BRPC correctness test fails, stop before E2E and report the log path.
- If an E2E command exits nonzero, report the command and log path in the final
  response.
- Keep generated project-facing report text in Chinese.

## Current Bring-up Notes

- `cache_ps.servers` and `distributed_client.servers` must describe the same PS
  shard layout.
- Treat `hash_method`, `num_shards`, and `servers` as separate routing fields.
- Do not assume `shard_id == server list index`; route by the explicit shard id.
- For single-PS bring-up, prefer `DRAM_PET_HASH` first, then add
  `DRAM_EXTENDIBLE_HASH` or TorchRec comparison lanes after BRPC is stable.
