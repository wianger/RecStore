---
name: benchmark-ps
description: Run RecStore PS/network transport benchmarks across RDMA, GRPC, and BRPC. Use when Codex needs to prompt for client/server hosts, record count, value size, and topology, then execute src/test/scripts/run_benchmark_ps.py and generate summary.csv plus Chinese summary.md.
---

# Benchmark PS

## Workflow

Use this skill from a RecStore checkout. Do not run helper scripts from this skill directory; call the project script directly.

1. Confirm the current directory is the RecStore repo root.
2. Prompt the user for:
   - transports (default = `rdma,grpc,brpc`)
   - client hosts (default = `127.0.0.1`)
   - server hosts (default = `127.0.0.1`)
   - server count (default = `1`)
   - client count (default = `1`)
   - record count (default = `1000000`)
   - value size (default = `512`)
   - batch keys (default = `1024`)
   - thread count (default = `16`)
   - runtime seconds (default = `5`)
   - repeat count (default = `1`)
   - execution backend (default = `local`)
   - result output directory (default = `results/benchmark_ps_$(date +%m%d%H%M)`)
3. Build:
   - `cmake -S . -B build`
   - `cmake --build build --target ps_transport_benchmark ps_server petps_server -j`
4. Run:
   - `python3 src/test/scripts/run_benchmark_ps.py`
5. Save generated configs, logs, `summary.csv`, and `summary.md` under the chosen output directory.
6. Report the result in Chinese and explicitly separate success, skip, and failure rows.

## Command Template

Use defaults only when the user accepts them.

```bash
cmake -S . -B build
cmake --build build --target ps_transport_benchmark ps_server petps_server -j
```

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports <transports> \
  --client-hosts <client_hosts> \
  --server-hosts <server_hosts> \
  --server-count <server_count> \
  --client-count <client_count> \
  --record-count <record_count> \
  --value-size <value_size> \
  --batch-keys <batch_keys> \
  --threads <threads> \
  --runtime-seconds <runtime_seconds> \
  --repeat <repeat> \
  --execution-backend <execution_backend> \
  --output-dir <output_dir>
```

If the user needs cross-host bring-up, use:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports <transports> \
  --client-hosts <client_hosts> \
  --server-hosts <server_hosts> \
  --server-count <server_count> \
  --client-count <client_count> \
  --record-count <record_count> \
  --value-size <value_size> \
  --batch-keys <batch_keys> \
  --threads <threads> \
  --runtime-seconds <runtime_seconds> \
  --repeat <repeat> \
  --execution-backend ssh \
  --remote-sync check \
  --remote-repo /app/RecStore \
  --output-dir <output_dir>
```

## Reporting Rules

- Treat all results as `PS/network` layer results.
- Do not describe them as storage-only or model-level conclusions.
- Do not claim RDMA succeeded when the runner returned skip code `77`.
- If any transport fails, still keep `summary.csv` and `summary.md`, then report which rows failed and where the logs are.
- Keep generated project-facing report text in Chinese.

## Current Bring-up Notes

- `run_benchmark_ps.py` writes `summary.csv` and `summary.md` even when all client rows fail. Always inspect those files before rerunning.
- If `GRPC` or `BRPC` fails during preload with `PutParameter failed`, inspect the client stderr log first. In the current repo snapshot, this is a real runtime failure, not a reporting bug.
- If `RDMA` single-shard fails with `RC write RPC wait timeout`, switch to the dedicated RC benchmark from `rdma-module` to confirm whether the lower RDMA transport is still healthy.
- If `RDMA` multi-shard fails with control-plane `get_meta timeout`, report it as multi-shard bring-up failure. Do not convert it into a throughput result.
