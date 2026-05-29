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
   - client IPs (default = `127.0.0.1`)
   - server shard IPs, one entry per shard (default = `127.0.0.1`)
   - optional explicit server plan, comma-separated `host:port:shard` or `server_index:host:port:shard`
   - optional explicit client plan, comma-separated `host` or `client_index:host`
   - client processes per IP (default = `1`)
   - record count (default = `1000000`)
   - value size (default = `512`)
   - batch keys (default = `1024`)
   - client threads per process (default = `16`)
   - client load threads per process (default = client threads per process)
   - server worker threads (default = `32`)
   - server RDMA polling threads (default = `1`; use client processes for RDMA client concurrency)
   - runtime seconds (default = `5`)
   - repeat count (default = `1`)
   - execution backend (default = `local`)
   - result output directory (default = `results/benchmark_ps_$(date +%m%d%H%M)`)
3. Build before any CTest-based validation, so stale binaries are not tested:
   - `cmake -S . -B build`
   - `cmake --build build --target ps_transport_benchmark ps_server petps_server -j`
4. Validate the runner and related PS tests before benchmark runs:
   - `python3 -m unittest src/test/scripts/test_run_benchmark_ps.py`
   - `ctest -R 'grpc_ps_client_test|dist_grpc_ps_client_test|brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_transport_benchmark|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client' --output-on-failure`
5. If `rdma` is selected, confirm RDMA verbs are available before running RDMA rows:
   - `/dev/infiniband` exists
   - at least one `/dev/infiniband/uverbs*` device exists
   - if missing, skip RDMA rows and report the environment limitation instead of treating it as a failed benchmark
6. Split incompatible transport groups:
   - run `grpc,brpc` with client thread settings such as `--client-threads-per-process 16 --client-load-threads-per-process 16`
   - run `rdma` separately with `--client-threads-per-process 1 --client-load-threads-per-process 1`
   - do not run `rdma,grpc,brpc` in one command unless the chosen thread settings are valid for every selected transport
7. For fair transport comparisons, align `--client-threads-per-process` and `--client-load-threads-per-process` across all compared transports. If the run intentionally uses different thread counts, label it as a mixed-concurrency capacity check, not a fair transport comparison.
8. Run `src/test/scripts/run_benchmark_ps.py` with the selected topology.
9. Save generated configs, logs, `summary.csv`, and `summary.md` under the chosen output directory.
10. Report the result in Chinese and explicitly separate success, skip, and failure rows.

## Command Template

Use defaults only when the user accepts them.

```bash
cmake -S . -B build
cmake --build build --target ps_transport_benchmark ps_server petps_server -j
python3 -m unittest src/test/scripts/test_run_benchmark_ps.py
ctest -R 'grpc_ps_client_test|dist_grpc_ps_client_test|brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_transport_benchmark|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client' --output-on-failure
```

For an interactive setup:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --interactive
```

For a local RDMA smoke run, prefer this stable baseline:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 1 \
  --record-count 2000 \
  --value-size 64 \
  --batch-keys 64 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 1 \
  --repeat 1 \
  --execution-backend local \
  --output-dir <output_dir>
```

For local RDMA concurrency, scale with `--client-processes-per-ip`; keep `--client-threads-per-process 1` for RDMA transactions:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1,127.0.0.1 \
  --client-processes-per-ip 2 \
  --record-count <record_count> \
  --value-size <value_size> \
  --batch-keys <batch_keys> \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds <runtime_seconds> \
  --repeat <repeat> \
  --execution-backend local \
  --output-dir <output_dir>
```

For RDMA bottleneck diagnosis, keep fake/status-only rows separate from default benchmark rows and label them as diagnostic. Use profile/fake knobs to separate synchronous request scheduling, server GET payload work, and client copy:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1,127.0.0.1 \
  --client-processes-per-ip 2 \
  --record-count <record_count> \
  --value-size <value_size> \
  --batch-keys <batch_keys> \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds <runtime_seconds> \
  --repeat 1 \
  --execution-backend local \
  --rdma-rc-profile-interval-ms 1000 \
  --output-dir <output_dir>/rdma_profile

python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1,127.0.0.1 \
  --client-processes-per-ip 2 \
  --record-count <record_count> \
  --value-size <value_size> \
  --batch-keys <batch_keys> \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds <runtime_seconds> \
  --repeat 1 \
  --execution-backend local \
  --prefetch-depth <depth> \
  --rdma-rc-qps-per-client-per-shard <at_least_depth> \
  --rdma-rc-profile-interval-ms 1000 \
  --output-dir <output_dir>/rdma_prefetch_<depth>
```

Generic PS RDMA `transactions/fetch` uses the prefetch/result pipeline by default when `--prefetch-depth` is not set.

For a mixed local reliability matrix, run RPC and RDMA as separate commands because their safe thread settings differ:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports grpc,brpc \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 1 \
  --record-count <record_count> \
  --value-size <value_size> \
  --batch-keys <batch_keys> \
  --client-threads-per-process 16 \
  --client-load-threads-per-process 16 \
  --runtime-seconds <runtime_seconds> \
  --repeat <repeat> \
  --execution-backend local \
  --output-dir <output_dir>/rpc

python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 1 \
  --record-count <record_count> \
  --value-size <value_size> \
  --batch-keys <batch_keys> \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds <runtime_seconds> \
  --repeat <repeat> \
  --execution-backend local \
  --output-dir <output_dir>/rdma
```

For a fair local transport matrix, keep per-process client threads aligned. Use multiple output directories under one matrix root:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports grpc,brpc,rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips <comma_separated_shard_ips> \
  --client-processes-per-ip <processes> \
  --record-count <record_count> \
  --value-size <value_size> \
  --batch-keys <batch_keys> \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds <runtime_seconds> \
  --repeat <repeat> \
  --execution-backend local \
  --server-worker-threads <threads> \
  --server-rdma-threads <rdma_threads> \
  --rdma-rc-qps-per-client-per-shard <qps_pool> \
  --output-dir <matrix_root>/b<batch>_s<shards>_p<processes>_t1
```

Recommended first matrix:

- `server_shards`: `1,2`
- `client_processes_per_ip`: `1,2,4,8`
- `client_threads_per_process`: `1`
- `batch_keys`: user-requested value, or `500` when reproducing the 2026-05-29 matrix

For explicit cross-host or multi-shard placement, use `--server-plan` and `--client-plan`:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports <transports> \
  --server-plan 0:server-a:25000:0,1:server-b:25001:3 \
  --client-plan 0:client-a,1:client-a \
  --record-count <record_count> \
  --value-size <value_size> \
  --batch-keys <batch_keys> \
  --client-threads-per-process <threads> \
  --runtime-seconds <runtime_seconds> \
  --repeat <repeat> \
  --execution-backend ssh \
  --remote-sync check \
  --remote-repo /app/RecStore \
  --output-dir <output_dir>
```

If `--server-plan` is provided, it overrides the `server_shard_ips` mapping. If `--client-plan` is provided, it overrides the `client_ips` / `client_processes_per_ip` mapping.

## Summary Format

Use the generated `<output_dir>/summary.md` and `<output_dir>/summary.csv`; do not rewrite benchmark numbers from logs by hand.

In the final Chinese response, include:

1. Validation results: build, unit tests, and CTest status.
2. Benchmark scope: transports, topology, record count, value size, batch keys, client threads per process, runtime seconds, repeat count, and output directory.
3. Success rows in a Markdown table, using `summary.csv` as the source:

```markdown
| transport | phase | client_processes | client_threads_per_process | server_shards | batch_keys | value_size | M keys/s | status | output_dir |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
```

4. Per-client detail when needed: transport, phase, client index, and M keys/s.
5. Skip / failure rows: transport, status, message, and log paths.
6. Warnings: non-empty `stderr.log`, teardown stack traces, allocator warnings, or timeout messages that did not change row status.

Treat `summary.csv` as authoritative when any client exits nonzero. `run_benchmark_ps.py` still writes summaries on partial or total failure.

For matrix reports, aggregate `run` rows by summing `key_ops_per_sec` across clients for each transport and case. Include a per-client min/max table when `client_processes_per_ip > 1`; large min/max spread is a scheduling or fairness signal and should not be hidden by the aggregate.

When replacing earlier exploratory reports, create one new root-level report with a clear name such as `benchmark_ps_transport_matrix_<date>.md`, then remove superseded root-level scratch reports and transient result directories. Keep the raw result directory for the final matrix.

## Reporting Rules

- Treat all results as `PS/network` layer results.
- Do not describe them as storage-only or model-level conclusions.
- Do not claim RDMA succeeded when the runner returned skip code `77`.
- If any transport fails, still keep `summary.csv` and `summary.md`, then report which rows failed and where the logs are.
- Check for non-empty stderr logs even when all rows are `success`; report request-path warnings separately from expected teardown output.
- Do not claim tests pass unless the exact command completed successfully.
- Keep generated project-facing report text in Chinese.
- Treat `--prefetch-depth`, `--rdma-rc-fake-get-mode`, and `--rdma-rc-skip-client-copy` as diagnostic-only knobs. Report them as bottleneck attribution, not as ordinary throughput comparisons.
- Do not compare rows with different `client_threads_per_process` as a fair transport comparison. If rows differ, explain that the result is a mixed-concurrency capacity observation.
- Do not interpret local multi-shard results where all shards run on `127.0.0.1` as distributed scaling. Label them as local multi-process stress results.
- Separate PS/network conclusions from storage-only, RDMA RC transport-only, and PyTorch/model conclusions.

## Verified Local Matrix

The current local environment has validated these paths after the latest benchmark-runner fixes. Revalidate them after changing PS, RDMA, benchmark, config-generation, or allocator code:

- Full CTest: `55/55` passed.
- PS-focused CTest: `8/8` passed.
- RDMA local single client: passed with `--client-threads-per-process 1`.
- RDMA local 2 clients: passed with `--client-threads-per-process 1 --client-processes-per-ip 2`.
- RDMA local 2 shards x 2 clients: passed with `--client-threads-per-process 1 --server-shard-ips 127.0.0.1,127.0.0.1 --client-processes-per-ip 2`.
- Fair local transport matrix with `batch_keys=500`, `value_size=512`, `client_threads_per_process=1`, `server_shards=1,2`, and `client_processes_per_ip=1,2,4,8` completed successfully.

Use these as bring-up baselines before increasing record count, runtime, or cross-host complexity.

## Current Bring-up Notes

- `run_benchmark_ps.py` writes `summary.csv` and `summary.md` even when all client rows fail. Always inspect those files before rerunning.
- `GRPC` / `BRPC` benchmark clients may be distributed clients even in local runs because the runner writes `distributed_client` config. Distributed RPC clients report success as `0`; ordinary RPC clients report success as nonzero. Keep `ps_transport_benchmark` return-code handling aligned with the concrete client type.
- RDMA transaction mode requires `--client-threads-per-process 1`; use `--client-processes-per-ip` for RDMA client concurrency. If the benchmark binary sees `--thread_num > 1`, it aborts by design.
- `--prefetch-depth > 0` is valid only for `transactions` + `mode=fetch`. If depth exceeds the default QP pool, also increase `--rdma-rc-qps-per-client-per-shard`.
- By default, RDMA `transactions/fetch` uses a depth-16 prefetch pipeline.
- In the 2026-05-29 local fair matrix, single-shard RDMA plateaued around `3.0 M keys/s`, while local two-shard RDMA was lower. Treat this as local PS/server scheduling behavior, not a NIC-limit conclusion.
- Large single-shard RDMA preload can fail as `RC write RPC wait timeout` after server-side `KVEngine value allocation failed`; the runner should generate slab allocator capacity with headroom. If this recurs, inspect generated config capacity and server logs before treating it as a transport failure.
- If `RDMA` multi-shard fails with control-plane `get_meta timeout`, report it as multi-shard bring-up failure. Do not convert it into a throughput result.
