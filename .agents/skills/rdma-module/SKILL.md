---
name: rdma-module
description: Work on RecStore RDMA parameter-server code, tests, and benchmarks. Use when Codex needs to inspect or modify src/ps/rdma, validate RDMA correctness with PetPS/RDMAPS tests, run RC transport benchmarks, compare RDMA read/push modes, or summarize RDMA profile output and bottlenecks.
---

# RDMA Module

## Scope

Use this skill from a RecStore checkout. The current RDMA module is the Parameter Server RDMA path, mainly:

- `src/ps/rdma/*`
- `src/test/test_rdma_rc_protocol.cpp`
- `src/test/test_rdmaps_client_adapter.cpp`
- `src/test/test_raw_verbs_allocator.cpp`
- `src/test/ps/rdma/petps_integration_test.cpp`
- `src/test/scripts/run_benchmark_ps.py`
- `src/test/scripts/run_rdma_transport_benchmarks.py`
- `src/test/scripts/run_rdma_rc_transport_benchmark.py`
- `docs/parameter_server/rdma.md`

Do not treat third-party RDMA code under `third_party/` as the RecStore RDMA module unless the task explicitly asks for it.

## Workflow

1. Confirm the current directory is the RecStore repo root.
2. Read `docs/parameter_server/rdma.md` before changing behavior or interpreting benchmark output.
3. Check whether RDMA hardware is available:
   - `/dev/infiniband` must exist.
   - at least one `/dev/infiniband/uverbs*` device must exist.
   - If missing, do not run RDMA integration or benchmark commands; report that the environment cannot exercise real verbs.
4. For code changes, first identify the layer being changed:
   - protocol/layout: `rdma_protocol.h`, `rc_transport.*`, `raw_verbs_transport.*`
   - PetPS client/server semantics: `petps_client.*`, `petps_server.cc`, `allshards_ps_client.*`
   - generic PS adapter: `rdma_ps_client_adapter.*`
   - benchmark/runtime flags: `rc_options.*`, `src/test/scripts/run_rdma_*.py`
5. Build the narrowest relevant targets.
6. Run correctness tests before benchmark tests.
7. If running benchmarks, save the command, parameters, raw logs, and parsed summary in a result directory.

## Correctness Commands

Use the narrowest useful subset first:

```bash
cmake -S . -B build
cmake --build build --target \
  test_rdma_rc_protocol \
  test_raw_verbs_allocator \
  test_rdmaps_client_adapter \
  petps_server \
  petps_integration_test \
  -j
```

```bash
ctest --test-dir build -R 'test_rdma_rc_protocol|test_raw_verbs_allocator|test_rdmaps_client_adapter' -VV
```

Run real PetPS RDMA integration only when verbs devices exist:

```bash
ctest --test-dir build -R 'petps_hashed_value_transfer_test' -VV
```

For broader integration coverage, configure with RDMA integration tests enabled, then rebuild:

```bash
cmake -S . -B build -DENABLE_RDMA_INTEGRATION_TESTS=ON
cmake --build build --target petps_server petps_integration_test -j
ctest --test-dir build -L rdma_integration -VV
```

If `ctest` returns skip code `77`, report the skip reason instead of treating it as a pass.

## Benchmark Commands

Build benchmark binaries:

```bash
cmake --build build --target ps_transport_benchmark rdma_rc_transport_benchmark -j
```

Generic PS benchmark RDMA single-client transactions smoke:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --server-count 1 \
  --client-count 1 \
  --record-count 10000 \
  --value-size 128 \
  --batch-keys 64 \
  --threads 1 \
  --load-threads 1 \
  --runtime-seconds 1 \
  --repeat 1 \
  --rdma-wait-timeout-ms 8000 \
  --rdma-rc-qps-per-client-per-shard 4 \
  --rdma-rc-slots-per-qp 1
```

Single-client RDMA transport benchmark, current PUT-v2 read mode:

```bash
python3 src/test/scripts/run_rdma_transport_benchmarks.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --iterations 300 \
  --batch-keys 500 \
  --rounds 20 \
  --rdma-warmup-rounds 10 \
  --report-mode summary \
  --rdma-only \
  --rdma-thread-num 1 \
  --rdma-put-protocol-version 2 \
  --rdma-put-v2-transfer-mode read \
  --rdma-wait-timeout-ms 20000 \
  --rdma-client-timeout-sec 60 \
  --show-runner-logs \
  --use-local-memcached auto
```

Single-client RDMA transport benchmark, current PUT-v2 push mode:

```bash
tools/benchmarks/run_rdma_transport_push_summary.sh
```

Multi-client RC-write stress benchmark:

```bash
./run_rdma_rc_transport_benchmark.sh
```

When testing low-load limits, prefer explicit parameters over wrapper defaults:

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 1 \
  --thread-num 1 \
  --iterations 100 \
  --rounds 10 \
  --warmup-rounds 3 \
  --batch-keys 16 \
  --value-size 512 \
  --op async_stream \
  --async-depth 16 \
  --qps-per-client-per-shard 16 \
  --report-mode summary \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 300 \
  --cluster-timeout 60 \
  --use-local-memcached auto
```

## Parameter Rules

- `--batch-keys` is keys per request, not total keys.
- Measured key volume is roughly `iterations * rounds * batch_keys`.
- `--qps-per-client-per-shard` is RC QP pool size, not a target QPS.
- For `async_stream`, require `qps-per-client-per-shard >= async-depth`.
- `read` and `push` PUT-v2 modes are different transport paths; do not merge them into one throughput conclusion.
- Higher `thread-num` can increase polling capacity but can also hide low-load fixed costs.
- `--fake-get-mode` and `--skip-client-copy` are diagnostic knobs, not default benchmark settings.

## Reporting Rules

- Reply to the user in Chinese.
- Do not claim RDMA correctness or benchmark success unless the relevant command completed successfully.
- If verbs devices are missing, state that only build/unit-level validation was possible.
- For benchmark reports, include:
  - exact command
  - `client-count`, `server-count`, `thread-num` or `rdma-thread-num`
  - `iterations`, `rounds`, `warmup-rounds`
  - `batch-keys`, `value-size`, `op`, `async-depth`
  - `qps-per-client-per-shard`
  - PUT protocol version and transfer mode
  - timeout values
  - aggregate `ops/s`, `key_ops/s`, and per-request latency fields when present
- Preserve raw logs when the run is long or flaky. Prefer result directories such as `results/rdma_<mode>_$(date +%m%d%H%M)`.

## Current Bring-up Notes

- Treat `run_benchmark_ps.py` and `run_rdma_rc_transport_benchmark.py` as different validation layers. If the generic PS benchmark fails but the dedicated RC benchmark succeeds, report that the failure is above or beside the RC transport baseline.
- In the current repo snapshot, the low-load single-client RC benchmark is a good smoke baseline. Prefer proving this path first before escalating to multi-client or multi-shard RDMA runs.
- As of `2026-05-27`, generic PS benchmark RDMA `transactions/fetch` is verified for `client-count=1` and `client-count=2`, with `threads=1` and `load-threads=1`, by reusing the same client across load and run inside each benchmark process.
- For generic PS benchmark RDMA concurrency, use `--client-count` for multi-process clients. Keep `--threads=1` per process unless the adapter's same-process multi-client/lane semantics are explicitly changed and revalidated.
- If server shutdown prints a `SIGTERM` stack trace after a successful run, do not classify that alone as a benchmark failure. Distinguish expected teardown from request-path failures such as `RC write RPC wait timeout`.

## Debugging Focus

When investigating failures or regressions, prioritize these invariants:

- slot layout and offsets must match across client, server, and transport
- commit/status words must preserve write ordering
- request buffer lifetime must outlive RDMA submission and completion
- response payload must be visible before status completion
- QP acquisition/release must be balanced under async paths
- multi-shard routing must use explicit shard ids, not sorted index assumptions
- client/server timeout failures should surface loudly, not appear as hangs

For profile interpretation, map symptoms to components:

- high `submit_request_ns`: client-side verbs submission or descriptor path
- high `wait_status_ns`: server processing, response writeback, or CQ/status delay
- high `copy_response_ns`: client response copy overhead
- high `poll_loop_ns` or `empty_scan_rounds`: server polling fixed cost
- high `get_batch_get_ns`: KV lookup path
- high `complete_response_ns` or `drain_pending_response_ns`: response writeback pressure
- nonzero `acquire_qp_failures`: QP pool/resource exhaustion, not normal latency
