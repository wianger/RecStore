---
name: rdma-module
description: Work on RecStore RDMA parameter-server code, tests, and benchmarks, including optimized PS RDMA GET where DRAM_PET_HASH binds to auto/staging-copy response mode. Use when Codex needs to inspect or modify src/ps/rdma, validate RDMA correctness with PetPS/RDMAPS tests, run RC transport benchmarks, compare RDMA read/push modes, or summarize RDMA profile output and bottlenecks.
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
5. Build the narrowest relevant targets before CTest so stale binaries are not tested.
6. Run correctness tests before benchmark tests.
7. If running benchmarks, save the command, parameters, raw logs, and parsed summary in a result directory.
8. Run benchmark commands sequentially unless the runner explicitly allocates distinct ports, namespaces, and runtime directories. RDMA tests can conflict through memcached/control-plane state, QP resources, and default ports.

## Correctness Commands

Use the narrowest useful subset first:

```bash
cmake -S . -B build
cmake --build build --target \
  test_rdma_rc_protocol \
  test_raw_verbs_allocator \
  test_rdmaps_client_adapter \
  test_allshards_ps_client \
  petps_server \
  petps_integration_test \
  -j
```

```bash
ctest --test-dir build -R 'test_rdma_rc_protocol|test_raw_verbs_allocator|test_rdmaps_client_adapter' -VV
```

For client-side coroutine wait changes, include `test_allshards_ps_client`:

```bash
ctest --test-dir build -R 'test_allshards_ps_client|test_rdmaps_client_adapter' -VV
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
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 1 \
  --record-count 10000 \
  --value-size 128 \
  --batch-keys 64 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 1 \
  --repeat 1 \
  --rdma-wait-timeout-ms 8000 \
  --rdma-rc-qps-per-client-per-shard 4 \
  --rdma-rc-slots-per-qp 1 \
  --execution-backend local \
  --output-dir results/rdma_ps_smoke_$(date +%m%d%H%M)
```

Generic PS RDMA fetch pipeline diagnosis:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 6 \
  --record-count 1000000 \
  --value-size 512 \
  --batch-keys 500 \
  --index-type DRAM_PET_HASH \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 5 \
  --repeat 1 \
  --execution-backend local \
  --prefetch-depth 16 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-slots-per-qp 1 \
  --rdma-rc-profile-interval-ms 1000 \
  --server-rdma-threads 16 \
  --rdma-rc-server-get-workers 0 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --rdma-get-response-mode auto \
  --output-dir results/rdma_ps_prefetch_$(date +%m%d%H%M)
```

Generic PS RDMA `transactions/fetch` uses a depth-16 prefetch/result pipeline by default. Use larger explicit `--prefetch-depth` values only when `--rdma-rc-qps-per-client-per-shard` is at least as large as the target depth.

For local upper-bound RC transport runs, prefer keeping server and clients on the RNIC-local socket while assigning disjoint physical cores. On the 2026-05-30 2-socket test host, the clean socket0-disjoint baseline is `server-numa-id=0`, `client-numa-id=0`, server core offset `0`, client core offset `16`, client stride `2`, `client-count=6`, and `thread-num=16`. This reached about `48.69M keys/s` (`24.93GB/s` at 512B). Do not use socket-split results as the transport ceiling unless the goal is specifically to isolate cross-socket CPU interference.

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 6 \
  --thread-num 16 \
  --iterations 300 \
  --rounds 8 \
  --warmup-rounds 2 \
  --batch-keys 500 \
  --value-size 512 \
  --op async_stream \
  --async-depth 16 \
  --qps-per-client-per-shard 16 \
  --slots-per-qp 1 \
  --report-mode summary \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 120 \
  --cluster-timeout 60 \
  --profile-interval-ms 1000 \
  --server-numa-id 0 \
  --client-numa-id 0 \
  --server-bind-core-offset 0 \
  --client-bind-core-offset 16 \
  --client-bind-core-stride 2 \
  --show-runner-logs
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
tools/benchmarks/run_rdma_rc_transport_benchmark.sh
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
- For `async_stream`, require `qps-per-client-per-shard * slots-per-qp >= async-depth`.
- For generic PS transaction benchmarks, keep `--client-threads-per-process 1 --client-load-threads-per-process 1` for RDMA unless same-process multi-lane semantics have been explicitly changed and revalidated; scale load with `--client-processes-per-ip`.
- `--prefetch-depth` on `run_benchmark_ps.py` overrides the default RDMA fetch pipeline depth. It is valid only with `transactions` and `mode=fetch`.
- `--server-rdma-threads`, `--rdma-rc-server-get-workers`, and `--rdma-rc-server-coroutines-per-thread` are distinct generic PS scheduling dimensions. Encode them in result dirs as `t<T>_n<N>_c<C>`.
- Keep `--rdma-rc-server-coroutines-per-thread=1` unless explicitly testing scanner scheduling. `C>1` can now combine with GET workers, but it is not a performance default.
- RDMA GET response mode is layout-dependent. Use `--rdma-get-response-mode auto` by default: it maps `DRAM_PET_HASH` to `staging_copy` and other index types to `direct_sg`. Do not pass old direct-SG enable/disable benchmark flags; they have been retired.
- Do not reintroduce `rdma_rc_get_inner_parallelism` as a default tuning dimension. On the 2026-05-30 direct-SG tests, inner parallelism reduced one sub-stage but cut end-to-end PS throughput roughly in half because queue/synchronization overhead dominated.
- Do not combine RDMA with GRPC/BRPC in the same `run_benchmark_ps.py` command when using RPC-oriented `--client-threads-per-process` values. Split RPC and RDMA runs if the safe thread settings differ.
- When comparing RDMA with GRPC/BRPC, either align `client_threads_per_process` across transports or label the comparison as a capacity-oriented mixed-concurrency run. Do not present mixed thread counts as a fair transport comparison.
- Local multi-shard runs where all `server-shard-ips` are `127.0.0.1` are single-machine stress tests, not evidence of cross-machine shard scaling.
- `read` and `push` PUT-v2 modes are different transport paths; do not merge them into one throughput conclusion.
- Higher `thread-num` can increase polling capacity but can also hide low-load fixed costs.
- `--fake-get-mode` and `--skip-client-copy` are diagnostic knobs, not default benchmark settings.
- Combine `--prefetch-depth` with `--rdma-rc-profile-interval-ms` when diagnosing generic PS bottlenecks so `pending_rpc_peak`, `get_batch_get_avg_ns`, `get_row_copy_avg_ns`, and `scan_hit_pct` are available.

## Reporting Rules

- Reply to the user in Chinese.
- Do not claim RDMA correctness or benchmark success unless the relevant command completed successfully.
- If verbs devices are missing, state that only build/unit-level validation was possible.
- For benchmark reports, include:
  - exact command
  - `client-processes-per-ip`, `server-shard-ips`, `client-threads-per-process`, and `server-rdma-threads`
  - `iterations`, `rounds`, `warmup-rounds`
  - `batch-keys`, `value-size`, `op`, `async-depth`
  - `qps-per-client-per-shard`
  - PUT protocol version and transfer mode
  - timeout values
  - aggregate `ops/s`, `key_ops/s`, and per-request latency fields when present
- Preserve raw logs when the run is long or flaky. Prefer result directories such as `results/rdma_<mode>_$(date +%m%d%H%M)`.
- Inspect non-empty stderr logs even when parsed rows are successful. Report expected teardown separately from request-path failures, timeouts, allocator failures, or QP acquisition failures.

## Current Bring-up Notes

- Treat `run_benchmark_ps.py` and `run_rdma_rc_transport_benchmark.py` as different validation layers. If the generic PS benchmark fails but the dedicated RC benchmark succeeds, report that the failure is above or beside the RC transport baseline.
- Before diagnosing a generic PS RDMA timeout as a transport issue, inspect server logs for allocation or config failures such as `KVEngine value allocation failed`.
- In the current repo snapshot, the low-load single-client RC benchmark is a good smoke baseline. Prefer proving this path first before escalating to multi-client or multi-shard RDMA runs.
- As of `2026-05-29`, generic PS benchmark RDMA `transactions/fetch` is verified locally with `client-processes-per-ip=1,2,4,8`, `server-shard-ips` counts of `1` and `2`, and `client-threads-per-process=1`.
- For generic PS benchmark RDMA concurrency, use `--client-processes-per-ip` for multi-process clients. Keep `--client-threads-per-process=1` per process unless the adapter's same-process multi-client/lane semantics are explicitly changed and revalidated.
- In the local `batch_keys=500`, `value_size=512`, `client_threads_per_process=1` matrix, single-shard RDMA plateaued around `3.0 M keys/s`; adding local shards on the same host reduced throughput. Treat this as a local PS/server scheduling observation, not as a NIC limit or distributed scaling result.
- As of `2026-05-30`, generic PS RDMA `p8/N8/T16/C1` repeat=3 averaged `14.45M keys/s`; same topology `C4` averaged `12.17M keys/s`. Treat coroutine scanner as functionally validated but not a throughput default.
- As of `2026-05-31`, direct-SG is not the universal GET default. `DRAM_EXTENDIBLE_HASH + direct-SG` is around the EH storage-only limit (`~19M keys/s`), while `DRAM_PET_HASH + direct-SG` regressed to about `14.93M keys/s`. `DRAM_PET_HASH + auto/staging-copy` reached about `44.87M keys/s`, close to the observed RDMA transport/device ceiling of about `48.7M keys/s`.
- Historical note: sglist/direct-SG improved the then-current EH path by only about `0.5M keys/s`; the later `~19M keys/s` EH result mainly came from CPU affinity. EH prefetch/array-view style micro-optimizations were only about `0.1M keys/s`.
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
- `pending_rpc_peak=1` in generic PS fetch benchmarks: synchronous client request path is limiting outstanding RDMA work. Default RDMA fetch benchmark rows should show a higher peak.
- nonzero `acquire_qp_failures`: QP pool/resource exhaustion, not normal latency
