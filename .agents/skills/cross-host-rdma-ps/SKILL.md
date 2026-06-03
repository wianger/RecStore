---
name: cross-host-rdma-ps
description: Run and diagnose RecStore cross-host PS RDMA benchmarks from one container using SSH into host Docker containers. Use when testing node190/node191-style multi-machine RecStore RDMA PS runs, NFS-mounted repos, remote-container execution, DRAM_PET_HASH auto/staging-copy GET, server poller profile, and cross-host vs local comparison.
---

# Cross-Host RDMA PS

Use this skill from the RecStore repo root when running PS/network RDMA
benchmarks across two machines via `src/test/scripts/run_benchmark_ps.py`.

## Fixed Context

Default topology used in the current lab:

- server host: `xieminhui@10.0.2.190`
- client host: `xieminhui@10.0.2.191`
- server RecStore endpoint IP: `10.0.2.190`
- client RecStore endpoint IP: `10.0.2.191`
- Docker container on both hosts: `recstore`
- repo path inside containers: `/app/RecStore`
- server port: `25000`

Keep SSH targets and RecStore endpoint hosts distinct. The runner may SSH with
`xieminhui@IP`, but RecStore configs should use the pure endpoint IP.

## Safety Rules

- Use the `$benchmark-ps` workflow, not the standalone RDMA module runner.
- For `DRAM_PET_HASH + auto/staging_copy`, always set
  `--rdma-rc-server-get-workers 0`.
- Treat all results as PS/network layer results.
- Use unique `--rdma-control-plane-port` values for each run.
- Do not reuse stale server/client processes. Verify and clean benchmark-only
  processes before important runs.
- Do not interpret diagnostic fake modes as normal benchmark throughput.

## Preflight

Confirm SSH and container access:

```bash
ssh xieminhui@10.0.2.190 "docker exec recstore bash -lc 'cd /app/RecStore && pwd'"
ssh xieminhui@10.0.2.191 "docker exec recstore bash -lc 'cd /app/RecStore && pwd'"
```

Confirm RDMA device visibility:

```bash
ssh xieminhui@10.0.2.190 "docker exec recstore bash -lc 'ls -l /dev/infiniband; for d in /sys/class/infiniband/*; do echo DEV:\${d##*/}; cat \$d/device/numa_node 2>/dev/null; cat \$d/ports/1/rate 2>/dev/null; done'"
ssh xieminhui@10.0.2.191 "docker exec recstore bash -lc 'ls -l /dev/infiniband; for d in /sys/class/infiniband/*; do echo DEV:\${d##*/}; cat \$d/device/numa_node 2>/dev/null; cat \$d/ports/1/rate 2>/dev/null; done'"
```

Clean only benchmark processes:

```bash
ssh xieminhui@10.0.2.190 "docker exec recstore bash -lc 'pgrep -af \"petps_server|ps_server|ps_transport_benchmark\" || true'"
ssh xieminhui@10.0.2.191 "docker exec recstore bash -lc 'pgrep -af \"petps_server|ps_server|ps_transport_benchmark\" || true'"
```

If residual benchmark processes exist, kill by exact PID when possible. Avoid
`pkill -f` if it matches the current shell command.

## Canonical Cross-Host RDMA GET Run

Use this for the current six-client cross-host PET_HASH profile baseline:

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --server-plan 0:xieminhui@10.0.2.190:25000:0 \
  --client-plan 0:xieminhui@10.0.2.191,1:xieminhui@10.0.2.191,2:xieminhui@10.0.2.191,3:xieminhui@10.0.2.191,4:xieminhui@10.0.2.191,5:xieminhui@10.0.2.191 \
  --record-count 1000000 \
  --value-size 512 \
  --batch-keys 500 \
  --index-type DRAM_PET_HASH \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 5 \
  --repeat 1 \
  --execution-backend ssh \
  --remote-sync check \
  --remote-repo /app/RecStore \
  --remote-container recstore \
  --prefetch-depth 16 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-slots-per-qp 1 \
  --server-rdma-threads 16 \
  --rdma-rc-server-get-workers 0 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --rdma-get-response-mode auto \
  --rdma-rc-profile-interval-ms 1000 \
  --rdma-rc-server-numa-id 0 \
  --rdma-rc-client-numa-id 0 \
  --rdma-server-bind-core-offset 0 \
  --rdma-client-bind-core-offset 16 \
  --rdma-client-bind-core-stride 2 \
  --rdma-control-plane-port <unused_port> \
  --output-dir results/<output_dir> \
  --show-runner-logs
```

## Local Comparison Run

Use this to compare against same-host RDMA capacity:

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
  --server-rdma-threads 16 \
  --rdma-rc-server-get-workers 0 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --rdma-get-response-mode auto \
  --rdma-rc-profile-interval-ms 1000 \
  --rdma-rc-server-numa-id 0 \
  --rdma-rc-client-numa-id 0 \
  --rdma-server-bind-core-offset 0 \
  --rdma-client-bind-core-offset 16 \
  --rdma-client-bind-core-stride 2 \
  --output-dir results/<output_dir>
```

## Diagnostic Variants

Keep one knob changed per run:

- Server device selection on node190:
  - `--rdma-rc-server-numa-id 0`
  - `--rdma-rc-server-numa-id 1`
  - `--rdma-rc-server-numa-id 2`
- Request/status path ceiling:
  - `--rdma-rc-fake-get-mode status_only`
- Server payload write without PET_HASH lookup:
  - `--rdma-rc-fake-get-mode payload_memset`
- Client response-copy check:
  - `--rdma-rc-skip-client-copy`
- Outstanding/QP checks:
  - `--prefetch-depth 32 --rdma-rc-qps-per-client-per-shard 32`
  - `--rdma-rc-slots-per-qp 2`

Label fake-mode results as diagnostics, not normal throughput.

## Result Extraction

Total run throughput:

```bash
awk -F, 'NR>1 && $3=="run" {sum+=$18; n++} END {printf "run_sum_mkeys=%.3f clients=%d\n", sum/1000000, n}' results/<output_dir>/summary.csv
```

Server profile tail:

```bash
rg 'component=rdma_rc_server_profile' results/<output_dir>/logs/rdma/repeat_0/server/server_0.log | tail -n 5
rg 'component=rdma_rc_transport_profile role=server' results/<output_dir>/logs/rdma/repeat_0/server/server_0.log | tail -n 5
```

Client profile tail:

```bash
rg 'component=rdma_rc_client_profile' results/<output_dir>/logs/rdma/repeat_0/client_0.stdout.log | tail -n 5
```

Compare these fields first:

- throughput: `key_ops_per_sec / 1e6`
- server: `scan_hit_pct`, `handled_get`, `handle_get_avg_ns`,
  `complete_response_avg_ns`, `poll_loop_avg_ns`
- server transport: `complete_count`, `response_payload_bytes`,
  `complete_avg_ns`
- client: `submit_avg_ns`, `wait_status_avg_ns`, `pending_rpc_peak`

## Interpretation Notes

Low server `scan_hit_pct` does not by itself prove clients are slow. It means
pollers rarely observe slots in the `READY && seq > last_seq` state during
their scan. In the current RC design, clients publish requests with RDMA WRITE
to server memory and the server has no receive completion event; it discovers
requests by polling commit words.

For the current layout with p6/q16/slots1:

- total server slots = `num_clients * qps_per_client_per_shard * slots_per_qp`
- current total slots = `6 * 16 * 1 = 96`
- 16 server poll threads scan assigned QP lanes

If `pending_rpc_peak` reaches the configured outstanding limit but server
`scan_hit_pct` remains low, focus on request slot state visibility, slot
reuse, wait/revoke timing, and poller scan strategy before tuning PET_HASH.

