# PS RDMA Profile 瓶颈定位

日期：2026-05-30

## 目标

用当前 PS RDMA 内置 profile 跑一轮更长的 default / diagnostic 对照，定位当前 PS RDMA benchmark 的主要瓶颈。

本轮使用的是 RecStore RDMA 内置 profile：

- client：`component=rdma_rc_client_profile`
- server：`component=rdma_rc_server_profile`
- transport：`component=rdma_rc_transport_profile`
- benchmark transaction：`PS_BENCHMARK_PROFILE`

系统环境里没有找到 `perf`；当前 `petps_server` 也没有像 `ps_server` 那样直接接 `CPUPROFILE`/gperftools guard。因此本轮先用内置 RDMA profile 定位瓶颈。

## 执行方式

三条 benchmark 串行执行，没有并发跑 benchmark：

1. PS RDMA default，真实 512B payload。
2. PS RDMA `status_only`，server 只返回 status。
3. PS RDMA `index_only`，server parse keys + index lookup，不复制 value payload。

结果目录：

```text
results/benchmark_ps_profile_0530/
```

共同参数：

- `server_shards=1`
- `client_processes_per_ip=4`
- `client_threads_per_process=1`
- `client_load_threads_per_process=1`
- `record_count=200000`
- `value_size=512`
- `batch_keys=500`
- `runtime_seconds=8`
- `prefetch_depth=16`
- `rdma_rc_qps_per_client_per_shard=16`
- `rdma_rc_slots_per_qp=1`
- `rdma_rc_profile_interval_ms=1000`

## 结果摘要

| 场景 | 聚合吞吐 | 结果目录 |
|---|---:|---|
| PS RDMA default | `5.336M keys/s` | `results/benchmark_ps_profile_0530/ps_default_8s` |
| PS RDMA `status_only` | `28.723M keys/s` | `results/benchmark_ps_profile_0530/ps_status_only_8s` |
| PS RDMA `index_only` | `12.300M keys/s` | `results/benchmark_ps_profile_0530/ps_index_only_8s` |

## Server Profile

以下是 run 稳态窗口的加权平均：只统计 `handled_get > 0` 且 `handled_put=0`、`handled_init=0` 的窗口，按 `handled_get` 加权。

| 场景 | `handle_get_avg_ns` | `get_batch_get_avg_ns` | `get_row_copy_avg_ns` | `complete_response_avg_ns` | `missing_rows` | `scan_hit_pct` |
|---|---:|---:|---:|---:|---:|---:|
| default | `89.651us` | `89.327us` | `48.459us` | `2.914us` | `0` | `99.99%` |
| `index_only` | `37.676us` | `37.487us` | `0` | `2.184us` | `0` | `99.96%` |
| `status_only` | `0.070us` | `0` | `0` | `2.121us` | `0` | `0.81%` |

解读：

- default 的 server `handle_get` 约 `89.7us/batch`，每 batch 是 `500 * 512B = 256KB`。
- `index_only` 约 `37.7us/batch`，说明 key parse + index lookup 本身占了不小一块。
- default 比 `index_only` 多约 `52.0us/batch`，和 `get_row_copy_avg_ns=48.5us/batch` 基本吻合。
- `status_only` 的 `handle_get` 只有约 `70ns/batch`，说明去掉 index/value 后 server 业务处理几乎不是问题。
- default 和 index-only 的 `scan_hit_pct` 都接近 `100%`，说明稳态不是空轮询造成的。
- `missing_rows=0`，不是 KV 容量或 load 缺失导致的 zero-fill 慢。

## Client / Benchmark Profile

client 侧按 `wait_count` 加权，只统计 `pending_rpc_peak >= 16` 的 run 窗口。

| 场景 | `submit_avg_ns` | `wait_status_avg_ns` | `copy_response_avg_ns` | `revoke_avg_ns` | `pending_rpc_peak` | `acquire_qp_failures` |
|---|---:|---:|---:|---:|---:|---:|
| default | `4.326us` | `351.518us` | `0` | `0.469us` | `16` | `0` |
| `index_only` | `3.543us` | `148.620us` | `0` | `0.385us` | `16` | `0` |
| `status_only` | `6.417us` | `7.895us` | `0` | `0.335us` | `16` | `0` |

benchmark transaction profile：

| 场景 | `make_keys_avg_ns` | `submit_avg_ns` | `consume/wait_plus_result_avg_ns` |
|---|---:|---:|---:|
| default | `11.837us` | `6.432us` | `355.150us` |
| `index_only` | `11.844us` | `5.506us` | `144.125us` |
| `status_only` | `31.456us` | `21.076us` | `10.819us` |

解读：

- default 下 `pending_rpc_peak=16`，说明当前配置的 prefetch depth 已经打满，不是“完全没有并发”。
- `copy_response_avg_ns=0`，说明 borrowed RC response payload 生效，client 中间 payload copy 已被移出主路径。
- default 的 client wait 约 `351.5us`，index-only 约 `148.6us`，status-only 约 `7.9us`，和 server `handle_get` 分层一致。
- default 的 submit/revoke 都是微秒以下到几微秒级，不是主瓶颈。

## 当前瓶颈判断

当前 PS RDMA default 的主瓶颈在 server 真实 GET payload 路径，不在 client copy，也不在 prefetch 并发度不足。

具体链路是：

```text
PetPSServer::HandleGet
  -> CachePS::GetParameterFlat
  -> KVEngineComposite::BatchGetFlat
  -> DramValueStore::ReadFlatFixedRows
  -> response payload 写回
```

瓶颈拆分：

1. `index_only` 的 `37.7us/batch` 表示 index lookup / key handling 已经是固定成本。
2. default 额外增加约 `52us/batch`，主要由 `get_row_copy_avg_ns=48.5us/batch` 解释。
3. `complete_response_avg_ns` 只有约 `2-3us/batch`，不是最大项。
4. `status_only` 能到 `28.7M keys/s`，说明 adapter/prefetch 修复后，轻 payload pipeline 的上限已经明显抬高。
5. default 只有 `5.3M keys/s`，主要是 512B value payload 的 batch get + row copy 把 server 单 polling worker 打满。

所以这轮 profile 的结论是：

**当前主要瓶颈 = server GET 的真实 payload 读取/复制，尤其是 `KVEngineComposite::BatchGetFlat` 到 `DramValueStore::ReadFlatFixedRows` 这段；其次是 index lookup。不是 client response copy，也不是 prefetch depth 没打满。**

## 后续优化方向

优先级建议：

1. 继续压 server payload path：减少 `BatchGetFlat` 中 handle/vector 临时结构、row loop、固定 row size 判断的成本。
2. 对 `DramValueStore::ReadFlatFixedRows` 做更贴近连续内存/批量 copy 的实现，确认是否能减少 per-row 函数/分支开销。
3. 单独 profile `ProbeParameterIndex` / `BatchGetFlat` 的 index lookup 开销，因为 `index_only=37.7us/batch` 已经不低。
4. 如果要冲 PS default 的更高吞吐，需要考虑多 server polling worker 对同一 shard 的 GET 处理是否能安全并行，而不是只增加 client prefetch。

## 本轮命令

Default：

```bash
python3 src/test/scripts/run_benchmark_ps.py --transports rdma --client-ips 127.0.0.1 --server-shard-ips 127.0.0.1 --client-processes-per-ip 4 --record-count 200000 --value-size 512 --batch-keys 500 --client-threads-per-process 1 --client-load-threads-per-process 1 --runtime-seconds 8 --repeat 1 --execution-backend local --prefetch-depth 16 --rdma-rc-qps-per-client-per-shard 16 --rdma-rc-slots-per-qp 1 --rdma-rc-profile-interval-ms 1000 --rdma-wait-timeout-ms 20000 --client-timeout 240 --cluster-timeout 80 --transaction-profile --output-dir results/benchmark_ps_profile_0530/ps_default_8s
```

Status-only：

```bash
python3 src/test/scripts/run_benchmark_ps.py --transports rdma --client-ips 127.0.0.1 --server-shard-ips 127.0.0.1 --client-processes-per-ip 4 --record-count 200000 --value-size 512 --batch-keys 500 --client-threads-per-process 1 --client-load-threads-per-process 1 --runtime-seconds 8 --repeat 1 --execution-backend local --prefetch-depth 16 --rdma-rc-qps-per-client-per-shard 16 --rdma-rc-slots-per-qp 1 --rdma-rc-profile-interval-ms 1000 --rdma-wait-timeout-ms 20000 --client-timeout 240 --cluster-timeout 80 --transaction-profile --rdma-rc-fake-get-mode status_only --output-dir results/benchmark_ps_profile_0530/ps_status_only_8s
```

Index-only：

```bash
python3 src/test/scripts/run_benchmark_ps.py --transports rdma --client-ips 127.0.0.1 --server-shard-ips 127.0.0.1 --client-processes-per-ip 4 --record-count 200000 --value-size 512 --batch-keys 500 --client-threads-per-process 1 --client-load-threads-per-process 1 --runtime-seconds 8 --repeat 1 --execution-backend local --prefetch-depth 16 --rdma-rc-qps-per-client-per-shard 16 --rdma-rc-slots-per-qp 1 --rdma-rc-profile-interval-ms 1000 --rdma-wait-timeout-ms 20000 --client-timeout 240 --cluster-timeout 80 --transaction-profile --rdma-rc-fake-get-mode index_only --output-dir results/benchmark_ps_profile_0530/ps_index_only_8s
```
