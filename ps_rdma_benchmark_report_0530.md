# PS RDMA Benchmark 优化与瓶颈报告（2026-05-30）

## 摘要

本报告整合根目录旧的 PS/RDMA benchmark 矩阵、路径差异、profile 瓶颈分析和本轮 GET payload offload 实验。旧报告中的结论有时间顺序差异，本文件作为当前统一版本。

当前结论：

1. 原始 PS RDMA benchmark 的早期瓶颈是 adapter/prefetch 路径过重：每次 prefetch 都有大 receive buffer 分配/清零、prefetch map 管理和结果 materialize 成本。
2. 这几块已经通过 buffer pool、borrowed RC response payload、fixed-row flat read 等修复缓解；`status_only` 已经能到约 `28M keys/s`，说明轻 payload pipeline 不再是最早那种几 M 的调度上限。
3. 真实 512B value GET 的当前主瓶颈转移到 server GET payload 路径：`PetPSServer::HandleGet -> CachePS::GetParameterFlat -> KVEngineComposite::BatchGetFlat -> DramValueStore::ReadFlatFixedRows`。
4. 本轮新增的 `--rdma_rc_server_get_workers` 把 GET payload 读取/填充从 polling thread 拆出去，`p4/N4` 从约 `5.07M` 提升到约 `10.06M keys/s`。
5. 在 `p8/N8` 上继续增加 polling threads 后，当前最高单次结果为 `14.57M keys/s`；同条件 `C1` 三次复跑平均 `14.45M keys/s`，明显高于 `C4` 的 `12.17M keys/s`。所以 coroutine scanner 功能可用，但不是当前性能默认。

## 测试边界

- 分支：`feat/newrdma`
- 基准提交：`b613ee98`
- 层级：PS/network，不是 storage-only，也不是模型端到端。
- 运行方式：local，`client_ips=127.0.0.1`，`server_shard_ips=127.0.0.1`。
- RDMA verbs 设备存在：`/dev/infiniband/uverbs*`。
- 本地单机多 client / 多 shard 结果不能解释为跨机器扩容能力。

通用参数：

```text
transport=rdma
record_count=200000
value_size=512
batch_keys=500
client_threads_per_process=1
client_load_threads_per_process=1
prefetch_depth=16
rdma_rc_qps_per_client_per_shard=16
rdma_rc_slots_per_qp=1
rdma_rc_profile_interval_ms=1000
runtime_seconds=8
repeat=1
```

## 路径差异

Generic PS benchmark：

```text
run_benchmark_ps.py
  -> ps_transport_benchmark
  -> RDMAPSClientAdapter
  -> BasePSClient prefetch/result API
  -> PetPSClient
  -> petps_server
  -> CachePS / KVEngine
```

RDMA RC 专项 benchmark：

```text
run_rdma_rc_transport_benchmark.py
  -> rdma_rc_transport_benchmark
  -> PetPSClient / AllShardsParameterClientWrapper
  -> RC-write slot transport
  -> petps_server
```

两者不能直接混成一个层级。RC 专项路径更薄，历史高并发结果能到 `30-40M keys/s`，但它不包含 generic PS adapter 的 prefetch id 语义、result materialize、transactions workload 和完整 PS wrapper 生命周期。

## 旧矩阵结论

`benchmark_ps_transport_matrix_0529.md` 的公平线程矩阵显示，在 `runtime_seconds=3`、`client_threads_per_process=1` 下：

| server_shards | client_processes | GRPC | BRPC | RDMA |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.467 | 0.333 | 2.947 |
| 1 | 2 | 0.638 | 0.519 | 2.977 |
| 1 | 4 | 0.859 | 0.998 | 2.976 |
| 1 | 8 | 1.170 | 1.229 | 3.006 |
| 2 | 1 | 0.465 | 0.470 | 1.398 |
| 2 | 2 | 0.620 | 0.683 | 1.626 |
| 2 | 4 | 0.640 | 0.954 | 1.487 |
| 2 | 8 | 0.785 | 1.210 | 1.817 |

解读：RDMA 明显高于 RPC，但单 shard 增加 client 进程后没有继续上涨，说明当时瓶颈已经在 PS/server/polling/payload 路径，而不是单个 client 线程数。2 shards 本地单机更慢，是本机多 server 进程竞争，不是跨机扩容结论。

## Profile 瓶颈

`results/benchmark_ps_profile_0530` 的 8s profile：

| 场景 | 聚合吞吐 | `handle_get_avg_ns` | `get_batch_get_avg_ns` | `get_row_copy_avg_ns` | `complete_response_avg_ns` |
|---|---:|---:|---:|---:|---:|
| default | 5.336M | 89.651us | 89.327us | 48.459us | 2.914us |
| index_only | 12.300M | 37.676us | 37.487us | 0 | 2.184us |
| status_only | 28.723M | 0.070us | 0 | 0 | 2.121us |

关键判断：

- `status_only` 到 `28.7M keys/s`，说明 adapter/prefetch 大洞已经明显缓解。
- `index_only` 到 `12.3M keys/s`，说明 key parse + index lookup 已经是可见固定成本。
- default 比 index-only 多约 `52us/batch`，与 `get_row_copy_avg_ns=48.5us/batch` 基本吻合。
- `copy_response_avg_ns=0`，说明 borrowed RC response payload 生效，client 中间 response copy 不再是主墙。
- `missing_rows=0`，不是 KVEngine 容量不足或 load 缺失导致的假慢。

## 本轮代码改动

新增参数：

```text
--rdma_rc_server_get_workers=N
```

语义：

- `0`：保持同步路径，polling thread 直接执行 `HandleGet`。
- `>0`：polling thread 只验证 GET descriptor、投递 `GetPayloadTask`，payload worker 执行 `HandleGet` 并填 response staging buffer，原 polling thread drain completion 后调用 `CompleteResponse`。

相关代码位置：

- `src/ps/rdma/petps_server.cc`
  - `PetPSServer::GetPayloadWorkerLoop`
  - `PetPSServer::EnqueueGetPayloadTask`
  - `PetPSServer::DrainGetPayloadCompletions`
  - `PetPSServer::CompleteResponseForSlot`
  - `PetPSServer::ProcessSlot`
- `src/ps/rdma/rc_options.cc`
- `src/ps/rdma/rc_options.h`
- `src/test/scripts/petps_cluster_runner.py`
- `src/test/scripts/run_benchmark_ps.py`
- `src/test/scripts/run_rdma_rc_transport_benchmark.py`
- `src/test/scripts/run_rdma_transport_benchmarks.py`

多 poller ownership：

- `ScanAssignedSlots` 按 `qp_index % thread_count` 分配 lane。
- GET task 记录 `poll_thread_id`。
- completion 按 `poll_thread_id` 分桶。
- 只有原始 poller drain 自己的 completion 并调用 `RcShardServerTransport::CompleteResponse`，避免多个 poller 同时操作同一 lane 的 pending response 状态。
- 最新代码已经允许 `rdma_rc_server_coroutines_per_thread > 1` 与 get workers 组合；coroutine scanner 每轮扫描前后都会 drain 原 poller 的 GET completion。
- 多 shard client wait 也改成 Boost coroutine cooperative wait：每个 shard RPC 一个 waiter，未完成时 yield，完成后再 assemble batch。

## GET worker 实验

核心命令模板：

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip <P> \
  --record-count 200000 \
  --value-size 512 \
  --batch-keys 500 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 8 \
  --repeat 1 \
  --execution-backend local \
  --prefetch-depth 16 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-slots-per-qp 1 \
  --rdma-rc-profile-interval-ms 1000 \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 240 \
  --cluster-timeout 80 \
  --transaction-profile \
  --server-rdma-threads <T> \
  --rdma-rc-server-get-workers <N> \
  --rdma-rc-server-coroutines-per-thread <C> \
  --output-dir <DIR>
```

参数标记：

- `T`：server RDMA polling thread 数。
- `N`：server GET payload worker 数。
- `C`：每个 polling thread 内的 Boost coroutine scanner 数。历史结果没有显式设置时等价于 `C=1`。

`p4` 结果：

| client processes | get workers | poll threads | 聚合吞吐 | 结果目录 |
|---:|---:|---:|---:|---|
| 4 | 0 | 1 | 5.067780M | `results/benchmark_ps_get_workers_0530/n0_p4` |
| 4 | 4 | 1 | 10.057410M | `results/benchmark_ps_get_workers_0530_tid_n4_p4` |
| 4 | 5 | 1 | 6.447510M | `results/benchmark_ps_get_workers_0530_tid_n5_p4` |
| 4 | 6 | 1 | 5.058550M | `results/benchmark_ps_get_workers_0530_tid_n6_p4` |

`p6` 结果：

| client processes | get workers | poll threads | 聚合吞吐 | 结果目录 |
|---:|---:|---:|---:|---|
| 6 | 4 | 1 | 9.600980M | `results/benchmark_ps_get_workers_shared_0530_n4_p6` |
| 6 | 5 | 1 | 5.968234M | `results/benchmark_ps_get_workers_shared_0530_n5_p6` |
| 6 | 6 | 1 | 6.186659M | `results/benchmark_ps_get_workers_shared_0530_n6_p6` |

`p8` worker 扫描：

| client processes | get workers | poll threads | 聚合吞吐 | min/max client | 结果目录 |
|---:|---:|---:|---:|---:|---|
| 8 | 4 | 1 | 9.524970M | 1.037M / 1.226M | `results/benchmark_ps_get_workers_shared_0530_n4_p8` |
| 8 | 5 | 1 | 7.014096M | 0.477M / 1.139M | `results/benchmark_ps_get_workers_shared_0530_n5_p8` |
| 8 | 6 | 1 | 9.855701M | 0.814M / 1.396M | `results/benchmark_ps_get_workers_shared_0530_n6_p8` |
| 8 | 7 | 1 | 8.703887M | 0.466M / 1.477M | `results/benchmark_ps_get_workers_shared_0530_n7_p8` |
| 8 | 8 | 1 | 13.968240M | 1.408M / 2.035M | `results/benchmark_ps_get_workers_shared_0530_n8_p8` |
| 8 | 8 | 1 | 10.782787M | 0.511M / 1.746M | `results/benchmark_ps_get_workers_shared_0530_n8_p8_rerun` |
| 8 | 12 | 1 | 7.258410M | 0.469M / 1.023M | `results/benchmark_ps_get_workers_shared_0530_n12_p8` |
| 8 | 16 | 1 | 6.676363M | 0.503M / 0.896M | `results/benchmark_ps_get_workers_shared_0530_n16_p8` |

解读：

- `N4` 在 `p4` 下最稳，约 2 倍提升。
- `N5/N6` 在 `p4/p6` 下退化，说明不是简单加 worker 就能线性提升。
- `p8/N8` 有明显高点，但复跑波动较大。
- `N12/N16` 明显退化，profile 中 `handle_get_avg_ns` 暴涨到 `412us/596us`，说明 worker 过多后出现 CPU/cache/memory 或 KVEngine 内部竞争。

## Poll thread 实验

为了保持同条件，这组保留 `p8/N8`，只改变 `server-rdma-threads`：

| client processes | get workers | poll threads | 聚合吞吐 | min/max client | 结果目录 |
|---:|---:|---:|---:|---:|---|
| 8 | 8 | 1 | 13.968240M | 1.408M / 2.035M | `results/benchmark_ps_get_workers_shared_0530_n8_p8` |
| 8 | 8 | 1 | 10.782787M | 0.511M / 1.746M | `results/benchmark_ps_get_workers_shared_0530_n8_p8_rerun` |
| 8 | 8 | 2 | 12.796790M | 1.203M / 1.735M | `results/benchmark_ps_get_workers_shared_0530_n8_p8_t2` |
| 8 | 8 | 4 | 13.352040M | 1.150M / 1.998M | `results/benchmark_ps_get_workers_shared_0530_n8_p8_t4` |
| 8 | 8 | 8 | 14.370420M | 1.760M / 1.844M | `results/benchmark_ps_get_workers_shared_0530_n8_p8_t8` |
| 8 | 8 | 8 | 14.198180M | 1.529M / 1.981M | `results/benchmark_ps_profile_0530_n8_p8_t8_rerun` |
| 8 | 8 | 12 | 14.506580M | 1.564M / 2.045M | `results/benchmark_ps_profile_0530_n8_p8_t12` |
| 8 | 8 | 16 | 14.571230M | 1.138M / 2.138M | `results/benchmark_ps_profile_0530_n8_p8_t16` |

对应 server profile 摘要：

| get workers | poll threads | `handle_get_avg_ns` | `get_row_copy_avg_ns` | `complete_response_avg_ns` | `poll_loop_avg_ns` |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 189.387us | 102.746us | 3.361us | 23.601us |
| 8 | 4 | 186.283us | 102.144us | 4.149us | 12.552us |
| 8 | 8 | 178.531us | 93.766us | 4.954us | 15.454us |
| 8 | 12 | 172.041us | 87.998us | 4.738us | 22.006us |
| 8 | 16 | 167.789us | 83.284us | 4.689us | 25.194us |

解读：

- 增加 poll threads 在 `p8/N8` 下有帮助，最高单次到 `14.57M keys/s`。
- `T8` per-client 最均衡；`T12/T16` 继续提高 aggregate，但 `T16` 的 client 间差异变大。
- `complete_response_avg_ns` 随 poller 增加略高，但总体不是最大项。
- 这仍是本地单机、repeat=1 的结果，需要 repeat=3/5 或 CPU affinity 后才能固化为默认推荐；当前候选可以按“吞吐优先 T16、稳定优先 T12”区分。

## 继续扩并发的复测

在当前较优的 `p8/N8/T8` 或 `p8/N8/T16` 基础上，又补了三类实验：

| 实验 | 参数变化 | 聚合吞吐 | 结果目录 |
|---|---|---:|---|
| 增加 client 进程 | `p8 -> p12`, `N8/T8/depth16/slots1` | 13.700087M | `results/benchmark_ps_get_workers_shared_0530_n8_p12_t8` |
| 增加单 client in-flight | `prefetch_depth=16 -> 32`, `slots_per_qp=1 -> 2`, `p8/N8/T8` | 8.865387M | `results/benchmark_ps_profile_0530_n8_p8_t8_depth32_slots2` |
| 增加 GET workers | `N8 -> N12`, `p8/T16/depth16/slots1` | 9.509550M | `results/benchmark_ps_profile_0530_n12_p8_t16` |

结论：

- 增加 client 进程到 `p12` 没有提升，`scan_hit_pct` 从 `0.345%` 降到 `0.272%`，说明更多 client 主要增加扫描和调度压力。
- 增加单 client in-flight 到 `depth32/slots2` 明显退化：server `scanned_slots` 翻倍，但 `handled_get` 下降，`scan_hit_pct` 从约 `0.410%` 降到 `0.123%`。
- 在 `T16` 下继续把 GET workers 加到 `N12` 也退化：`handle_get_avg_ns` 从 `168us` 升到 `282us`，说明 worker 过量后 CPU/cache/memory bandwidth 或 KVEngine 内部共享结构竞争变重。

## Coroutine scanner 复测

这组只改变每个 polling thread 内的 coroutine scanner 数 `C`，其余都保持一致：

```text
client_processes=8
rdma_rc_server_get_workers=8
server_rdma_threads=16
prefetch_depth=16
rdma_rc_qps_per_client_per_shard=16
rdma_rc_slots_per_qp=1
runtime_seconds=8
repeat=3
```

| C | repeat 0 | repeat 1 | repeat 2 | 三轮平均 | min/max aggregate | 结果目录 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 14.427890M | 14.542330M | 14.387140M | 14.452453M | 14.387140M / 14.542330M | `results/benchmark_ps_profile_0530_n8_p8_t16_c1_repeat3` |
| 4 | 12.246780M | 11.728780M | 12.541760M | 12.172440M | 11.728780M / 12.541760M | `results/benchmark_ps_profile_0530_n8_p8_t16_c4_repeat3` |

尾部 server profile 均值：

| C | `handle_get_avg_ns` | `get_row_copy_avg_ns` | `complete_response_avg_ns` | `poll_loop_avg_ns` | `scan_hit_pct` |
|---:|---:|---:|---:|---:|---:|
| 1 | 163-166us | 83-87us | 4.6-4.7us | 24.6-25.5us | 0.58-0.60% |
| 4 | 161-165us | 81-83us | 4.6-4.7us | 23.1-25.1us | 1.83-1.90% |

解读：

- `C4` 会提高扫描命中比例，但吞吐反而下降，说明当前损失主要来自 coroutine 调度开销或 poll loop cadence 被改变，而不是 GET payload 本身更慢。
- 两组 `handle_get` 和 `get_row_copy` 基本同量级，瓶颈仍在 server GET payload path。
- `C1` 是当前推荐性能默认；`C>1` 只作为扫描调度实验维度，结果目录需要显式带 `c<C>`。

## 当前最优参数 Profile

当前最高单次结果是：

```text
client_processes=8
rdma_rc_server_get_workers=8
server_rdma_threads=16
prefetch_depth=16
rdma_rc_qps_per_client_per_shard=16
rdma_rc_slots_per_qp=1
```

结果目录：

```text
results/benchmark_ps_profile_0530_n8_p8_t16
```

吞吐：

```text
aggregate = 14.571230M keys/s
min_client = 1.138M keys/s
max_client = 2.138M keys/s
```

同参数 `C1` repeat=3 复跑结果为：

```text
results/benchmark_ps_profile_0530_n8_p8_t16_c1_repeat3
aggregate_avg = 14.452453M keys/s
aggregate_min = 14.387140M keys/s
aggregate_max = 14.542330M keys/s
```

加权 profile：

| 层级 | 指标 | 数值 |
|---|---|---:|
| server | `handle_get_avg_ns` | 167.512us |
| server | `get_batch_get_avg_ns` | 166.766us |
| server | `get_row_copy_avg_ns` | 83.267us |
| server | `complete_response_avg_ns` | 4.643us |
| server | `poll_loop_avg_ns` | 25.270us |
| server | `scan_hit_pct` | 0.594% |
| client | `submit_avg_ns` | 39.702us |
| client | `wait_status_avg_ns` | 107.615us |
| client | `copy_response_avg_ns` | 0 |
| client | `acquire_qp_failures` | 0 |
| server transport | `complete_avg_ns` | 4.187us |
| server transport | `drain_response_avg_ns` | 0.889us |

当前最优参数下的瓶颈仍然是 server GET payload 路径：

```text
PetPSServer::GetPayloadWorkerLoop
  -> PetPSServer::HandleGet
  -> CachePS::GetParameterFlat
  -> KVEngineComposite::BatchGetFlat
  -> DramValueStore::ReadFlatFixedRows
```

一次 batch 是 `500 * 512B = 256KB`。当前每 batch 的 `HandleGet` 约 `167.5us`，其中 `get_row_copy` 约 `83.3us`，约占一半。`complete_response` 只有 `4-5us`，server transport complete 也只有约 `4.2us`，client `copy_response_avg_ns=0`，因此当前不是 RDMA completion 或 client copy 主导。

## 当前瓶颈排序

已经缓解：

1. adapter 每 prefetch 分配 receive buffer + 整块 memset。
2. `GetPrefetchResultFlat` 的 `vector::assign` 额外开销。
3. `PetPSClient::WaitRPCFinish` 的中间 response copy。
4. DRAM fixed-row flat read 的部分 per-row 开销。

当前主要瓶颈：

1. server GET payload copy / `BatchGetFlat` / `ReadFlatFixedRows`。
2. index lookup / key handling 固定成本。
3. polling/completion 调度能力，特别是高 client 并发下单 poller 对 lane 扫描和 response completion 的限制；`T12/T16` 能缓解但已经接近平台期。
4. worker 数过高、client in-flight 过深或 client 进程过多后的 CPU/cache/memory contention。

## 代码风险与后续建议

当前实现仍是 benchmark-safe 的实验开关：

- 默认 `--rdma_rc_server_get_workers=0`，不改变现有同步路径。
- 只 offload GET，PUT / UPDATE / InitTable 保持同步路径。
- get workers 已可与 coroutine scanning 组合；但 `p8/N8/T16` 下 `C4` repeat=3 平均 `12.17M keys/s`，低于 `C1` 的 `14.45M keys/s`。因此 `C1` 保持默认，`C>1` 只作为显式实验维度，结果目录需要带上 `c<C>`。
- completion 保留由 poller 调用 `CompleteResponse`，避免 worker 直接操作 transport lane state。

建议下一步：

1. 若继续研究 coroutine scanner，先加 CPU affinity/NUMA 绑定后再做 `C2/C4` repeat=3/5；当前无绑定本地结果已经显示 `C4` 不适合作为性能默认。
2. 对 `HandleGet` 内部继续 profile：拆 `CachePS::GetParameterFlat`、`KVEngineComposite::BatchGetFlat`、`DramValueStore::ReadFlatFixedRows`。
3. 优先优化 `DramValueStore::ReadFlatFixedRows` 的 fixed-row copy，或者让 KVEngine 更直接写入 RDMA response staging buffer。
4. 如果要接近 RC 专项 `30-40M keys/s`，需要继续减少 generic PS API 的 result materialize 和 server payload copy 成本；仅靠加 worker/coroutine 不会线性到达。

## 验证记录

本轮已经跑过：

```bash
cmake --build build --target petps_server -j
python3 src/test/scripts/run_benchmark_ps.py ... --server-rdma-threads 2 --rdma-rc-server-get-workers 2 --output-dir results/rdma_get_workers_poll_threads_smoke_0530
python3 src/test/scripts/run_benchmark_ps.py ... --server-rdma-threads 2 --rdma-rc-server-get-workers 8 --output-dir results/benchmark_ps_get_workers_shared_0530_n8_p8_t2
python3 src/test/scripts/run_benchmark_ps.py ... --server-rdma-threads 4 --rdma-rc-server-get-workers 8 --output-dir results/benchmark_ps_get_workers_shared_0530_n8_p8_t4
python3 src/test/scripts/run_benchmark_ps.py ... --server-rdma-threads 8 --rdma-rc-server-get-workers 8 --output-dir results/benchmark_ps_get_workers_shared_0530_n8_p8_t8
python3 src/test/scripts/run_benchmark_ps.py ... --server-rdma-threads 16 --rdma-rc-server-get-workers 8 --rdma-rc-server-coroutines-per-thread 4 --repeat 3 --output-dir results/benchmark_ps_profile_0530_n8_p8_t16_c4_repeat3
python3 src/test/scripts/run_benchmark_ps.py ... --server-rdma-threads 16 --rdma-rc-server-get-workers 8 --rdma-rc-server-coroutines-per-thread 1 --repeat 3 --output-dir results/benchmark_ps_profile_0530_n8_p8_t16_c1_repeat3
git diff --check
```

最终提交前还应重跑：

```bash
python3 -m unittest \
  src/test/scripts/test_petps_cluster_runner.py \
  src/test/scripts/test_run_benchmark_ps.py \
  src/test/scripts/test_run_rdma_rc_transport_benchmark.py \
  src/test/scripts/test_run_rdma_transport_benchmarks.py

cmake --build build --target petps_server test_allshards_ps_client rdma_rc_transport_benchmark ps_transport_benchmark -j

ctest --test-dir build -R 'test_allshards_ps_client|test_rdma_rc_protocol|test_raw_verbs_allocator|test_rdmaps_client_adapter' -VV
```
