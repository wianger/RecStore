# PS Transport Benchmark 矩阵结论（2026-05-29）

## 结论摘要

本轮重测把 `client_threads_per_process` 全部对齐为 `1`，避免把 RPC 多线程和 RDMA 单线程直接比较。测试层级是 **PS/network**，不是 storage-only，也不是模型端到端。

在本地单机、`value_size=512`、`batch_keys=500`、`runtime_seconds=3` 的配置下：

- 单 shard 下，RDMA 稳定在约 `2.95-3.01 M keys/s`，明显高于 GRPC/BRPC。
- RDMA 随 client 进程数从 `1` 增加到 `8` 基本不再提升，说明当前瓶颈不在单个 client 的线程数，而更可能在单 shard server 处理路径、RDMA PS benchmark 调度/拷贝路径或本地单机资源竞争。
- 2 shards 没有带来 RDMA 提升，反而下降到 `1.40-1.82 M keys/s`。这说明本地单机多 shard 不是网络扩容模型，而是额外 server 进程、路由、端口和 CPU 调度开销叠加在同一台机器上。
- RPC 在多 client 进程下有一定扩展性，但总体低于 RDMA；BRPC 在 `p>=4` 时略好于 GRPC。
- RDMA 多 client 的 per-client 吞吐不均衡比较明显，尤其单 shard `p=8`，总吞吐没有坏掉，但 client 间分配不均，说明存在调度/队列/轮询资源竞争。

## 测试环境与参数

- 代码分支：`feat/newrdma`
- 测试日期：2026-05-29
- 层级：PS/network
- 执行方式：local，`127.0.0.1`
- transports：`grpc,brpc,rdma`
- `record_count=200000`
- `value_size=512`
- `batch_keys=500`
- `client_threads_per_process=1`
- `client_load_threads_per_process=1`
- `runtime_seconds=3`
- `repeat=1`
- `server_worker_threads=32`
- `server_rdma_threads=1`
- `rdma_rc_qps_per_client_per_shard=16`
- raw results：`results/benchmark_ps_matrix_0529`

验证：

- `cmake --build build --target ps_transport_benchmark ps_server petps_server -j` 通过。
- `python3 -m unittest src/test/scripts/test_run_benchmark_ps.py` 通过，23 个测试 OK。
- PS 相关 CTest 通过，8/8。
- RDMA verbs 设备存在：`/dev/infiniband/uverbs*`。
- 本轮矩阵没有非空 `stderr.log`。

## 总吞吐矩阵

单位是 `M keys/s`。总吞吐为同一 transport 下所有 client 的 `key_ops_per_sec` 求和。

| server_shards | client_processes | client_threads_per_process | GRPC | BRPC | RDMA |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0.467 | 0.333 | 2.947 |
| 1 | 2 | 1 | 0.638 | 0.519 | 2.977 |
| 1 | 4 | 1 | 0.859 | 0.998 | 2.976 |
| 1 | 8 | 1 | 1.170 | 1.229 | 3.006 |
| 2 | 1 | 1 | 0.465 | 0.470 | 1.398 |
| 2 | 2 | 1 | 0.620 | 0.683 | 1.626 |
| 2 | 4 | 1 | 0.640 | 0.954 | 1.487 |
| 2 | 8 | 1 | 0.785 | 1.210 | 1.817 |

## Per-client 离散度

`min/max` 是同一 transport、同一 case 内单 client 的 run 吞吐范围，单位 `M keys/s`。

| case | GRPC min/max | BRPC min/max | RDMA min/max |
|---|---:|---:|---:|
| s1 p1 | 0.467 / 0.467 | 0.333 / 0.333 | 2.947 / 2.947 |
| s1 p2 | 0.318 / 0.320 | 0.257 / 0.262 | 1.478 / 1.499 |
| s1 p4 | 0.203 / 0.250 | 0.221 / 0.322 | 0.536 / 0.835 |
| s1 p8 | 0.107 / 0.406 | 0.125 / 0.328 | 0.003 / 0.962 |
| s2 p1 | 0.465 / 0.465 | 0.470 / 0.470 | 1.398 / 1.398 |
| s2 p2 | 0.308 / 0.312 | 0.341 / 0.342 | 0.744 / 0.882 |
| s2 p4 | 0.156 / 0.171 | 0.221 / 0.290 | 0.039 / 0.641 |
| s2 p8 | 0.079 / 0.231 | 0.110 / 0.416 | 0.065 / 0.481 |

## 现象解释

### 1. 为什么对齐线程后 RDMA 明显更高

RDMA 避免了 GRPC/BRPC 的通用 RPC 栈、序列化和内核网络路径开销。当前 fetch workload 是固定大小 value 读取，`batch_keys=500` 时每个请求携带足够多 key，RDMA 的批处理收益比较明显。因此在 `client_threads_per_process=1` 的公平基线下，RDMA 单 shard 单 client 已达到约 `2.95 M keys/s`，而 GRPC/BRPC 只有 `0.33-0.47 M keys/s`。

### 2. 为什么 RDMA 增加 client 进程后没有继续上涨

单 shard 下 RDMA 从 `p=1` 到 `p=8` 都在 `~3.0 M keys/s` 附近，说明瓶颈已经从 client 侧转移到共享资源：

- 单 shard server 的请求处理、KV 查找和结果准备路径可能达到上限。
- RDMA PS benchmark 不是纯 RC transport bandwidth 测试，还包含 PS 路由、batch 组织、结果拷贝和同步等待。
- 所有 server/client 都在同一台机器上，本地 CPU 调度、cache/memory 带宽和进程竞争会掩盖真实跨机 RDMA 网络扩展性。
- `server_rdma_threads=1`，polling/完成处理路径可能成为固定上限。

这解释了为什么继续加 client 进程只是在多个 client 之间重新分摊吞吐，而总吞吐没有线性增长。

### 3. 为什么 2 shards 的 RDMA 反而更低

这轮 2 shards 仍然跑在同一台机器。它不是“多机器多 shard 扩容”，而是本机启动更多 server shard 进程并让 client 做更多路由和连接管理。RDMA 结果从单 shard 的 `~3.0 M keys/s` 降到 `1.4-1.8 M keys/s`，说明本机多 shard 带来的额外调度和资源竞争大于并行收益。

跨机器测试时这个结论需要重测：如果每个 shard 在独立 server IP 上，2 shards 才可能代表真正的服务端水平扩展。

### 4. 为什么 RPC 多 client 有增长但仍低于 RDMA

GRPC/BRPC 单 client 单线程受 RPC 栈开销影响较大。增加 client 进程后，多个请求流可以并行覆盖一部分 RPC 往返和 server dispatch 开销，所以总吞吐从 `~0.3-0.5 M keys/s` 增加到 `~1.2 M keys/s`。但其每个 key 的协议和拷贝成本仍高于 RDMA，所以总体仍低。

### 5. 为什么 per-client 不均衡明显

RDMA 的 `p=8` case 中 client 间吞吐差异很大，单 shard 下最低 client 只有 `0.003 M keys/s`，最高接近 `0.962 M keys/s`。这说明本地多 client 并发下，部分 client 可能长期等在完成队列、server 处理或调度资源上。总吞吐没有下降，说明不是整体失败，而是公平性/调度问题。

## 后续建议

1. 如果目标是 PS RDMA 真实上限，应在多机器上跑：`client_ips` 传 client 机器列表，`server_shard_ips` 传每个 shard 的 server IP，避免本地单机资源竞争污染结论。
2. 如果继续本地诊断，优先扫：
   - `server_rdma_threads=1,2,4`
   - `rdma_rc_qps_per_client_per_shard=16,32,64`
   - `client_processes_per_ip=1,2,4,8`
3. 对 RDMA p=8 的不均衡，应增加 profile：看 server polling、client wait、copy、KV lookup 的耗时分布。
4. 当前报告取代旧的根目录 RDMA benchmark 报告；旧 raw 临时 compare 目录已经删除，仅保留本轮矩阵 raw data。
