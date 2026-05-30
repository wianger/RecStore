# PS Benchmark 与 RDMA RC 专项 Benchmark 路径对比（2026-05-29）

## 结论摘要

当前仓库里有两条容易被混在一起讨论的 RDMA benchmark 路径：

- `run_benchmark_ps.py` + `ps_transport_benchmark`：这是 generic PS/network 层测试，走 `RDMAPSClientAdapter`、distributed client config、transactions workload、load/run 两阶段和上层 fetch pipeline。2026-05-29 本地公平矩阵里，RDMA 单 shard 约 `2.95-3.01 M keys/s`。
- `run_rdma_rc_transport_benchmark.py` + `rdma_rc_transport_benchmark`：这是 RDMA RC 专项传输闭环测试，直接压 `PetPSClient` / `AllShardsParameterClientWrapper` 的 RC slot transport，`async_stream16` 多 client 聚合在 `results/rdma_max_0529004246` 中能到 `33.99-36.40 M keys/s`。

这两个结果不能直接当成同一层的性能差距。后者是更薄的 RC transport 专项压测，前者包含 PS 路由、key 生成、prefetch/result 语义、adapter 生命周期、KV 查找、response copy、load 阶段和本地多进程调度等额外成本。

## 构建与通用前置检查

```bash
cd /app/RecStore
cmake -S . -B build
cmake --build build --target ps_transport_benchmark rdma_rc_transport_benchmark ps_server petps_server -j
python3 -m unittest src/test/scripts/test_run_benchmark_ps.py
python3 -m unittest src/test/scripts/test_run_rdma_rc_transport_benchmark.py
```

真实 RDMA 路径需要 verbs 设备：

```bash
test -d /dev/infiniband
ls /dev/infiniband/uverbs*
```

如果 `petps_server` 报 `librdkafka.so.1` 找不到，先补运行时库路径：

```bash
export LD_LIBRARY_PATH=/app/RecStore/build/lib:${LD_LIBRARY_PATH}
```

## 路径 A：Generic PS/network Benchmark

### 入口和用途

- runner：`src/test/scripts/run_benchmark_ps.py`
- benchmark binary：`build/bin/ps_transport_benchmark`
- server binary：
  - RDMA：`build/bin/petps_server`
  - GRPC/BRPC：`build/bin/ps_server`
- 适用层级：PS/network
- 典型结果目录：`results/benchmark_ps_matrix_0529`
- 对应报告：`benchmark_ps_transport_matrix_0529.md`

### 本地公平矩阵复现命令

单 shard、单 client 进程、RDMA/GRPC/BRPC 同线程公平对比：

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --transports grpc,brpc,rdma \
  --server-shard-ips 127.0.0.1 \
  --client-ips 127.0.0.1 \
  --client-processes-per-ip 1 \
  --record-count 200000 \
  --value-size 512 \
  --batch-keys 500 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --server-worker-threads 32 \
  --server-rdma-threads 1 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --runtime-seconds 3 \
  --repeat 1 \
  --execution-backend local \
  --output-dir results/benchmark_ps_matrix_0529/b500_s1_p1_t1
```

单 shard、4 client 进程：

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --transports grpc,brpc,rdma \
  --server-shard-ips 127.0.0.1 \
  --client-ips 127.0.0.1 \
  --client-processes-per-ip 4 \
  --record-count 200000 \
  --value-size 512 \
  --batch-keys 500 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --server-worker-threads 32 \
  --server-rdma-threads 1 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --runtime-seconds 3 \
  --repeat 1 \
  --execution-backend local \
  --output-dir results/benchmark_ps_matrix_0529/b500_s1_p4_t1
```

2 shards、4 client 进程，本地单机压力测试：

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --transports grpc,brpc,rdma \
  --server-shard-ips 127.0.0.1,127.0.0.1 \
  --client-ips 127.0.0.1 \
  --client-processes-per-ip 4 \
  --record-count 200000 \
  --value-size 512 \
  --batch-keys 500 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --server-worker-threads 32 \
  --server-rdma-threads 1 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --runtime-seconds 3 \
  --repeat 1 \
  --execution-backend local \
  --output-dir results/benchmark_ps_matrix_0529/b500_s2_p4_t1
```

### 已有结果摘录

`results/benchmark_ps_matrix_0529/b500_s1_p1_t1/summary.csv`：

| transport | server_shards | client_processes | client_threads | run M keys/s |
|---|---:|---:|---:|---:|
| GRPC | 1 | 1 | 1 | 0.467 |
| BRPC | 1 | 1 | 1 | 0.333 |
| RDMA | 1 | 1 | 1 | 2.947 |

`benchmark_ps_transport_matrix_0529.md` 中的总吞吐矩阵显示：

| server_shards | client_processes | RDMA M keys/s |
|---:|---:|---:|
| 1 | 1 | 2.947 |
| 1 | 2 | 2.977 |
| 1 | 4 | 2.976 |
| 1 | 8 | 3.006 |
| 2 | 1 | 1.398 |
| 2 | 2 | 1.626 |
| 2 | 4 | 1.487 |
| 2 | 8 | 1.817 |

解读边界：

- 这是 PS/network 层，不是纯 RC transport。
- `server_shard_ips=127.0.0.1,127.0.0.1` 是本地多 server 进程压力，不是跨机器 shard 扩容。
- RDMA transactions 当前要求 `--client-threads-per-process=1`，并通过 `--client-processes-per-ip` 扩 client 并发。

## 路径 B：RDMA RC 专项 Benchmark（30-40M keys/s 路径）

### 入口和用途

- runner：`src/test/scripts/run_rdma_rc_transport_benchmark.py`
- benchmark binary：`build/bin/rdma_rc_transport_benchmark`
- server binary：`build/bin/petps_server`
- 适用层级：RDMA RC transport / PetPS 专项闭环
- 典型结果目录：`results/rdma_max_0529004246`

这条路径主要压：

```text
rdma_rc_transport_benchmark
  -> PetPSClient 或 AllShardsParameterClientWrapper
  -> RC-write slot transport
  -> petps_server
```

它不走 generic PS adapter，也不走 `transactions` 的随机 key 生成和 PS wrapper 生命周期。

### 单 client smoke

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 1 \
  --thread-num 1 \
  --iterations 1600 \
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
  --quiet
```

已有结果：`results/rdma_max_0529004246/rc_smoke_c1_t1_b16_d16_qp16.log`

| clients | thread_num | batch_keys | async_depth | agg M keys/s |
|---:|---:|---:|---:|---:|
| 1 | 1 | 16 | 16 | 2.394 |

### 多 client 高吞吐命令

16 clients、16 server polling threads：

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 16 \
  --thread-num 16 \
  --iterations 1280 \
  --rounds 4 \
  --warmup-rounds 1 \
  --batch-keys 500 \
  --value-size 512 \
  --op async_stream \
  --async-depth 16 \
  --qps-per-client-per-shard 16 \
  --report-mode summary \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 300 \
  --cluster-timeout 60 \
  --quiet
```

对应结果：`results/rdma_max_0529004246/rc_c16_t16_b500_d16_qp16.log`

| clients | thread_num | batch_keys | async_depth | agg M keys/s |
|---:|---:|---:|---:|---:|
| 16 | 16 | 500 | 16 | 33.993 |

16 clients、24 server polling threads：

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 16 \
  --thread-num 24 \
  --iterations 1280 \
  --rounds 4 \
  --warmup-rounds 1 \
  --batch-keys 500 \
  --value-size 512 \
  --op async_stream \
  --async-depth 16 \
  --qps-per-client-per-shard 16 \
  --report-mode summary \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 300 \
  --cluster-timeout 60 \
  --quiet
```

对应结果：`results/rdma_max_0529004246/rc_c16_t24_b500_d16_qp16.log`

| clients | thread_num | batch_keys | async_depth | agg M keys/s |
|---:|---:|---:|---:|---:|
| 16 | 24 | 500 | 16 | 36.399 |

### 参数扫描摘录

来自 `results/rdma_max_0529004246`：

| log | clients | thread_num | batch_keys | async_depth | agg M keys/s |
|---|---:|---:|---:|---:|---:|
| `rc_c16_t4_b500_d16_qp16.log` | 16 | 4 | 500 | 16 | 14.336 |
| `rc_c16_t8_b500_d16_qp16.log` | 16 | 8 | 500 | 16 | 27.579 |
| `rc_c16_t16_b500_d16_qp16.log` | 16 | 16 | 500 | 16 | 33.993 |
| `rc_c16_t24_b500_d16_qp16.log` | 16 | 24 | 500 | 16 | 36.399 |
| `rc_c32_t8_b500_d16_qp16.log` | 32 | 8 | 500 | 16 | 18.758 |

这说明专项 RC 路径对 `thread_num` 很敏感；继续堆 client 不一定更好，`c32_t8` 反而低于 `c16_t16/t24`。

## 具体路径差异

### 1. Client API 和 wrapper 层不同

RDMA RC 专项 benchmark：

```text
rdma_rc_transport_benchmark.cc
  -> BenchmarkClient
  -> PetPSClient("127.0.0.1", 1234, shard)
  -> client->PutParameter(...)
  -> client->GetParameter(..., async=true, ...)
  -> client->WaitRPCFinish(...)
  -> client->RevokeRPCResource(...)
```

Generic PS benchmark：

```text
run_benchmark_ps.py
  -> ps_transport_benchmark --workload=transactions
  -> CreateBenchmarkClient("RDMA")
  -> RDMAPSClientAdapter
  -> distributed_client config / shard routing
  -> PrefetchParameter(...)
  -> GetPrefetchResultFlat(...)
  -> PetPSClient / AllShards wrapper
  -> RC-write slot transport
```

差异核心：专项 RC 直接压 PetPS client；generic PS 先经过统一 PS client/adapter 语义。

### 2. Workload 不同

RDMA RC 专项：

- 固定 key 集合：`1000001 + i`。
- 每轮循环重复同一批 key。
- `async_stream` 会保持固定 async depth，在一个 client 进程内持续提交/回收。
- 统计的是每轮 `iterations * async_depth * batch_keys`。

Generic PS transactions：

- 先 `load`，再 `run`。
- run 阶段按 `record_count` 和 `distribution` 生成随机 key。
- fetch 模式默认 RDMA prefetch depth 为 `16`，但还要经过 `PrefetchParameter` / `GetPrefetchResultFlat` 的上层接口。
- 统计的是指定 `running_seconds` 内完成的 transaction batches。

### 3. Server 线程参数含义相近但入口不同

RDMA RC 专项：

- `--thread-num` 传给 `PetPSClusterRunner.thread_num`，控制 `petps_server` RDMA polling thread 数。
- 高吞吐结果中 `thread_num=16/24`。

Generic PS benchmark：

- `--server-rdma-threads` 控制 RDMA server polling thread 数。
- 0529 公平矩阵固定为 `server_rdma_threads=1`。
- `--server-worker-threads=32` 是 PS/KV worker 配置，不等于 RDMA polling thread。

因此 `36M keys/s` 的 `thread_num=24` 不能和 `3M keys/s` 的 `server_rdma_threads=1` 直接相除得出“上层慢 12 倍”的结论。这里同时改变了层级和 server polling 资源。

### 4. 并发模型不同

RDMA RC 专项：

- `--client-count` 是 runner 启动的 benchmark client 进程数。
- 每个 client 用 `async_stream16` 保持 16 个在途请求。
- `qps_per_client_per_shard * slots_per_qp >= async_depth` 是硬约束。

Generic PS benchmark：

- `--client-processes-per-ip` 扩 client 进程。
- RDMA transactions 目前要求每个进程 `--client-threads-per-process=1`。
- fetch 默认 prefetch depth 是 16，但 pending/consume 发生在 PS transactions 的 fetch pipeline 内。

### 5. 结果解释边界不同

`rdma_rc_transport_benchmark` 的 30-40M keys/s 说明：

- 当前 RC slot transport 和 PetPS 专项路径在足够 server polling thread、足够 client async stream 下，能提供更高的传输闭环上限。
- 它适合做 transport baseline 和低层瓶颈排查。

它不说明：

- generic PS transactions 已经能到 30-40M keys/s。
- PyTorch/model 层能到 30-40M rows/s。
- 多 shard 跨机器扩展已经验证。

`run_benchmark_ps.py` 的 2.95-3.01M keys/s 说明：

- 在本地单机、`server_rdma_threads=1`、generic PS transactions/fetch 路径下，当前上层闭环约在 3M keys/s。
- 增加本地 client 进程后没有继续增长，瓶颈更像 server polling/PS 路径/本机资源竞争，而不是 client 数不足。

它不说明：

- 底层 RC transport 只能到 3M keys/s。
- 多机多 shard 扩容不可行。

## 建议的下一步对齐测试

为了把两条路径拉到更接近的对照条件，建议先跑下面两组。

### A. Generic PS 提高 RDMA polling threads

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --transports rdma \
  --server-shard-ips 127.0.0.1 \
  --client-ips 127.0.0.1 \
  --client-processes-per-ip 16 \
  --record-count 200000 \
  --value-size 512 \
  --batch-keys 500 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --server-worker-threads 32 \
  --server-rdma-threads 16 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-profile-interval-ms 1000 \
  --runtime-seconds 3 \
  --repeat 1 \
  --execution-backend local \
  --output-dir results/benchmark_ps_align_0529/rdma_1s16c_rdmath16
```

再把 `--server-rdma-threads` 改成 `24` 跑一轮：

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --transports rdma \
  --server-shard-ips 127.0.0.1 \
  --client-ips 127.0.0.1 \
  --client-processes-per-ip 16 \
  --record-count 200000 \
  --value-size 512 \
  --batch-keys 500 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --server-worker-threads 32 \
  --server-rdma-threads 24 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-profile-interval-ms 1000 \
  --runtime-seconds 3 \
  --repeat 1 \
  --execution-backend local \
  --output-dir results/benchmark_ps_align_0529/rdma_1s16c_rdmath24
```

这能回答：generic PS 低吞吐里有多少来自 `server_rdma_threads=1`。

### B. RDMA RC 降到 generic PS 的 polling 配置

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 16 \
  --thread-num 1 \
  --iterations 1280 \
  --rounds 4 \
  --warmup-rounds 1 \
  --batch-keys 500 \
  --value-size 512 \
  --op async_stream \
  --async-depth 16 \
  --qps-per-client-per-shard 16 \
  --report-mode summary \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 300 \
  --cluster-timeout 60 \
  --quiet
```

这能回答：在同样只有一个 RDMA polling thread 时，专项 RC 路径还剩多少吞吐。

## 读数口径

- PS/network 结果用 `summary.csv` 中 `phase=run` 的 `key_ops_per_sec / 1e6`。
- RDMA RC 专项结果用 aggregate 表里的 `agg_key_ops/s / 1e6`。
- 不要把 `load` 阶段吞吐和 `run` 阶段吞吐混在一起。
- 不要把 `status_only`、`skip_client_copy`、`fake_get_mode` 诊断行当成默认 GET 路径。
- 多 client 聚合要同时看 per-client 离散度，否则会掩盖调度不公平。
