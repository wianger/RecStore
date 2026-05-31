# PS RDMA Benchmark 瓶颈报告（2026-05-30）

本文只讨论 PS/network 层的 RDMA GET benchmark，不讨论 storage-only 或模型端到端性能。当前结论来自本地单机测试，不能直接解释为跨机器扩展能力。

## 结论

当前 512B value GET 的主要瓶颈在 server GET payload 路径：

```text
PetPSServer::HandleGet
  -> CachePS::GetParameterFlat
  -> KVEngineComposite::BatchGetFlat
  -> index_->BatchGet
  -> value_store_->ReadFlatFixedRows / row copy
```

对应代码：

- `src/ps/rdma/petps_server.cc`：`PetPSServer::HandleGet`
- `src/ps/base/cache_ps_impl.h`：`CachePS::GetParameterFlat`
- `src/storage/kv_engine/engine_composite.h`：`KVEngineComposite::BatchGetFlat`
- `src/storage/index/dram/extendible_hash.cpp`：`ExtendibleHash::BatchGet` / `ExtendibleHash::Extract`
- `src/storage/index/dram/pet_hash_index.h`：`DramPetHashIndex::BatchGet`
- `src/storage/value_store/dram_value_store.h`：`DramValueStore::ReadFlatFixedRows`

一次 GET batch 是 `500 * 512B = 256KB`。当前实现需要 server CPU 先把每行 value 聚合拷贝到连续 response staging buffer，然后 transport 再 RDMA write 到 client response slot：

```text
value store rows
  -> server CPU memcpy 到 response staging buffer
  -> RDMA write 到 client response slot
  -> write status
```

因此，当前“拖后腿”的不是 BasePS 虚调用，也不是 RDMA completion 回写，而是 `BatchGetFlat` 里的 index lookup 和 value row copy。

## 当前最好结果

当前较优参数：

```text
transport=rdma
client_processes_per_ip=8
server_shards=1
record_count=200000
value_size=512
batch_keys=500
client_threads_per_process=1
prefetch_depth=16
rdma_rc_qps_per_client_per_shard=16
rdma_rc_slots_per_qp=1
server_rdma_threads=16
rdma_rc_server_get_workers=8
rdma_rc_server_coroutines_per_thread=1
```

结果：

| 场景 | 聚合吞吐 | 结果目录 |
|---|---:|---|
| 单次 profile run | 14.571230M keys/s | `results/benchmark_ps_profile_0530_n8_p8_t16` |
| C1 repeat=3 平均 | 14.452453M keys/s | `results/benchmark_ps_profile_0530_n8_p8_t16_c1_repeat3` |

对应 profile：

| 指标 | 数值 |
|---|---:|
| `handle_get_avg_ns` | 约 163-168us |
| `get_batch_get_avg_ns` | 约 162-167us |
| `get_index_lookup_avg_ns` | 约 76us |
| `get_row_copy_avg_ns` | 约 83-84us |
| `complete_response_avg_ns` | 约 4.6us |
| client `copy_response_avg_ns` | 0 |

`complete_response` 很小，client 侧中间 copy 也已经消掉。大头在 `get_index_lookup` 和 `get_row_copy`。

## 为什么不是继续加 GET 线程

GET worker 拆分是有效的。早期 `p4/N4` 能把吞吐从约 `5.07M` 提到约 `10.06M keys/s`。但在当前最优附近，继续加 GET worker 会退化。

同样参数下只改 GET worker 数：

| index | get workers | 聚合吞吐 |
|---|---:|---:|
| `DRAM_EXTENDIBLE_HASH` | N8 | 14.852399M |
| `DRAM_EXTENDIBLE_HASH` | N12 | 9.146058M |
| `DRAM_EXTENDIBLE_HASH` | N16 | 7.252709M |
| `DRAM_PET_HASH` | N8 | 15.113670M |
| `DRAM_PET_HASH` | N12 | 10.190840M |
| `DRAM_PET_HASH` | N16 | 7.890341M |

N16 的 profile 显示退化主要来自 index lookup：

| 场景 | 聚合吞吐 | `handle_get_avg_ns` | `get_index_lookup_avg_ns` | `get_row_copy_avg_ns` | `complete_response_avg_ns` |
|---|---:|---:|---:|---:|---:|
| N8/T16 | 14.572350M | 约 163us | 约 76us | 约 84us | 约 4.6us |
| N16/T16 | 7.340866M | 约 487-510us | 约 371-395us | 约 107-111us | 约 4.8us |

这说明线程数不是唯一限制。更多 worker 同时访问同一份 index/value store 后，cache miss、TLB、内存带宽和 hash/index 内部共享结构竞争会放大，单个 GET 请求反而变慢。

## 已验证但收益有限的方向

### `DRAM_PET_HASH`

只把 index 从默认 `DRAM_EXTENDIBLE_HASH` 换成 `DRAM_PET_HASH`：

| index | 聚合吞吐 |
|---|---:|
| `DRAM_EXTENDIBLE_HASH` | 14.852399M |
| `DRAM_PET_HASH` | 15.113670M |

收益约 `1.8%`，不是突破口。带 profile 时，PET_HASH 的 `index_lookup_avg_ns` 没有下降，吞吐小幅提高更像是访问形态和 row copy 分布变化。

### `ExtendibleHash::BatchGet` 批内 prefetch

在 `ExtendibleHash::BatchGet` 里对未来 key 的 directory entry 和 block 做 prefetch：

| 场景 | 聚合吞吐 |
|---|---:|
| baseline `DRAM_EXTENDIBLE_HASH` N8 | 14.852399M |
| prefetch 初版 | 14.900620M |
| prefetch + 当前 key hash 复用 | 14.985290M |

收益约 `0.9%`。这个改动可以保留为小优化，但不能解决主瓶颈。

### coroutine scanner

`C1` 和 `C4` repeat=3 对比：

| coroutines per poller | 三轮平均吞吐 |
|---:|---:|
| C1 | 14.452453M |
| C4 | 12.172440M |

`C4` 提高了扫描命中比例，但吞吐下降，说明 coroutine 调度和 poll loop cadence 的代价超过收益。当前性能默认仍应是 `C1`。

### 更多 client / 更深 in-flight

在当前较优参数附近：

| 实验 | 参数变化 | 聚合吞吐 |
|---|---|---:|
| 增加 client 进程 | `p8 -> p12`, `N8/T8/depth16/slots1` | 13.700087M |
| 增加单 client in-flight | `prefetch_depth=16 -> 32`, `slots_per_qp=1 -> 2` | 8.865387M |
| 增加 GET workers | `N8 -> N12`, `p8/T16/depth16/slots1` | 9.509550M |

这几项都没有提升。更多并发主要增加扫描、调度和共享资源竞争。

## 2 shard 结果不能当作 server scale 结论

把 `--server-shard-ips` 从一个 `127.0.0.1` 改成两个 `127.0.0.1,127.0.0.1` 后：

| 场景 | 聚合吞吐 |
|---|---:|
| 单 shard N8 | 14.985290M |
| 本机 2 shard fanout | 1.323520M |

这个结果不能解释为“两台 shard server 处理能力差”。当前 generic PS multi-shard fetch 语义是：

```text
一个 500-key 逻辑 GET
  -> client 按 key hash 拆成 shard0 子 batch + shard1 子 batch
  -> 分别发多个 shard RPC
  -> 等所有 shard RPC 完成
  -> 按原始 key 顺序 merge 回一个逻辑结果
```

它测到的是 distributed client 的 fanout、等待和 merge 成本，不是干净的 server-side shard scale。要验证 shard scale-out，需要让不同 client 进程显式绑定不同 shard，避免每个 batch 被拆成多个 shard 子请求。

## 和 RDMA RC 专项 benchmark 的差异

Generic PS benchmark 路径：

```text
run_benchmark_ps.py
  -> ps_transport_benchmark
  -> RDMAPSClientAdapter / BasePSClient prefetch API
  -> PetPSClient
  -> petps_server
  -> CachePS / KVEngine
```

RDMA RC 专项 benchmark 路径更薄：

```text
run_rdma_rc_transport_benchmark.py
  -> rdma_rc_transport_benchmark
  -> PetPSClient / AllShardsParameterClientWrapper
  -> RC-write slot transport
  -> petps_server
```

历史 RC 专项能到 `30-40M keys/s`，但它不包含 generic PS adapter 的 prefetch/result 语义、transactions workload 和完整 PS wrapper 生命周期。现在 generic PS 的轻 payload `status_only` 能到约 `28.7M keys/s`，说明 transport/prefetch 固定开销已经不是最早的几 M 上限；真实 512B GET 仍被 server payload path 限制。

## 后续优化方向

当前还能继续做的小优化包括 index prefetch、CPU affinity、NUMA 绑定和更细 profile，但这些都不像是数量级突破。

如果目标是接近 RC 专项 `30-40M keys/s`，需要减少或绕过 server payload copy。推荐方向是重新设计 GET response 协议：

```text
index lookup 得到 value handles
  -> value store 暴露 row refs / DirectPtr / MR lkey
  -> server 用 RDMA SG list 直接写 client response slot
  -> 最后写 StatusWord
```

这个方向的关键前提：

1. value store 的 row 内存必须可用于 RDMA SGE，通常需要 value slab/pool 整体注册 MR。
2. transport 需要支持 `WriteSg`，按 HCA `max_sge` 把 500 rows 拆成若干 WR。
3. miss row 需要注册过的 zero row buffer。
4. client 侧可以暂时保持连续 response buffer 语义，避免第一版就改 `GetPrefetchResultFlat` 和上层 PS API。

第一版不建议做 client-side scatter result。server-side SG write 能先消掉 `value row -> response staging buffer` 这次 CPU copy，同时保留 client 连续结果语义，风险更可控。

## 验证记录

已跑过的关键验证：

```bash
cmake --build build --target petps_server ps_transport_benchmark -j

python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 8 \
  --record-count 200000 \
  --value-size 512 \
  --batch-keys 500 \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 4 \
  --repeat 1 \
  --execution-backend local \
  --prefetch-depth 16 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-slots-per-qp 1 \
  --server-rdma-threads 16 \
  --rdma-rc-server-get-workers 8 \
  --rdma-rc-server-coroutines-per-thread 1
```

主要结果目录：

- `results/benchmark_ps_profile_0530_n8_p8_t16`
- `results/benchmark_ps_profile_0530_n8_p8_t16_c1_repeat3`
- `results/benchmark_ps_get_workers_scale_0530_n8_p8_t16_c1_txprofile`
- `results/benchmark_ps_get_workers_scale_0530_n12_p8_t16_c1_txprofile`
- `results/benchmark_ps_get_workers_scale_0530_n16_p8_t16_c1_txprofile`
- `results/benchmark_ps_index_pet_hash_0530_n8_p8_t16_c1_txprofile`
- `results/benchmark_ps_extendible_prefetch_reuse_hash_0530_n8_p8_t16_c1_txprofile`
- `results/benchmark_ps_extendible_prefetch_reuse_hash_0530_s2_p8_t16_n8_c1_txprofile`

## 2026-05-30 收口更新：direct-SG 与绑核固化

本轮完成后，保留的有效优化是：

- GET 默认请求 `direct-SG + fallback`：client 默认在 GET descriptor 上标记 direct-SG 能力，server 优先用 value store row refs 直接 RDMA SG write 到 client response slot；如果 row refs / MR 条件不满足，自动回退到原 staging-copy response path。
- value backing MR 注册：server 启动时把 DRAM value store 背后的 slab/pool region 注册为 RDMA local MR，供 direct-SG 使用。
- `RawVerbsTransport::WriteSg`：支持多 SGE RDMA write，当前每个 WR 最多 32 个 SGE。
- 显式 NUMA / bind-core 控制：runner 支持 server/client 分别设置 NUMA id 和 bind-core offset/stride，避免本机压测时 server poller 与 client 落到同一物理核或 sibling 上。

本轮清理掉的无效实验路径：

- `rdma_rc_get_inner_parallelism`：i2 profile 中虽然能降低一部分 lookup 子阶段耗时，但总 `handle_get_avg_ns` 大幅升高，吞吐从约 `18.4M` 降到约 `9.2M keys/s`，因此删除该开关和 worker 队列路径。
- direct-SG 外部开关：`--rdma-rc-get-direct-sg`、fallback 开关和 chunk rows 开关不再作为 benchmark 参数暴露。direct-SG 是默认尝试路径，fallback 是内部安全兼容路径。

### 绑核结论

这台机器拓扑为 2 socket、每 socket 28 个物理核、2 threads/core。socket0 的第一硬线程是 CPU `0,2,4,...54`，对应 sibling 是 `56,58,...110`。

早期只按逻辑 offset 隔离是不够的：`p8/t16` 会让 client 进入 server poller 的 sibling 区间，仍可能污染结果。更稳妥的本机压测策略有两种：

- `socket-split`：server 在 socket0，client 在 socket1。CPU 干扰隔离干净，但牺牲 RNIC/PCIe locality。
- `socket0-disjoint`：server/client 都在 socket0，但显式使用不重叠物理核。例如 server 用 core index `0..15`，6 个 client 用 core index `16..27`。

`socket-split` transport 从线速附近降到约 `14.98GB/s`，主要不是协议问题，而是 client 放到远端 NUMA 后引入了 RNIC/PCIe/DMA/control-path 跨 socket 成本。

### 当前有效 benchmark 结果

| 层级 | 配置 | 结果 | 512B 等价带宽 | 结果路径 |
|---|---|---:|---:|---|
| RDMA RC transport | socket-split, `p8/t16/depth16` | `29.26M keys/s` | `14.98GB/s` | `results/rdma_bindcore_0530/transport_p8_t16_d16_socket_split.log` |
| RDMA RC transport | socket0-disjoint, `p6/t16/depth16` | `48.69M keys/s` | `24.93GB/s` | `results/rdma_bindcore_0530/transport_p6_t16_d16_socket0_disjoint_short.log` |
| generic PS RDMA | socket-split, `p8/t16/i1/direct-SG` | `19.47M keys/s` | `9.97GB/s` | `results/rdma_bindcore_0530/benchmark_ps_p8_t16_i1_directsg_socket_split` |
| generic PS RDMA | socket0-disjoint, `p6/t16/i1/direct-SG` | `18.87M keys/s` | `9.66GB/s` | `results/rdma_bindcore_0530/benchmark_ps_p6_t16_i1_directsg_socket0_disjoint` |

解释：

- transport 在 `socket0-disjoint` 下回到约 `25GB/s`，说明 RDMA transport 本身已经接近本机 25GB/s 极限。
- generic PS 没有随 transport 上限同步上涨，当前仍在约 `19M keys/s` 左右。
- 因此当前 PS 层瓶颈不是 RDMA 带宽，而是 server GET 处理路径，尤其 direct-SG 前的 index lookup / batch get。

### 当前 profile 判断

`benchmark_ps_p6_t16_i1_directsg_socket0_disjoint` 稳态 profile：

| 指标 | 典型值 |
|---|---:|
| `handle_get_avg_ns` | 约 `372-373us` |
| `get_batch_get_avg_ns` | 约 `315us` |
| `get_index_lookup_avg_ns` | 约 `268-269us` |
| `get_row_copy_avg_ns` | `0` |
| `get_direct_sg_avg_ns` | 约 `372us` |
| `complete_response_avg_ns` | 约 `2.6us` |

direct-SG 已经把原来的 row-copy staging buffer 成本消掉，但 `index_lookup` 仍然占主要时间。后续若继续优化，应优先看 KV/index lookup 的并发访问形态、cache/TLB 行为和 batch lookup 实现，而不是继续增加 RDMA QP、client 数、GET worker 或 coroutine scanner。

### 固化后的推荐命令

本机 transport 上限建议用同 socket 物理核错开的配置：

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

generic PS RDMA 建议同样使用显式 NUMA 和 core offset：

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --client-ips 127.0.0.1 \
  --server-shard-ips 127.0.0.1 \
  --client-processes-per-ip 6 \
  --record-count 1000000 \
  --value-size 512 \
  --batch-keys 500 \
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
  --rdma-rc-profile-interval-ms 1000 \
  --rdma-rc-server-numa-id 0 \
  --rdma-rc-client-numa-id 0 \
  --rdma-server-bind-core-offset 0 \
  --rdma-client-bind-core-offset 16 \
  --rdma-client-bind-core-stride 2 \
  --client-timeout 180 \
  --cluster-timeout 60 \
  --show-runner-logs \
  --output-dir results/rdma_bindcore_0530/benchmark_ps_p6_t16_i1_directsg_socket0_disjoint
```
