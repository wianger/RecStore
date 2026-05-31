# PS RDMA GET Benchmark 完整报告（2026-05-31）

本文整理 RecStore PS/network 层 RDMA GET 路径从 `~2M keys/s` 到
`~44.87M keys/s` 的阶段性结果。本文只讨论 PS/network 和 storage-only
benchmark，不讨论 PyTorch/model 端到端性能；所有数字均来自本地单机
`127.0.0.1` 验证，不能直接解释为跨机器扩展能力。

## 1. 结论

当前最重要的结论不是“继续堆并发”，而是：

1. `DRAM_EXTENDIBLE_HASH` 下，PS RDMA 的 `~19M keys/s` 已经基本碰到本地
   `BatchGetFlat(500 random keys)` 的存储层上限。
2. `DRAM_PET_HASH` 的本地随机 batch lookup 能到 `~51.96M keys/s`，说明存储层
   还有空间。
3. `direct-SG` 对 `DRAM_PET_HASH` 的离散 row refs 不友好，真实 PS RDMA GET
   只有 `~14.93M keys/s`。
4. `DRAM_PET_HASH + staging-copy` 能跑到 `~44.87M keys/s`，已经接近当前
   RDMA transport/device 观测上限 `~48.7M keys/s`。
5. 因此现阶段应固化为按 KVEngine/layout 选择 response path：
   `DRAM_PET_HASH -> staging-copy`，其他 index 默认保留 `direct-SG`。

一句话概括：

```text
EH 的瓶颈在 index lookup；
PET 的 index 很快，但 direct-SG 不适配它的离散 row layout；
PET + staging-copy 是当前最优的稳定主线。
```

## 2. 优化时间线

| 阶段 | 关键变化 | 代表 commit / 动作 | 吞吐 |
|---|---|---|---:|
| 初始 RDMA RC GET | CPU 根据 index 做 batch get，再拷贝到连续 response buffer，由 RDMA write 返回 | `feat(rdma): add verbs RC write transport`、`checkpoint current transport changes` | `~2M keys/s` |
| pipeline / prefetch | benchmark fetch 路径增加 pipeline 和 prefetch 诊断 | `bench(ps): add rdma prefetch diagnostics`、`perf(ps): pipeline rdma fetch benchmark` | `~3M keys/s` |
| 减少中间复制 | 优化 benchmark fetch path，减少 vector 和结果搬运开销 | `perf(ps): optimize rdma fetch path`、`perf(rdma): optimize PS benchmark fetch path` | `~5M keys/s` |
| GET payload worker | 将 poll thread 中的 index lookup / batch get / payload 填充拆到专门 worker | `perf(rdma): offload PS GET payload handling` | `~9-10M keys/s` |
| GET worker / poller 扩容 | 继续把 GET worker 数按更大的 4 的倍数扫描，并增加 server RDMA poll 线程 | `feat(rdma): tune ps scheduling experiments`、`feat(rdma): split get profile timing` | `~14-15M keys/s` |
| direct-SG / sglist 协议 | server 根据 row refs 组装 SG list，按 HCA `max_sge` 拆分 WR，避免 CPU row copy | `feat(rdma): enable direct-sg ps get path` | 较当时基线约 `+0.5M keys/s`，收益不明显 |
| 绑核固化 | 固定 server/client RDMA 相关线程的 CPU affinity，减少调度漂移 | 绑核参数与 runner 固化 | EH 稳定到 `~19M keys/s` |
| EH 小优化 | `ExtendibleHash::BatchGet` prefetch、array view、小对象复制减少 | `perf(rdma): prefetch extendible hash batch get` | 约 `+0.1M keys/s` |
| 存储层对齐实验 | 新增 `benchmark_kv_engine read_mode=batch_get_flat`，用 500 random keys 对齐 PS GET 形态 | 本轮临时诊断后固化 | EH `~19.45M`，PET `~51.96M` |
| PET response path 绑定 | `run_benchmark_ps.py --rdma-get-response-mode=auto`：PET 自动走 staging-copy | 本轮固化 | PET `~44.87M keys/s` |

## 3. 关键结果矩阵

### 3.1 Storage-only 对齐实验

本实验使用本地 KVEngine 随机 `500` batch keys，直接调用
`BaseKV::BatchGetFlat`，用于判断 PS RDMA 是否已经被 storage-only 上限卡住。

参数：

```text
record_count=300000
value_size=512
batch_keys=500
threads=16
workload=c
distribution=uniform
dram_allocator=PERSIST_LOOP_SLAB
read_mode=batch_get_flat
```

| 层级 | index | 路径 | 吞吐 |
|---|---|---|---:|
| storage-only | `DRAM_EXTENDIBLE_HASH` | `BatchGetFlat(500 random keys)` | `19.45M keys/s` |
| storage-only | `DRAM_PET_HASH` | `BatchGetFlat(500 random keys)` | `51.96M keys/s` |

结果路径：

- `results/kv_batchget_probe_0531/eh_batch_get_flat.log`
- `results/kv_batchget_probe_0531/pet_batch_get_flat.log`

### 3.2 PS RDMA GET 对齐实验

主要参数：

```text
transport=rdma
client_processes_per_ip=6
server_shards=1
record_count=300000
value_size=512
batch_keys=500
client_threads_per_process=1
prefetch_depth=16
rdma_rc_qps_per_client_per_shard=16
rdma_rc_slots_per_qp=1
server_rdma_threads=16
rdma_rc_server_get_workers=0
rdma_rc_server_coroutines_per_thread=1
```

| 层级 | index | response path | 聚合吞吐 |
|---|---|---|---:|
| PS/network | `DRAM_EXTENDIBLE_HASH` | direct-SG | `19.37M keys/s` |
| PS/network | `DRAM_PET_HASH` | index_only | `89.13M keys/s` |
| PS/network | `DRAM_PET_HASH` | direct-SG | `14.93M keys/s` |
| PS/network | `DRAM_PET_HASH` | staging-copy | `44.87M keys/s` |
| transport/device 观测上限 | 不涉及 | RDMA transport cap | `48.7M keys/s` |

结果路径：

- `results/benchmark_ps_bottleneck_probe_0531/real_directsg`
- `results/benchmark_ps_bottleneck_probe_0531/index_only_pet_hash`
- `results/benchmark_ps_bottleneck_probe_0531/real_directsg_pet_hash`
- `results/benchmark_ps_bottleneck_probe_0531/real_staging_pet_hash`

## 4. 对旧结论的修正

2026-05-30 的阶段性判断是：GET 默认请求 `direct-SG + fallback`，希望通过
server-side SG write 消掉 `value row -> response staging buffer` 这次 CPU copy。

2026-05-31 的 PET 实验修正了这个判断：

- `direct-SG` 对 `DRAM_EXTENDIBLE_HASH` 是合理路径，但 EH 的主要瓶颈已经是
  `index_->BatchGet`，所以 direct-SG 很难继续拉开吞吐。
- `direct-SG` 对 `DRAM_PET_HASH` 并不合理。PET lookup 快，但随机 key 下 row refs
  更离散，server 组装 SGE/WR 和多次 payload SG write 的成本超过了省掉 CPU copy
  的收益。
- `staging-copy` 虽然多一次 CPU row copy，但能把 500 行 value 聚合成连续 response
  buffer，再用普通 RDMA payload write 返回；这个路径和 PET 当前 layout 更匹配。

因此当前固化策略不是“永远 direct-SG”，而是“按 KVEngine 与 value layout 选择
response path”。

## 5. 并发度与 server-side 线程实验结论

PET direct-SG 下增加 client 进程数和 GET worker 数没有改善：

| case | clients | GET workers | 聚合吞吐 |
|---|---:|---:|---:|
| `p6_g0` | 6 | 0 | `14.93M keys/s` |
| `p8_g0` | 8 | 0 | `14.95M keys/s` |
| `p6_g4` | 6 | 4 | `9.82M keys/s` |
| `p8_g4` | 8 | 4 | `9.83M keys/s` |

这说明 PET direct-SG 的主要问题不是 client 并发不足。GET worker offload 在这组
参数下还会引入队列锁、调度和 completion 回流成本，导致吞吐下降。

PET staging-copy 固化后，又进一步扫描了更高 client 负载、降低 client 数并增加
server RDMA poll 线程、以及降低 client 数并启用 GET worker 的配置：

```text
index=DRAM_PET_HASH
response_mode=auto -> staging_copy
record_count=1000000
value_size=512
batch_keys=500
prefetch_depth=16
rdma_rc_qps_per_client_per_shard=16
rdma_rc_slots_per_qp=1
rdma_rc_server_coroutines_per_thread=1
```

| case | clients | server poll threads | GET workers | repeat 结果 | 平均吞吐 |
|---|---:|---:|---:|---|---:|
| baseline | 6 | 16 | 0 | `38.55 / 44.14 / 38.67M` | `40.45M keys/s` |
| 增加 client 负载 | 8 | 16 | 0 | `32.57 / 30.46 / 30.46M` | `31.16M keys/s` |
| 降 client、加 poll | 4 | 24 | 0 | `27.20 / 24.66 / 24.42M` | `25.42M keys/s` |
| 降 client、加 GET worker | 4 | 16 | 8 | `19.17 / 19.22 / 17.40M` | `18.60M keys/s` |

结果路径：

- `results/benchmark_ps_rdma_pet_auto_capacity_0531`
- `results/benchmark_ps_rdma_pet_auto_capacity_c8_0531`
- `results/benchmark_ps_rdma_pet_auto_c4_poll24_0531`
- `results/benchmark_ps_rdma_pet_auto_c4_poll16_get8_0531`

这些实验都正常完成，`run_config.json` 中 `rdma_get_response_mode` 均解析为
`staging_copy`。结果说明：当前最优区间仍是 `clients=6, server poll=16,
GET workers=0`。单纯增加 client 负载会让调度和 poll 压力恶化；降低 client 后即使
增加 poll 线程也喂不满链路；把 PET staging-copy 的 GET payload 工作拆到额外
GET worker 则会引入队列、同步和 completion 回流成本，吞吐明显下降。

因此下一阶段不应继续盲目堆 client、poll 或 GET worker 数，而应聚焦到更细粒度的
性能与稳定性优化：降低 repeat 间抖动、缩短 server wait/status 路径、减少
staging-copy 的内存访问成本，并用 profile 数据定位 poller、payload copy、completion
回收和 client wait 的实际占比。

## 6. 当前固化的接口

### 6.1 PS RDMA response mode

`run_benchmark_ps.py` 新增：

```text
--rdma-get-response-mode auto|direct_sg|staging_copy
```

策略：

| mode | 行为 |
|---|---|
| `auto` | 默认值；`DRAM_PET_HASH` 自动选择 `staging_copy`，其他 index 自动选择 `direct_sg` |
| `direct_sg` | 强制 GET descriptor 请求 direct-SG |
| `staging_copy` | 强制 GET 不设置 direct-SG flag，走连续 response buffer staging-copy |

推荐复现 PET 当前最优路径：

```bash
python3 src/test/scripts/run_benchmark_ps.py \
  --transports rdma \
  --index-type DRAM_PET_HASH \
  --rdma-get-response-mode auto \
  --batch-keys 500 \
  --value-size 512
```

### 6.2 KVEngine batch benchmark

`benchmark_kv_engine` 保留：

```text
--read_mode=batch_get_flat
--batch_keys=<N>
```

`tools/benchmarks/run_ycsb_compare.py` 对应支持：

```text
--read-mode batch_get_flat
--batch-keys <N>
```

`summary.csv` 会记录 `batch_keys`，用于把 storage-only 和 PS/network 的 batch
形态对齐。

## 7. 代码路径

PS RDMA GET 主路径：

```text
ps_transport_benchmark
  -> RDMAPSClientAdapter / PetPSClient
  -> petps_server poller
  -> PetPSServer::HandleGet
  -> CachePS::GetParameterDirectFixedRows / GetParameterFlat
  -> KVEngineComposite::BatchGetFlat / DirectFixedRows
  -> index_->BatchGet
  -> direct-SG 或 staging-copy response
```

相关文件：

- `src/benchmark/ps_transport_benchmark.cc`
- `src/ps/rdma/petps_client.cc`
- `src/ps/rdma/petps_server.cc`
- `src/ps/base/cache_ps_impl.h`
- `src/storage/kv_engine/engine_composite.h`
- `src/storage/kv_engine/benchmark_kv_engine.cc`
- `src/storage/index/dram/extendible_hash.cpp`
- `src/storage/index/dram/pet_hash_index.h`
- `src/storage/value_store/dram_value_store.h`
- `src/test/scripts/run_benchmark_ps.py`
- `tools/benchmarks/run_ycsb_compare.py`

## 8. 后续方向

短期建议：

1. 把 `DRAM_PET_HASH + staging-copy + auto mode` 作为 PS RDMA GET 默认性能 case。
2. 保留 `batch_get_flat` storage-only benchmark，用于每次改 index/value layout 后先测
   存储层上限。
3. 保留 `direct_sg` 强制模式，作为 EH 与 SG path 回归测试。
4. 为 direct-SG 增加 WR/SGE per request profile，避免只看总吞吐时误判。
5. 把 `clients=6, server poll=16, GET workers=0` 作为当前推荐容量参数；新增实验
   参数必须和该基线对比，并记录 repeat 间稳定性。

中期再考虑：

1. 针对 PET staging-copy 做细节优化，重点提升性能和稳定性，而不是继续扩大线程数：
   - 降低 repeat 间吞吐抖动；
   - 优化 payload staging buffer 的填充和内存访问；
   - 进一步拆分并统计 poller、server lookup/copy、RDMA completion、client wait 的耗时；
   - 检查绑核、NUMA、本地控制面和 benchmark 启停过程对稳定性的影响。
2. 优化 PET value row 连续性或 SG write batching。
3. 根据 direct-SG 生成的 WR 数动态 fallback 到 staging-copy。
4. 清理已经被 `auto` 覆盖的临时 benchmark 参数，减少 runner 参数面。
5. 重新设计 shard scale-out benchmark，避免把 distributed fanout/merge 成本误认为
   server-side shard 扩展能力。

## 9. 已验证项

本轮固化前已验证：

```bash
python3 -m unittest src/test/scripts/test_run_benchmark_ps.py

cmake --build build --target ps_transport_benchmark benchmark_kv_engine -j
```

还跑过两个 smoke：

- `benchmark_kv_engine --read_mode=batch_get_flat --batch_keys=64`
- `run_benchmark_ps.py --index-type DRAM_PET_HASH --rdma-get-response-mode auto`

其中 PET auto smoke 的 `run_config.json` 显示 `rdma_get_response_mode=staging_copy`，
`summary.csv` 正常产出成功行。
