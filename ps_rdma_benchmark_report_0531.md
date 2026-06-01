# PS RDMA GET Benchmark 完整报告（2026-05-31，2026-06-01 更新）

本文整理 RecStore PS/network 层 RDMA GET 路径从 `~2M keys/s` 到
稳定 `~40-42M keys/s`、单轮最高 `~45-46M keys/s` 的阶段性结果。本文只讨论
PS/network 和 storage-only benchmark，不讨论 PyTorch/model 端到端性能；所有数字均
来自本地单机 `127.0.0.1` 验证，不能直接解释为跨机器扩展能力。

## 1. 范围与结论

本轮工作的主线是把 RDMA GET 接入 PS benchmark，并持续优化 batch embedding fetch
路径。当前结论可以分成三层：

1. `DRAM_EXTENDIBLE_HASH` 下，PS RDMA 的 `~19M keys/s` 已经基本碰到本地
   `BatchGetFlat(500 random keys)` 的 storage-only 上限。
2. `DRAM_PET_HASH` 的本地随机 batch lookup 能到 `~51.96M keys/s`，说明 storage
   层还有空间，但它和 direct-SG 的离散 row layout 适配不好。
3. `DRAM_PET_HASH + staging-copy` 在当前稳定默认参数下 repeat 平均约
   `41.64M keys/s`，单轮可到 `45.62M keys/s`；历史 `qps=20` tuning profile
   最高到 `46.69M keys/s`，但不够稳定，不能作为默认。

当前固化策略：

```text
DRAM_PET_HASH -> staging-copy
其他 index     -> direct-SG
```

一句话概括：

```text
EH 的瓶颈在 index lookup；
PET 的 index 很快，但 direct-SG 不适配它的离散 row layout；
PET + staging-copy 是当前最优主线。
```

## 2. 当前最佳配置

推荐把下面配置作为当前 PS RDMA GET capacity baseline：

```text
transport=rdma
index=DRAM_PET_HASH
rdma_get_response_mode=auto -> staging_copy
record_count=1000000
value_size=512
batch_keys=500
client_processes_per_ip=6
client_threads_per_process=1
client_load_threads_per_process=1
server_shards=1
server_rdma_threads=16
rdma_rc_server_get_workers=0
rdma_rc_server_coroutines_per_thread=1
prefetch_depth=16
rdma_rc_qps_per_client_per_shard=16
rdma_rc_slots_per_qp=1
server/client NUMA=0
server bind offset=0
client bind offset=16
client bind stride=2
```

复测结果：

| case | repeat 结果 | 平均吞吐 |
|---|---:|---:|
| `DRAM_PET_HASH + auto/staging-copy` | `38.55 / 44.14 / 38.67M` | `40.45M keys/s` |
| `DRAM_PET_HASH + auto/staging-copy`，2026-06-01 固化复测 | `41.14 / 40.66 / 42.26 / 38.52 / 45.62M` | `41.64M keys/s` |

`qps=16` 是当前稳定默认。历史 PET staging-copy 单轮结果可到 `~44.87M keys/s`，
2026-06-01 复测最高 `45.62M keys/s`；`qps=20` 曾冲到 `46.69M keys/s`，但 repeat
更常落在 `41-43M keys/s`，因此只作为 tuning profile。

结果路径：

- `results/benchmark_ps_rdma_pet_auto_capacity_0531`
- `results/rdma_ps_solidify_qps16_repeat5_0601004853`

## 3. 优化时间线

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
| storage-only 对齐 | 新增 `benchmark_kv_engine read_mode=batch_get_flat`，用 500 random keys 对齐 PS GET 形态 | 本轮诊断后固化 | EH `~19.45M`，PET `~51.96M` |
| PET response path 绑定 | `run_benchmark_ps.py --rdma-get-response-mode=auto`：PET 自动走 staging-copy | 本轮固化 | PET 稳定 `~40-42M keys/s`，单轮最高 `45.62M keys/s` |

这条时间线里最关键的修正是：sglist/direct-SG 不是从 `~14M` 到 `~19M` 的主要来源；
`~19M` 主要来自绑核。sglist 只带来约 `+0.5M keys/s`，EH 小优化也只有约
`+0.1M keys/s`。

## 4. 瓶颈诊断证据

### 4.1 Storage-only 对齐实验

本实验使用本地 KVEngine 随机 `500` batch keys，直接调用 `BaseKV::BatchGetFlat`，
用于判断 PS RDMA 是否已经被 storage-only 上限卡住。

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

解释：

- EH 的 storage-only 上限已经接近 PS RDMA EH 结果，因此继续优化 RDMA 协议本身收益有限。
- PET 的 storage-only 上限明显更高，因此 PS RDMA 仍有利用 PET 的空间。

### 4.2 PS RDMA GET 对齐实验

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
| PS/network | `DRAM_PET_HASH` | staging-copy, stable `qps=16` | `41.64M keys/s` avg，`45.62M keys/s` best |
| PS/network | `DRAM_PET_HASH` | staging-copy, tuning `qps=20` | up to `46.69M keys/s` |
| transport/device 观测上限 | 不涉及 | RDMA transport cap | `48.7M keys/s` |

结果路径：

- `results/benchmark_ps_bottleneck_probe_0531/real_directsg`
- `results/benchmark_ps_bottleneck_probe_0531/index_only_pet_hash`
- `results/benchmark_ps_bottleneck_probe_0531/real_directsg_pet_hash`
- `results/benchmark_ps_bottleneck_probe_0531/real_staging_pet_hash`

解释：

- `index_only=89.13M keys/s` 说明 PET index 本身不是主要瓶颈。
- `PET + direct-SG=14.93M keys/s` 说明 direct-SG 对 PET 的离散 row refs 不友好。
- `PET + staging-copy` 的稳定复测均值为 `41.64M keys/s`，单轮可到 `45.62M keys/s`，
  说明连续 response buffer 更适合当前 PET layout。

## 5. direct-SG 结论修正

2026-05-30 的阶段性判断是：GET 默认请求 `direct-SG + fallback`，希望通过
server-side SG write 消掉 `value row -> response staging buffer` 这次 CPU copy。

2026-05-31 的 PET 实验修正了这个判断：

- `direct-SG` 对 `DRAM_EXTENDIBLE_HASH` 是合理路径，但 EH 主要瓶颈已经是
  `index_->BatchGet`。
- `direct-SG` 对 `DRAM_PET_HASH` 并不合理。PET lookup 快，但随机 key 下 row refs
  更离散，server 组装 SGE/WR 和多次 payload SG write 的成本超过了省掉 CPU copy 的收益。
- `staging-copy` 虽然多一次 CPU row copy，但能把 500 行 value 聚合成连续 response
  buffer，再用普通 RDMA payload write 返回；这个路径和 PET 当前 layout 更匹配。

因此当前策略不是“永远 direct-SG”，而是“按 KVEngine 与 value layout 选择 response path”。

## 6. 并发度与线程扫描

### 6.1 PET direct-SG 扫描

PET direct-SG 下增加 client 进程数和 GET worker 数没有改善：

| case | clients | GET workers | 聚合吞吐 |
|---|---:|---:|---:|
| `p6_g0` | 6 | 0 | `14.93M keys/s` |
| `p8_g0` | 8 | 0 | `14.95M keys/s` |
| `p6_g4` | 6 | 4 | `9.82M keys/s` |
| `p8_g4` | 8 | 4 | `9.83M keys/s` |

这说明 PET direct-SG 的主要问题不是 client 并发不足。GET worker offload 在这组参数下
还会引入队列锁、调度和 completion 回流成本，导致吞吐下降。

### 6.2 PET staging-copy 扫描

PET staging-copy 固化后，又扫描了更高 client 负载、降低 client 数并增加 server RDMA
poll 线程、以及降低 client 数并启用 GET worker 的配置：

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
增加 poll 线程也喂不满链路；把 PET staging-copy 的 GET payload 工作拆到额外 GET
worker 会引入队列、同步和 completion 回流成本，吞吐明显下降。

### 6.3 2026-06-01 固化复测

固定当前默认参数后，又跑了一次 repeat=5：

```text
index=DRAM_PET_HASH
response_mode=auto -> staging_copy
record_count=1000000
value_size=512
batch_keys=500
client_processes_per_ip=6
server_rdma_threads=16
rdma_rc_server_get_workers=0
prefetch_depth=16
rdma_rc_qps_per_client_per_shard=16
rdma_rc_slots_per_qp=1
same-socket NUMA/core binding
```

| repeat | aggregate M keys/s | per-client min/max M keys/s |
|---:|---:|---:|
| 0 | 41.135 | 6.851 / 6.860 |
| 1 | 40.657 | 6.767 / 6.782 |
| 2 | 42.260 | 7.041 / 7.047 |
| 3 | 38.518 | 6.414 / 6.426 |
| 4 | 45.619 | 7.599 / 7.607 |

平均 `41.638M keys/s`，median `41.135M keys/s`。每轮内部 client 分布很均匀，
说明当前抖动主要是 repeat 级别的环境/调度波动，而不是 client 间负载倾斜。

结果路径：

- `results/rdma_ps_solidify_qps16_repeat5_0601004853`

## 7. 固化接口

### 7.1 PS RDMA response mode

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
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports rdma \
  --index-type DRAM_PET_HASH \
  --rdma-get-response-mode auto \
  --batch-keys 500 \
  --value-size 512
```

### 7.2 KVEngine batch benchmark

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

`summary.csv` 会记录 `batch_keys`，用于把 storage-only 和 PS/network 的 batch 形态对齐。

## 8. 代码路径

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
- `tools/benchmarks/run_benchmark_ps.py`
- `tools/benchmarks/run_ycsb_compare.py`

## 9. 后续方向

短期建议：

1. 把 `DRAM_PET_HASH + staging-copy + auto mode` 作为 PS RDMA GET 默认性能 case。
2. 保留 `batch_get_flat` storage-only benchmark，用于每次改 index/value layout 后先测
   存储层上限。
3. 保留 `direct_sg` 强制模式，作为 EH 与 SG path 回归测试。
4. 为 direct-SG 增加 WR/SGE per request profile，避免只看总吞吐时误判。
5. 把 `clients=6, server poll=16, GET workers=0, prefetch=16, qps=16` 作为当前
   推荐容量参数；新增实验参数必须和该基线对比，并记录 repeat 间稳定性。

中期再考虑：

1. 针对 PET staging-copy 做细节优化，重点提升性能和稳定性，而不是继续扩大线程数。
2. 细化 profile：拆分 PET lookup、row copy、payload staging、completion/status 写回、
   client wait/revoke，并记录 per-poller min/max。
3. 优化 payload staging buffer 的填充、对齐和固定 row copy；软件 prefetch 已验证为负结果，
   当前保留的是通用 power-of-two shift row addressing。
4. 保留 direct-SG 回归与 WR/SGE profile，后续根据连续 row 比例或 WR 数决定是否动态 fallback。
5. 固化 benchmark 启停、warmup/run 分段和 median/p90 报告，降低 repeat 间解释成本。
6. 清理已经被 `auto` 覆盖的临时 benchmark 参数，减少 runner 参数面。
7. 重新设计 shard scale-out benchmark，避免把 distributed fanout/merge 成本误认为
   server-side shard 扩展能力。

## 10. 已验证项

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

2026-06-01 固化复测前后还验证：

```bash
cmake --build build --target ps_transport_benchmark petps_server -j
python3 -m unittest src/test/scripts/test_run_benchmark_ps.py src/test/scripts/test_petps_cluster_runner.py
ctest --test-dir build -R 'test_rdmaps_client_adapter|test_allshards_ps_client' -VV
```
