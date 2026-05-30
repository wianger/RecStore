# PS Benchmark 当前修复固化与对比结果

日期：2026-05-30

## 目标

本轮目标是把前面已经做过的 PS RDMA benchmark 修复重新编译、回归，并用同一组参数跑当前对比。这里的“固化”按工程动作理解为：

- 确认相关修复已经在代码路径中生效。
- 重新 build，避免使用旧二进制。
- 跑针对性回归测试。
- 串行跑 benchmark，并把 raw logs、summary 和本报告落盘。

本轮没有自动 commit。

## 已固化的关键修复

相关代码路径：

- `src/ps/rdma/rdma_ps_client_adapter.cc`
- `src/ps/rdma/rdma_ps_client_adapter.h`
- `src/ps/rdma/petps_client.cc`
- `src/ps/rdma/petps_client.h`
- `src/ps/rdma/petps_server.cc`
- `src/ps/rdma/rc_options.cc`
- `src/storage/kv_engine/engine_composite.h`
- `src/storage/value_store/dram_value_store.h`
- `src/storage/value_store/value_store.h`
- `src/benchmark/ps_transport_benchmark.cc`
- `src/test/scripts/run_benchmark_ps.py`

主要修复点：

| 修复 | 关键函数 | 作用 |
|---|---|---|
| Adapter prefetch buffer pool | `RDMAPSClientAdapter::AcquirePrefetchBuffer`、`ReleasePrefetchBuffer`、`PrefetchParameter`、`MarkPrefetchConsumed` | 避免每次 prefetch 重新申请并整块清零 `500 * 512B` response buffer，只初始化 status word |
| Borrow RC response payload | `PetPSClient::BorrowGetResultPayload`、`RDMAPSClientAdapter::BorrowPrefetchResult`、`GetPrefetchResultFlat` | 单 shard prefetch 直接从 RC response slot payload materialize，跳过 client 中间 response copy |
| Result copy 诊断开关 | `--rdma_adapter_skip_prefetch_result_copy` / `--rdma-adapter-skip-prefetch-result-copy` | 用于确认最后 vector materialize 是否是主瓶颈 |
| Fake GET `index_only` | `PetPSServer::HandleGet`、`CachePS::GetParameterFlat`、`KVEngineComposite::BatchGetFlat` | 只做 key parse + index lookup，不复制 value payload，用于拆分 index 与 value copy |
| DRAM fixed-row flat read | `ValueStore::ReadFlatFixedRows`、`DramValueStore::ReadFlatFixedRows`、`KVEngineComposite::BatchGetFlat` | 对固定 row size 的 DRAM_VALUE_STORE 走连续 flat copy 快路径 |
| Prefetch depth guard | `ps_transport_benchmark.cc`、`run_benchmark_ps.py` | 检查 `prefetch_depth <= qps_per_client_per_shard * slots_per_qp`，避免 benchmark 配置假并发 |

## 验证

已执行：

```bash
cmake --build build --target test_rdmaps_client_adapter test_ps_transport_benchmark ps_transport_benchmark rdma_rc_transport_benchmark petps_server -j
ctest --test-dir build -R 'test_rdmaps_client_adapter|test_ps_transport_benchmark' -VV
python3 -m unittest src/test/scripts/test_run_benchmark_ps.py
./build/bin/test_kvengine --gtest_filter='*BatchGetFlat*'
```

结果：

- build 成功。
- `test_rdmaps_client_adapter` 通过。
- `test_ps_transport_benchmark` 通过。
- `src/test/scripts/test_run_benchmark_ps.py` 通过，24 tests。
- `*BatchGetFlat*` 通过，54 tests。

## 串行执行说明

本轮 benchmark 命令是串行执行的：每条 `run_benchmark_ps.py` 或 `run_rdma_rc_transport_benchmark.py` 完整退出后才启动下一条。并行执行的只有后续读取 `summary.csv`、`stdout.log`、`server_0.log` 这类只读汇总命令，不会启动 server/client，也不会占用 RDMA QP。

## 统一参数

PS benchmark 核心参数：

- `transports=rdma`
- `client-ips=127.0.0.1`
- `server-shard-ips=127.0.0.1`
- `server_shards=1`
- `client_processes_per_ip=4`
- `client_threads_per_process=1`
- `client_load_threads_per_process=1`
- `record_count=200000`
- `value_size=512`
- `batch_keys=500`
- `runtime_seconds=3`
- `prefetch_depth=16`
- `rdma_rc_qps_per_client_per_shard=16`
- `rdma_rc_slots_per_qp=1`
- `rdma_rc_profile_interval_ms=1000`

结果根目录：

```text
results/benchmark_ps_current_compare_0530/
```

## 对比结果

| 场景 | 层级 | 聚合吞吐 | 结果目录 | 解释 |
|---|---|---:|---|---|
| PS RDMA default | PS/network | `5.108M keys/s` | `results/benchmark_ps_current_compare_0530/ps_default` | 当前真实 payload 路径 |
| PS RDMA `status_only` | PS/network 诊断 | `28.402M keys/s` | `results/benchmark_ps_current_compare_0530/ps_status_only` | 去掉 server payload/index/value copy，只返回 status |
| PS RDMA `index_only` | PS/network 诊断 | `11.676M keys/s` | `results/benchmark_ps_current_compare_0530/ps_index_only` | server parse keys + index lookup，不复制 value payload |
| PS RDMA `skip_prefetch_result_copy` | PS/network 诊断 | `5.283M keys/s` | `results/benchmark_ps_current_compare_0530/ps_skip_prefetch_result_copy` | 跳过 adapter 最后一跳 vector materialize，收益很小 |
| RDMA RC `async_stream`, 4 clients, 1 thread | transport diagnostic | `3.886M keys/s` | `results/benchmark_ps_current_compare_0530/rdma_rc_async_stream_4c_t1` | 低并发专项线，不代表 RC 上限 |
| RDMA RC `async_stream`, 4 clients, 16 threads | transport diagnostic | `48.143M keys/s` | `results/benchmark_ps_current_compare_0530/rdma_rc_async_stream_4c_t16` | 高并发专项容量线，和 PS default 不是同一语义层 |

PS 聚合吞吐按各 client `summary.csv` 的 run phase `key_ops_per_sec` 求和计算。

## 关键 profile 证据

PS default 的 server profile 稳态段：

- `handle_get_avg_ns` 约 `93.8us - 94.7us/batch`
- `get_batch_get_avg_ns` 约 `93.5us - 94.4us/batch`
- `get_row_copy_avg_ns` 约 `50.2us - 50.8us/batch`
- `get_missing_rows=0`
- `pending_rpc_peak=16`
- client `copy_response_avg_ns=0`

这说明：

1. 当前 default 已经能打满配置的 prefetch depth，不是“完全没有并发”。
2. `copy_response_avg_ns=0` 表示 P3 borrowed response payload 生效，client 中间 response copy 已经不在主路径。
3. `get_missing_rows=0` 表示不是 KVEngine 容量/缺失导致的假慢。
4. `skip_prefetch_result_copy` 只从 `5.108M` 到 `5.283M keys/s`，说明最后 vector materialize 不是当前主墙。

诊断行对比更明确：

- `status_only=28.402M keys/s`：PS adapter 调度大洞已经被补上，否则 status-only 不可能到这个量级。
- `index_only=11.676M keys/s`：只做 index lookup 就明显低于 status-only，说明 key parse/index lookup 也有可见成本。
- `default=5.108M keys/s`：加回真实 512B value payload 后吞吐腰斩，server GET payload copy / batch get 成为当前主瓶颈。
- RC 高并发专项 `48.143M keys/s`：底层 transport 在更高并发调度下有明显容量，但它不是 generic PS API 语义，不能直接当作 PS default 目标值。

## 当前结论

之前最致命的问题是 PS adapter/prefetch 热路径与 RDMA RC 专项 benchmark 不对齐：每次 prefetch 分配/清零大 response buffer，再经过 generic prefetch map/result materialize。现在这部分已经明显改善，`status_only` 已经能跑到 `28.402M keys/s`。

当前剩余主瓶颈不是笼统的“server 慢”，而是更具体的：

```text
PetPSServer::HandleGet
  -> CachePS::GetParameterFlat
  -> KVEngineComposite::BatchGetFlat
  -> DramValueStore::ReadFlatFixedRows
  -> response payload 写回
```

在 `batch_keys=500`、`value_size=512` 下，真实 payload 每 batch 约 `256KB`。当前单 shard、单 server polling worker 的真实 GET 路径约 `90us+ / batch`，其中 row copy 约 `50us / batch`，所以 PS default 稳在约 `5.1M keys/s` 是和 profile 对得上的。

## 本轮命令

PS default：

```bash
python3 src/test/scripts/run_benchmark_ps.py --transports rdma --client-ips 127.0.0.1 --server-shard-ips 127.0.0.1 --client-processes-per-ip 4 --record-count 200000 --value-size 512 --batch-keys 500 --client-threads-per-process 1 --client-load-threads-per-process 1 --runtime-seconds 3 --repeat 1 --execution-backend local --prefetch-depth 16 --rdma-rc-qps-per-client-per-shard 16 --rdma-rc-slots-per-qp 1 --rdma-rc-profile-interval-ms 1000 --rdma-wait-timeout-ms 20000 --client-timeout 180 --cluster-timeout 60 --transaction-profile --output-dir results/benchmark_ps_current_compare_0530/ps_default
```

PS `status_only`：

```bash
python3 src/test/scripts/run_benchmark_ps.py --transports rdma --client-ips 127.0.0.1 --server-shard-ips 127.0.0.1 --client-processes-per-ip 4 --record-count 200000 --value-size 512 --batch-keys 500 --client-threads-per-process 1 --client-load-threads-per-process 1 --runtime-seconds 3 --repeat 1 --execution-backend local --prefetch-depth 16 --rdma-rc-qps-per-client-per-shard 16 --rdma-rc-slots-per-qp 1 --rdma-rc-profile-interval-ms 1000 --rdma-wait-timeout-ms 20000 --client-timeout 180 --cluster-timeout 60 --transaction-profile --rdma-rc-fake-get-mode status_only --output-dir results/benchmark_ps_current_compare_0530/ps_status_only
```

PS `index_only`：

```bash
python3 src/test/scripts/run_benchmark_ps.py --transports rdma --client-ips 127.0.0.1 --server-shard-ips 127.0.0.1 --client-processes-per-ip 4 --record-count 200000 --value-size 512 --batch-keys 500 --client-threads-per-process 1 --client-load-threads-per-process 1 --runtime-seconds 3 --repeat 1 --execution-backend local --prefetch-depth 16 --rdma-rc-qps-per-client-per-shard 16 --rdma-rc-slots-per-qp 1 --rdma-rc-profile-interval-ms 1000 --rdma-wait-timeout-ms 20000 --client-timeout 180 --cluster-timeout 60 --transaction-profile --rdma-rc-fake-get-mode index_only --output-dir results/benchmark_ps_current_compare_0530/ps_index_only
```

PS skip result copy 诊断：

```bash
python3 src/test/scripts/run_benchmark_ps.py --transports rdma --client-ips 127.0.0.1 --server-shard-ips 127.0.0.1 --client-processes-per-ip 4 --record-count 200000 --value-size 512 --batch-keys 500 --client-threads-per-process 1 --client-load-threads-per-process 1 --runtime-seconds 3 --repeat 1 --execution-backend local --prefetch-depth 16 --rdma-rc-qps-per-client-per-shard 16 --rdma-rc-slots-per-qp 1 --rdma-rc-profile-interval-ms 1000 --rdma-wait-timeout-ms 20000 --client-timeout 180 --cluster-timeout 60 --transaction-profile --rdma-adapter-skip-prefetch-result-copy --output-dir results/benchmark_ps_current_compare_0530/ps_skip_prefetch_result_copy
```

RC async stream，4 clients、1 thread：

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py --benchmark-binary ./build/bin/rdma_rc_transport_benchmark --server-count 1 --client-count 4 --thread-num 1 --iterations 100 --rounds 20 --warmup-rounds 5 --batch-keys 500 --value-size 512 --op async_stream --async-depth 16 --qps-per-client-per-shard 16 --report-mode summary --rdma-wait-timeout-ms 20000 --client-timeout 300 --cluster-timeout 60 > results/benchmark_ps_current_compare_0530/rdma_rc_async_stream_4c_t1/stdout.log 2> results/benchmark_ps_current_compare_0530/rdma_rc_async_stream_4c_t1/stderr.log
```

RC async stream，4 clients、16 threads：

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py --benchmark-binary ./build/bin/rdma_rc_transport_benchmark --server-count 1 --client-count 4 --thread-num 16 --iterations 100 --rounds 20 --warmup-rounds 5 --batch-keys 500 --value-size 512 --op async_stream --async-depth 16 --qps-per-client-per-shard 16 --report-mode summary --rdma-wait-timeout-ms 20000 --client-timeout 300 --cluster-timeout 60 > results/benchmark_ps_current_compare_0530/rdma_rc_async_stream_4c_t16/stdout.log 2> results/benchmark_ps_current_compare_0530/rdma_rc_async_stream_4c_t16/stderr.log
```

注意：`src/test/scripts/run_rdma_rc_transport_benchmark.py` 当前不支持 `--output-dir` 和 `--use-local-memcached auto` 参数，所以本轮通过 shell redirect 保存 RC stdout/stderr。
