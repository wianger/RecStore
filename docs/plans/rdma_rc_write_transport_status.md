# RDMA RC Write Transport Status

更新时间：2026-05-25

本文记录当前 RDMA RC write transport 压测、激进优化尝试和瓶颈判断，供后续继续排查参考。

## 当前代码状态

本轮围绕当前重写后的 `src/ps/rdma` RC write transport 做验证和优化，没有基于历史 Mayfly benchmark 路径继续实现。

已新增或修改的主要内容：

- 新增当前路径 benchmark：`src/benchmark/rdma_rc_transport_benchmark.cc`
- 新增 benchmark runner：`src/test/scripts/run_rdma_rc_transport_benchmark.py`
- 新增 multi-client stress runner：`src/test/scripts/run_petps_multiclient_stress.py`
- `petps_cluster_runner.py` 支持 `rdma_rc_qps_per_client_per_shard`
- RDMA integration CTest 增加 stress 入口
- `petps_integration_test.cpp` 增加重复 PUT/GET、multi-shard、async-looking prefetch stress
- `rc_transport.*` 增加更详细的 write completion context
- `petps_server.cc` 增加 invalid descriptor / wrong shard 的结构化错误日志

本轮激进优化尝试：

- client 侧 descriptor/payload write 改为 unsignaled
- client 侧 commit write 保持 signaled
- server 侧 response payload write 改为 unsignaled
- server 侧 status write 保持 signaled
- 不再每个 RPC 立刻等待当前 signaled completion
- 同一 lane / client slot 复用前 drain 上一个 signaled completion
- client/server transport 析构时 best-effort drain 残留 pending completion
- 每 RPC 成功路径日志从 `LOG(INFO)` 降为 `VLOG(1)`，错误日志保留
- benchmark 的 `async_get` / `async_depth` 输出 buffer 改为预分配复用

## 已通过验证

构建：

```bash
cmake --build build --target petps_integration_test rdma_rc_transport_benchmark -j4
```

结果：通过。

脚本和 runner：

```bash
python3 -m py_compile \
  src/test/scripts/run_rdma_rc_transport_benchmark.py \
  src/test/scripts/run_petps_multiclient_stress.py \
  src/test/scripts/petps_cluster_runner.py
```

结果：通过。

```bash
python3 -m unittest src/test/scripts/test_petps_cluster_runner.py
```

结果：20 个测试通过。

协议和 wrapper 单测：

```bash
ctest --test-dir build -R "test_rdma_rc_protocol|test_allshards_ps_client" -VV
```

结果：2 个 CTest 通过。

单 shard / 双 client stress：

```bash
python3 src/test/scripts/run_petps_multiclient_stress.py \
  --server-count 1 \
  --client-count 2 \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.RepeatedPutGetStressSingleShard:PetPSIntegrationTest.AsyncGetPrefetchStressSingleShard \
  --client-timeout 60 \
  --cluster-timeout 35 \
  --use-local-memcached=auto
```

结果：两个 client 均通过。

multi-shard / 双 client stress：

```bash
python3 src/test/scripts/run_petps_multiclient_stress.py \
  --server-count 2 \
  --client-count 2 \
  --config-path ./src/test/configs/recstore_config.rdma_multishard_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.RepeatedPutGetStressMultiShard \
  --client-timeout 60 \
  --cluster-timeout 35 \
  --use-local-memcached=auto
```

结果：两个 client 均通过。

## Benchmark 结果

### 95% GET / 5% PUT 实际工作负载

参数：

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 1 \
  --thread-num 1 \
  --iterations 100 \
  --rounds 5 \
  --warmup-rounds 3 \
  --batch-keys 4096 \
  --value-size 16 \
  --op mixed \
  --get-ratio 95 \
  --rdma-rc-qps-per-client-per-shard 64 \
  --client-timeout 180 \
  --cluster-timeout 60 \
  --use-local-memcached=auto
```

结果记录：

- 激进 completion 优化后：约 `1.408M key_ops/s`
- 日志降级后重跑：约 `1.270M key_ops/s`
- 结论：该负载没有因为 deferred completion 或日志降级出现稳定收益，结果存在 run-to-run 波动。

### 单 client async depth 极限点

参数：

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 1 \
  --thread-num 4 \
  --iterations 100 \
  --rounds 10 \
  --warmup-rounds 5 \
  --batch-keys 1024 \
  --value-size 4 \
  --op async_depth \
  --async-depth 32 \
  --rdma-rc-qps-per-client-per-shard 128 \
  --client-timeout 300 \
  --cluster-timeout 80 \
  --use-local-memcached=auto
```

结果记录：

- 优化前历史最好 clean rerun：约 `7.06M key_ops/s`
- deferred completion 后：约 `7.38M key_ops/s`
- 日志降级后重跑：约 `7.12M key_ops/s`
- benchmark buffer 复用后重跑：约 `7.08M key_ops/s`
- 结论：单 client async depth 极限点大约在 `7.0M - 7.4M key_ops/s` 区间，deferred completion 只有小幅收益，不是数量级瓶颈。

### 多 client 并发探测

共同参数：

- `server-count=1`
- `batch-keys=1024`
- `value-size=4`
- `op=async_depth`
- `async-depth=32`
- `rdma_rc_qps_per_client_per_shard=128`

结果：

| clients | server poll threads | rounds/iterations | aggregate key_ops/s | per-client key_ops/s 典型值 | 观察 |
|---:|---:|---:|---:|---:|---|
| 1 | 4 | 10 x 100 | `7.08M - 7.38M` | `7.08M - 7.38M` | 单 client 上限 |
| 4 | 8 | 5 x 100 | `11.10M` | `2.77M` | 聚合提升，但单 client 明显下降 |
| 8 | 16 | 4 x 80 | `12.54M` | `1.56M` | 边际收益变小 |
| 16 | 16 | 3 x 50 | `13.43M` | `0.84M` | 继续涨但基本进入平台/实现瓶颈区 |

16 client 短测完整 aggregate：

```text
clients=16 agg_ops/s=409.77 agg_key_ops/s=13,427,425 mean_req_us_avg=39,046.12
```

## 当前瓶颈判断

这轮数据说明：当前实现并发没有完全到硬极限，但继续加 client 的收益迅速变低。更大的问题不在单个 RDMA write completion 的等待成本上。

更可能的瓶颈路径：

1. server 侧轮询模型是所有 poll threads 扫固定 slot。client 数和 QP 数上去后，线程会重复扫描大量 idle slot，cache miss 和无效轮询成本上升。
2. 每个 request 仍是固定 slot 的单 in-flight lane 语义。async depth 依赖多 QP/lane 堆出来，缺少单 lane 多 outstanding 或真正 request queue。
3. server 完成每个 GET 都要调用 `CachePS::GetParameterFlat` 并写 response payload。小 payload 下协议/QP/轮询开销占比很高。
4. client 侧 `WaitStatus` 是 busy-yield 轮询，多个 client 进程同时运行时 CPU 调度压力明显。
5. 当前 `PutParameter` / `UpdateParameter` 仍有 payload 构造和逐批同步路径，95/5 mixed 负载会被 PUT 路径拖住。
6. benchmark 里的 key 集合固定，主要测 hot-key read 路径和协议开销，不代表真实随机 key 分布。

## 当前建议

暂停继续盲目加参数，下一步应认真定位瓶颈：

1. 加 server 侧 profiling counters：每个 poll thread 的 scanned_slots、ready_slots、handled_ops、empty_scan_rounds、per-op latency。
2. 加 client 侧 counters：submit time、wait status time、revoke time、QP acquire failure/等待、pending depth。
3. 给 benchmark 增加 `--key-space` / `--random-keys` / `--reuse-output-buffer` 等参数，把协议上限和存储路径上限分开测。
4. 对 server poll loop 做结构性优化：按 active QP/slot 做分区，或引入 doorbell / completion notification，减少全表扫描 idle slot。
5. 评估单 lane 多 outstanding request 或 ring slot，而不是只靠增加 QP 数堆并发。
6. 重新测 server poll thread sweep：`1/2/4/8/16/32`，每档记录 CPU 使用、aggregate key_ops/s 和 p95 request latency。

## 注意事项

- 当前 worktree 里存在 third_party submodule 和 `tmpdocs/` 的既有脏状态，本轮没有清理或回滚。
- 当前 benchmark 数据来自本地环境单次或少数几次运行，适合做方向判断，不适合作为最终性能结论。
- 95/5 mixed 的稳定吞吐仍需要更长 rounds 和更多 client/server 参数组合复测。
