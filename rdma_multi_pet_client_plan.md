# RDMA 多 PetPSClient 逻辑客户端修复计划

日期：2026-06-03

状态：已实现并验证。当前跨机推荐 baseline 见
`cross_host_rdma_benchmark_diagnosis_0602.md` 和
`.agents/skills/cross-host-rdma-ps/SKILL.md`。

## 背景

当前 `ps_transport_benchmark` 的 RDMA transactions 路径强制：

```text
FLAGS_thread_num == 1
```

原因不是 RDMA 理论上不能一个进程内多个 worker，而是当前 `PetPSClient` 的逻辑 client id 来自进程级 gflags：

```text
client_id = FLAGS_global_id - FLAGS_num_server_processes
```

如果一个 benchmark process 内直接创建多个 `PetPSClient`，它们会拿到同一个 `client_id`，写同一组 server request slot 和 client response slot，导致 slot 覆盖、status 混乱和协议错误。

因此，若要支持“一个客户端进程内多个 worker / 多个 PetPSClient”，必须先让每个 PetPSClient 拿到独立的逻辑 client id，并让 server 的 `num_client_processes` 表示逻辑 client 总数，而不仅是 OS process 总数。

## 已有实验背景

在不修改代码的前提下，已经尝试通过增加 OS client process 数和调整每 client 的 QP/depth 来观察请求密度：

| 配置 | 结果目录 | run 吞吐 |
|-|-|-:|
| p6/q16/depth16 | `results/benchmark_ps_cross_host_rdma_p6_profile_fields_0602` | 16.488 M keys/s |
| p6/q2/depth2 | `results/benchmark_ps_cross_host_rdma_p6_q2_d2_0603` | 6.762 M keys/s |
| p6/q4/depth4 | `results/benchmark_ps_cross_host_rdma_p6_q4_d4_0603` | 11.032 M keys/s |
| p8/q2/depth2 | `results/benchmark_ps_cross_host_rdma_p8_q2_d2_0603` | 6.772 M keys/s |
| p8/q4/depth4 | `results/benchmark_ps_cross_host_rdma_p8_q4_d4_0603` | 12.939 M keys/s |
| p8/q8/depth8 | `results/benchmark_ps_cross_host_rdma_p8_q8_d8_0603` | 16.487 M keys/s |
| p8/q16/depth16 | `results/benchmark_ps_cross_host_rdma_p8_q16_d16_0603` | 16.571 M keys/s |

结论：

- `q2/q4` 太浅，会显著拉长 client wait/status 闭环，吞吐明显低于基线。
- `p8/q8` 基本追平 `p6/q16`，说明 q8 已经足够接近当前跨机平台。
- 单纯增加 OS client process 到 p8 或继续使用 q16 不能突破约 `16.5 M keys/s`。
- `p12/q2` 触发控制面 metadata timeout，不适合作为数据面吞吐结论。

因此，下一步如果要验证“更多客户端线程/请求源是否能提升密度”，更合适的是修复同一进程内多个 `PetPSClient` 的 logical client id，而不是继续盲目增加 OS process。

## 目标

1. 允许 RDMA benchmark 中一个 client process 内创建多个独立 `PetPSClient`。
2. 每个 `PetPSClient` 拥有唯一 logical client id。
3. server 仍按 logical client id 分配 request/response slot。
4. `thread_num=1` 时行为完全兼容现有路径。
5. 先修 benchmark / PetPS RDMA 路径，不扩散到 Python/op-layer。

## 推荐设计

### 逻辑 client id

新增一个 client base id 参数，例如：

```text
--rdma_rc_client_id_base=<int>
```

在单进程多 worker 时：

```text
logical_client_id = rdma_rc_client_id_base + worker_tid
```

server 看到的总客户端数改为：

```text
logical_num_clients = client_process_count * client_threads_per_process
```

对于现有 `thread_num=1`：

```text
rdma_rc_client_id_base = client_process_index
logical_client_id = client_process_index
```

### Runner 参数

`PetPSClusterRunner.build_client_cmd()` 当前为第 `client_index` 个进程设置：

```text
--global_id = num_servers + client_index
--num_client_processes = client_process_count
```

扩展后建议：

```text
logical_clients_per_process = client_threads_per_process
logical_client_count = client_process_count * logical_clients_per_process
client_id_base = client_index * logical_clients_per_process
```

并传：

```text
--num_client_processes=<logical_client_count>
--rdma_rc_client_id_base=<client_id_base>
```

`--global_id` 仍可保留为进程级 node id，用于控制面和 raw verbs node id；但 `PetPSClient` 的 logical client id 不应再强制从 `global_id` 推导。

### PetPSClient 构造

给 `PetPSClient` 增加可选 logical client id override：

```text
PetPSClient(host, port, shard, logical_client_id = -1)
```

或新增 setter / gflag：

```text
--rdma_rc_logical_client_id
```

benchmark 多 worker 场景更适合构造时显式传入，避免多个线程改全局 flag。

`InitializeTransport()` 中：

```text
if explicit logical_client_id >= 0:
  client_id_ = explicit logical_client_id
else:
  client_id_ = FLAGS_global_id - FLAGS_num_server_processes
```

### Benchmark client 创建

当前 `RunPrefetchFetchTransactions()` 每个 worker 可以从 `reusable_clients[tid]` 拿 client。修复后：

1. RDMA 不再强制 `thread_num == 1`。
2. RDMA 下预创建 `thread_num` 个 reusable clients。
3. 第 `tid` 个 reusable client 使用：

```text
logical_client_id = rdma_rc_client_id_base + tid
```

4. preload 和 run 继续复用同一个 client，避免 preload/run 的 client id 不一致。

## 必须保持的约束

1. 每个 logical client id 只能由一个 worker 拥有。
2. 多个 PetPSClient 不能共享同一组 request/response slot。
3. server `num_client_processes` 必须等于 logical client 总数。
4. `prefetch_depth <= qps_per_client_per_shard * slots_per_qp` 仍按每个 logical client 独立检查。
5. `global_id` 仍是进程级 raw verbs node id，不能简单把 worker tid 映射到 global id，否则同进程多 QP metadata 会混乱。

## 验证计划

### 本机 smoke

```text
p1 process, thread_num=2, qps=8, depth=8
p1 process, thread_num=4, qps=8, depth=8
```

检查：

- 不再触发 `FLAGS_thread_num == 1`；
- server profile 中 `client_id` 范围正确；
- 无 slot descriptor mismatch；
- preload 和 run 都成功。

### 跨机验证

对比：

```text
p6/t1/q16/depth16  baseline
p6/t2/q8/depth8
p6/t4/q4/depth4
p8/t1/q8/depth8
```

其中 `p6/t2` 的 logical client 数为 12，`p6/t4` 的 logical client 数为 24。重点看是否比单纯增加进程更稳定，是否能避免 p12 控制面启动压力。

## 风险

1. 同一 OS process 内多个 PetPSClient 共享同一个 raw verbs global node id，底层 QP metadata key 是否允许多 client-id 共用同一 node id 需要确认。
2. 如果 raw verbs metadata 仍按 `global_id` 唯一区分节点，多 PetPSClient 可能覆盖 metadata；这时需要把 logical client id 纳入 metadata namespace 或 QP key。
3. 控制面 `num_client_processes` 从进程数变为逻辑 client 数后，其他 runner/test 需要同步理解。
4. benchmark 内多个 worker 的 core 绑定、load phase 和 run phase 需要保持一致，否则会混入调度噪声。

## 第一版成功标准

1. `thread_num=1` 现有 benchmark 完全不变。
2. `thread_num=2` 能在本机和跨机成功启动。
3. 每个 worker 的 `PetPSClient` 使用不同 logical client id。
4. server 无 descriptor/client id mismatch。
5. 跨机结果能用于判断 client 请求环路密度是否是主瓶颈。

## 实施结果

已完成的代码改动：

1. 新增 `--rdma_rc_client_id_base` 和
   `--rdma_rc_num_logical_clients`，将 OS client process 数和 logical
   client 数拆开。
2. `PetPSClient` 支持显式 logical client id override。
3. `ps_transport_benchmark` 的 RDMA transactions 路径允许
   `thread_num > 1`，并为每个 worker 创建独立 logical client。
4. server slot 协议仍按 logical client id 分配 request/response slot。
5. raw verbs node id 仍映射到 OS client process；同进程多个 logical
   client 通过扩展 raw lane namespace 避免 QP metadata/response buffer 冲突。
6. `PetPSClusterRunner` 根据
   `client_processes * client_threads_per_process` 传递 logical client 总数，
   并为每个 client process 传递 client id base。

验证：

```text
python3 -m unittest src/test/scripts/test_petps_cluster_runner.py src/test/scripts/test_run_benchmark_ps.py
cmake --build build --target ps_transport_benchmark petps_server -j
git diff --check
```

跨机关键结果：

| 配置 | logical clients | 结果目录 | run 吞吐 |
|-|-:|-|-:|
| p6/t1/q16/depth16 | 6 | `results/benchmark_ps_cross_host_rdma_p6_profile_fields_0602` | 16.488 M keys/s |
| p6/t2/q16/depth16 | 12 | `results/benchmark_ps_cross_host_rdma_p6t2_q16_d16_0603` | 31.639 M keys/s |
| p8/t2/q16/depth16 | 16 | `results/benchmark_ps_cross_host_rdma_p8t2_q16_d16_0603` | 31.270 M keys/s |
| p4/t3/q16/depth16 | 12 | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_0603` | 39.087 M keys/s |
| p3/t4/q16/depth16 | 12 | `results/benchmark_ps_cross_host_rdma_p3t4_q16_d16_0603` | 35.733 M keys/s |
| p4/t4/q16/depth16 | 16 | `results/benchmark_ps_cross_host_rdma_p4t4_q16_d16_0603` | 38.861 M keys/s |

结论：

- 修复多 logical client 后，跨机 RDMA PET_HASH 吞吐从约 `16.5 M keys/s`
  提升到 `39.1 M keys/s`，接近本机约 `41.5 M keys/s` 基线。
- 旧低分主要来自 client 请求源密度和组织方式不足，而不是 server
  GET handler 或 PET_HASH 查找能力不足。
- 当前最佳点是 `p4/t3/q16/depth16`；继续增加 OS process 或每进程
  logical client 数不一定更好。
