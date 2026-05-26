# RDMA 模块运行手册

本文档整理当前 RecStore RDMA 主路径的边界、参数、验证入口、已知限制和下一步路线图。默认工作目录为仓库根目录：

```bash
cd /app/RecStore
```

## 1. 适用范围

当前文档只覆盖 Parameter Server 里的 RDMA 主路径，不讨论 gRPC / bRPC 的通用网络栈。

RecStore 现在有两条 RDMA 入口：

| 层级 | 入口 | 用途 |
|------|------|------|
| PetPS RDMA | `petps_server` + `PetPSClient` | RDMA 数据面、协议验证、transport benchmark |
| Op-layer RDMA | `RDMAPSClientAdapter` + `KVClientOp` | 通过统一 op 接口验证 RDMA 后端 |

两条入口复用同一套 RDMA 传输，但初始化和调用方式不同：

- PetPS integration 和 benchmark 主要通过 C++ gflags 传参。
- Op-layer / Python client 主要通过环境变量和测试配置传参。
- 不要把脚本参数、C++ gflag 和环境变量混用到错误入口。

Op-layer RDMA 目前不是 gRPC/bRPC 的完整替代：

- `AsyncGetParameter` 和 `Command` 还未实现。
- `UpdateParameter` 走同步 read-modify-write。
- 更适合作为 correctness / integration 路径，而不是完整性能替代路径。

## 2. 当前 RDMA 架构

### 2.1 传输模式

当前 PetPS RDMA 入口使用 RC-write slot transport：

- `RequestDescriptor + payload` 先写入 server request slot。
- `CommitWord` 最后写入，server 通过轮询 commit word 发现请求。
- server 处理后先写 client response payload。
- `StatusWord` 最后写回，client 通过轮询 status word 判断完成。

当前实现还支持 `slots_per_qp` 这一层逻辑复用：同一条 QP lane 上可以挂多个逻辑 slot，实际的 in-flight 上限是 `qps-per-client-per-shard * slots-per-qp`，不是只看 QP 数。

### 2.2 代码分工

`src/ps/rdma/raw_verbs_transport.*` 已经不再是 `shm_open + mmap` baseline，而是直接走真实 verbs RC 路径：

- 打开 RDMA 设备
- 注册本地 MR
- 创建 QP / CQ
- 交换 memcached metadata
- 轮询 RDMA write completion

`src/ps/rdma/rc_transport.*` 负责：

- 固定 slot 布局
- shard / client / lane offset 计算
- request / response 提交流程
- 连接和回写的 profile 统计

`src/ps/rdma/petps_client.*` 和 `src/ps/rdma/petps_server.*` 负责：

- 把 RDMA slot 映射成 PetPS 的 request / response
- client QP 选择与 in-flight 管理
- server slot 扫描、协议处理、response 完成

### 2.3 现在的关键约束

- `qps-per-client-per-shard` 不是“吞吐参数”，而是 client 侧可用的 QP 池规模。
- `slots-per-qp` 是每条 QP 上的逻辑 slot 数，默认是 `1`，主要用来扩展 in-flight 上限而不是增加 QP 数。
- `async_stream` 下，`qps-per-client-per-shard * slots-per-qp` 不能小于 `async-depth`。
- 如果硬件 QP 资源不足，即使参数合法，`RawVerbsTransport` 也会在 client 初始化阶段直接报错并拒绝启动。
- 这比“跑到一半再卡住”更容易定位，也更符合当前的 fail-fast 目标。

## 3. 构建

常用 RDMA 目标：

```bash
cmake --build ./build --target \
  ps_transport_benchmark \
  rdma_rc_transport_benchmark \
  petps_server \
  petps_integration_test \
  recstore_torch_ops \
  test_allshards_ps_client \
  -j
```

如果刚改过 `src/ps/rdma/*`、`src/test/scripts/*rdma*` 或 op-layer 相关代码，先重编对应目标再判断行为。旧的 `petps_server`、`ps_transport_benchmark`、`rdma_rc_transport_benchmark` 或 `recstore_torch_ops` 二进制很容易造成“源码已改但测试仍卡住”的假象。

## 4. 参数说明

下面这张表只列 RDMA benchmark 和 runtime 中最常见、最容易误解的参数。

| 参数 | 含义 | 备注 |
|------|------|------|
| `--iterations` | 每个 round 内执行的请求次数 | 影响单轮总请求数，通常和 `batch-keys` 一起看 |
| `--rounds` | 计入统计的测量轮数 | 最终吞吐由这些轮次聚合得到 |
| `--warmup-rounds` / `--rdma-warmup-rounds` | 热身轮数 | 不计入结果，只用于预热连接、缓存和内存路径 |
| `--batch-keys` / `--batch_keys` | 每次请求携带的 key 数 | 决定单次 RPC 的负载大小 |
| `--thread-num` / `--rdma-thread-num` | server 侧 RDMA polling thread 数 | 影响 server 轮询和并发处理能力 |
| `--client-count` | benchmark client 进程数 | 仅 `rdma_rc_transport_benchmark` 支持多 client |
| `--server-count` / `--num_shards` | server / shard 数量 | 多分片时要保证配置和路由一致 |
| `--qps-per-client-per-shard` | 每个 client 到每个 shard 的 QP 数 | 单独只表示 QP 池规模；真实 in-flight 上限还要乘 `--slots-per-qp` |
| `--slots-per-qp` | 每条 QP 上可复用的逻辑 slot 数 | 默认 `1`；适合在不增加 QP 数的情况下提高在途深度 |
| `--async-depth` | `async_stream` 里单 client 的在途请求深度 | 低负载上限测试的关键参数 |
| `--rdma-put-protocol-version` | PUT 协议版本 | `1` 是 legacy，`2` 是当前主路径 |
| `--rdma-put-v2-transfer-mode` | PUT-v2 payload 传输方式 | `read` 表示 server 读 payload，`push` 表示 client 主动写 payload |
| `--rdma-wait-timeout-ms` | RDMA 请求等待超时 | 过短会导致 benchmark 误判为超时失败 |
| `--profile-interval-ms` | RDMA RC 统计输出间隔 | `0` 表示不做周期性 profiling，仅输出 benchmark 结果 |
| `--server-coroutines-per-thread` | 每个 polling thread 上的 server 协程数 | 值越大越偏向 coop 扫描，不代表一定更快 |
| `--fake-get-mode` | benchmark-only fake GET 行为 | `none`、`status_only`、`payload_memset` |
| `--skip-client-copy` | 是否跳过 client 端 GET payload 拷贝 | 只用于 benchmark 排查，不适合作为默认配置 |

### 4.1 重要解读

- `batch-keys` 是请求粒度，不是总样本数。
- `iterations * rounds * batch-keys` 才是测量期的 key 总量。
- `read` 和 `push` 的结果不能直接混成一个吞吐结论。
- `qps-per-client-per-shard` 变大通常会增加连接并发，但也会增加资源占用和初始化成本。
- `slots-per-qp` 变大通常会增加单条 QP 的并发深度，但也会增加每条 lane 的本地状态和回写压力。
- `profile-interval-ms` 只影响周期性统计输出，不应和吞吐提升直接画等号。

### 4.2 启动前硬约束

当前 benchmark 已经加入启动前校验：

- `async_stream` 要求 `qps-per-client-per-shard * slots-per-qp >= async-depth`
- 否则直接拒绝启动，避免把 client slot 池打满后在运行中报 `no idle RC write slot available`

如果你要做低负载上限测试，建议先确保这个约束满足，再看真实瓶颈。

## 5. Profile 统计怎么读

当前 RDMA profile 已经拆成三层，足够定位大部分低负载瓶颈。

### 5.1 Client 侧

来自 `component=rdma_rc_client_profile`，重点看这些字段：

- `submit_request_ns`
- `wait_status_ns`
- `copy_response_ns`
- `revoke_resource_ns`
- `pending_rpc_peak`
- `acquire_qp_count`
- `acquire_qp_failures`

解读方式：

- `submit_request_ns` 高，说明一次请求发出去的固定开销重。
- `wait_status_ns` 高，说明 completion / 回写链路慢，或者 server 处理慢。
- `copy_response_ns` 高，说明 response copy 成为低负载瓶颈。
- `pending_rpc_peak` 高，说明在途请求池压力大。
- `acquire_qp_failures` 不是“慢”，而是“资源不够”。

### 5.2 Server 侧

来自 `component=rdma_rc_server_profile`，重点看这些字段：

- `poll_loop_ns`
- `scan_rounds`
- `scanned_slots`
- `ready_slots`
- `empty_scan_rounds`
- `handle_get_ns`
- `get_batch_get_ns`
- `get_zero_fill_ns`
- `get_row_copy_ns`
- `handle_put_ns`
- `handle_update_ns`
- `handle_init_ns`
- `complete_response_ns`

解读方式：

- `poll_loop_ns` 高，通常说明空轮询太多，是低负载上限的第一嫌疑。
- `empty_scan_rounds` 高，说明扫描很多次但没命中请求。
- `get_batch_get_ns` 高，说明 batch 查找本身重。
- `get_zero_fill_ns` 高，说明缺失值或响应缓冲清零成本偏高。
- `get_row_copy_ns` 高，说明数据搬运是瓶颈。
- `complete_response_ns` 高，说明回写完成链路重。

### 5.3 Transport 侧

来自 `component=rdma_rc_transport_profile`，重点看这些字段：

- `submit_request_ns`
- `drain_pending_submit_ns`
- `complete_response_ns`
- `drain_pending_response_ns`
- `submit_descriptor_write_count`
- `submit_commit_write_count`
- `response_payload_write_count`
- `response_status_write_count`

解读方式：

- `submit_request_ns` 高，说明 client 侧 verbs 提交开销偏大。
- `drain_pending_submit_ns` 高，说明上一笔请求没有及时完成，出现提交背压。
- `complete_response_ns` 高，说明 server 侧回写成本偏大。
- `drain_pending_response_ns` 高，说明 response path 也有背压。

## 6. Benchmark 建议

### 6.1 Benchmark 入口选择

当前有两个 benchmark 二进制，它们不是重复目标：

| 目标 | Runner | 适用场景 |
|------|--------|----------|
| `ps_transport_benchmark` | `src/test/scripts/run_rdma_transport_benchmarks.py` | 通用 PS transport 对比入口，可跑 RDMA / gRPC / bRPC；RDMA 模式主要用于验证 PUT-v2 `read` 和 `push` 两条路径。 |
| `rdma_rc_transport_benchmark` | `src/test/scripts/run_rdma_rc_transport_benchmark.py` | RDMA RC 专项压测入口，支持 `client-count`、`async_stream`、QP 池、server coroutine、fake get、skip client copy 等 RC 诊断参数。 |

选择规则：

- 想比较 RDMA 和 gRPC / bRPC，或验证 PUT-v2 `read` / `push` 传输模式，用 `ps_transport_benchmark`。
- 想测 RDMA RC 本身的单 shard、多 client、低负载上限或 async pipeline，用 `rdma_rc_transport_benchmark`。
- 不要把两个入口的参数混用。`--rdma-put-v2-transfer-mode` 属于通用入口；`--qps-per-client-per-shard`、`--async-depth`、`--client-count` 属于 RC 专项入口。

### 6.2 通用 PS Transport 入口

建议先跑 PetPS RC-write correctness 基线，再做 benchmark：

```bash
python3 src/test/scripts/run_rdma_transport_benchmarks.py \
  --benchmark-binary ./build/bin/ps_transport_benchmark \
  --iterations 300 \
  --batch-keys 500 \
  --rounds 20 \
  --rdma-warmup-rounds 10 \
  --report-mode summary \
  --rdma-only \
  --rdma-thread-num 1 \
  --rdma-put-protocol-version 2 \
  --rdma-put-v2-transfer-mode read \
  --rdma-wait-timeout-ms 20000 \
  --rdma-client-timeout-sec 60 \
  --show-runner-logs \
  --use-local-memcached auto
```

summary 表中的 `put_v2` 列用于确认 PUT-v2 payload transfer mode；`read` 和 `push` 的结果不能直接混比。

如果你想在不继续增加 QP 数的情况下抬高 `async_stream` 深度，可以优先调 `--slots-per-qp`，再看 `qps-per-client-per-shard` 是否还需要一起放大。

### 6.3 RDMA RC 专项入口

最小真实 RDMA RC 闭环命令：

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 1 \
  --thread-num 1 \
  --iterations 2 \
  --rounds 1 \
  --warmup-rounds 0 \
  --batch-keys 4 \
  --value-size 16 \
  --op get \
  --report-mode summary \
  --qps-per-client-per-shard 1 \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 60 \
  --cluster-timeout 30 \
  --use-local-memcached auto
```

多 client 最小解析验证：

```bash
python3 src/test/scripts/run_rdma_rc_transport_benchmark.py \
  --benchmark-binary ./build/bin/rdma_rc_transport_benchmark \
  --server-count 1 \
  --client-count 2 \
  --thread-num 1 \
  --iterations 2 \
  --rounds 1 \
  --warmup-rounds 0 \
  --batch-keys 4 \
  --value-size 16 \
  --op get \
  --report-mode summary \
  --qps-per-client-per-shard 1 \
  --rdma-wait-timeout-ms 20000 \
  --client-timeout 60 \
  --cluster-timeout 30 \
  --use-local-memcached auto \
  --quiet
```

判定标准：

- 输出必须包含 `phase=measure summary`。
- 单 client 表必须出现 `RDMA RC Benchmark Summary` 和 `RDMA RC Aggregate Summary`。
- 多 client `--quiet` 模式至少要在 aggregate 表中看到 `clients` 等于实际 client 数。
- 如果只看到原始 benchmark 行但没有 summary 表，优先检查 runner 的 summary 正则和多 client stdout 行边界。

### 6.4 当前更重要的 benchmark 目标

现在的重点不是继续把 `client-count` 往上堆，而是：

- 低负载下提高单 shard 的 req qps 上限
- 看清哪些是固定开销，哪些是资源上限
- 找出可以稳定提升上限的最小改动

因此，建议优先用这些组合看上限：

- `op=get`
- `op=async_get`
- 小 `batch-keys`
- 小到中等 `async-depth`
- `client-count=1/2`

### 6.5 低负载上限测试建议

建议把这些参数固定住，只改一个维度：

- `value_size`
- `batch-keys`
- `iterations`
- `rounds`
- `thread-num`
- `server-count`

优先扫描的维度：

- `async-depth`
- `server-coroutines-per-thread`
- `qps-per-client-per-shard`
- `profile-interval-ms`

## 7. 当前状态

当前 RDMA RC 主路径已经不是“单 QP 单 slot”的旧模型了，而是按 `qp_index + slot_in_qp` 共同寻址：

- client 侧会为每个 `qp_index` 分配一组逻辑 slot。
- server 侧会按 `client_id / qp_index / slot_in_qp` 反解请求槽位。
- `slots_per_qp` 已经贯通到 client、server、runner、测试和 benchmark 参数解析。

这意味着文档里判断并发能力时，不能再只看 `qps-per-client-per-shard`，而要看 `qps-per-client-per-shard * slots-per-qp`。

如果你要补 benchmark 数字，建议把结果放到单独的 `results/` 目录，不要把一次临时跑出来的吞吐值长期固化在主文档里。

## 8. 当前路线图

目标是提升低负载下的 req qps 上限，而不是单纯追更高并发。

### 8.1 第一优先级：减少 server 空转

先看 `petps_server` 的：

- `poll_loop_ns`
- `empty_scan_rounds`
- `scan_hit_pct`

如果低负载下空扫太多，说明当前轮询模式过重。优先方向是：

- 减少无效扫描
- 让“没有活跃请求”的时候更轻
- 避免 polling thread 大量空转抢 CPU

### 8.2 第二优先级：压低 client 提交和等待成本

再看 `PetPSClient` 的：

- `submit_request_ns`
- `wait_status_ns`
- `copy_response_ns`
- `pending_rpc_peak`

如果低负载下这些值仍然偏高，说明单请求固定开销太大。优先方向是：

- 减少锁争用
- 减少 QP 选择和状态维护开销
- 减少 response copy 和 slot 清理成本

### 8.3 第三优先级：压薄 transport 提交/完成路径

再看 `rc_transport` 的：

- `submit_avg_ns`
- `complete_avg_ns`
- `drain_submit_avg_ns`
- `drain_response_avg_ns`

如果这里高，说明 verbs 提交 / 完成本身已经是瓶颈。优先方向是：

- 减少不必要的 write / complete 往返
- 让提交和完成路径更短
- 避免让背压在低负载下提前出现

### 8.4 不作为主路线的方向

- 继续加 `client-count`
- 继续抬高 `async-depth` 到超过 QP 池承受范围
- 把吞吐增长归因于“更多并发”而不是“更低的固定开销”

## 9. 已知限制和失败模式

### 9.1 QP 资源不足

如果 `client-count` 或 `qps-per-client-per-shard` 太大，可能出现：

- `ibv_create_qp failed`
- `no idle RC write slot available`
- client 初始化阶段直接报错退出

如果是 `slots-per-qp` 太小，更常见的是启动前被 `async_stream` 的容量校验拦住，或者运行时更早触发 `no idle RC write slot available`。

这是资源不足，不是正常性能退化。

### 9.2 参数不合法

`async_stream` 下如果 `qps-per-client-per-shard * slots-per-qp < async-depth`，benchmark 会直接拒绝启动。

### 9.3 旧二进制

如果看到了看似“不对劲”的行为，先确认 binary 是最新构建的目标。RDMA 这条链路对旧二进制非常敏感。

### 9.4 memcached 依赖

RDMA 脚本通过 memcached 交换元数据。常用建议是：

```bash
--use-local-memcached auto
```

含义：

| 值 | 行为 |
|----|------|
| `auto` | 优先复用外部 memcached；不可用时启动本地 memcached |
| `always` | 总是启动本地 memcached |
| `never` | 只使用已经存在的外部 memcached |

## 10. 验证入口

### 10.1 PetPS Integration

单分片：

```bash
python3 src/test/scripts/run_petps_integration.py \
  --server-count 1 \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.UpdateGetRoundTripSingleShard \
  --use-local-memcached=auto \
  --show-runner-logs \
  --client-timeout=20 \
  --cluster-timeout=35
```

多分片：

```bash
python3 src/test/scripts/run_petps_integration.py \
  --server-count 2 \
  --client-count 1 \
  --config-path ./src/test/configs/recstore_config.rdma_multishard_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripMultiShard \
  --use-local-memcached=auto \
  --show-runner-logs \
  --client-timeout=25 \
  --cluster-timeout=45
```

多分片排障时优先检查：

- `distributed_client.num_shards`
- `distributed_client.servers`
- `server-count`
- `num_server_processes`
- key 到 shard 的路由是否一致

当前 integration 覆盖的场景：

- `PutGetRoundTripSingleShard`
- `UpdateGetRoundTripSingleShard`
- `PutGetRoundTripMultiShard`

这些场景由 `run_petps_integration.py` 驱动，实际是否通过仍要看当次运行结果；文档这里只记录覆盖范围，不把一次运行结论长期固化在正文里。

### 10.2 Op-layer RDMA

Op-layer RDMA 使用配置：

```text
src/test/configs/recstore_config.op_rdma.json
```

常用 ctest：

```bash
ctest --test-dir ./build -R "^test_op_runtime_support$|^test_op$" -VV
ctest --test-dir ./build -R "^pytorch_client_test_rdma_basic$" -VV
ctest --test-dir ./build -R "^pytorch_client_test_rdma$|^pytorch_client_test_rdma_auto$" -VV
```

也可以手工运行基本 RDMA 客户端测试：

```bash
RECSTORE_CONFIG=./src/test/configs/recstore_config.op_rdma.json \
RECSTORE_CLIENT_TEST_PHASE=basic \
RECSTORE_USE_LOCAL_MEMCACHED=auto \
python3 src/test/framework/pytorch/test_client.py ./build/lib/lib_recstore_ops.so
```

这些测试的 `SKIP_RETURN_CODE` 是 `77`。skip 只说明 helper 的前置检查没有通过，不等价于真实 RDMA benchmark 一定不可运行。

当前 op-layer RDMA 覆盖：

- `test_op_runtime_support`
- `test_op`
- `pytorch_client_test_rdma_basic`
- `pytorch_client_test_rdma`

这些测试覆盖 PyTorch custom op 到 `RDMAPSClientAdapter -> PetPSClient -> verbs RC` 的 init/write/read、prefetch 和 table-aware update roundtrip；如果 helper 前置条件不满足，`SKIP_RETURN_CODE` 仍然是 `77`。

## 11. 单测和脚本测试

协议 helper 和 wrapper：

```bash
ctest --test-dir ./build -R "^test_allshards_ps_client$" -VV
```

runner 参数拼接：

```bash
python3 -m unittest src/test/scripts/test_petps_cluster_runner.py
python3 -m unittest src/test/scripts/test_run_rdma_rc_transport_benchmark.py
python3 -m unittest src/test/scripts/test_run_rdma_transport_benchmarks.py
```

这些测试不证明 RDMA 数据面可用，只证明协议编码、分片 wrapper 和脚本 plumbing 没有明显回归。其中 `test_run_rdma_rc_transport_benchmark.py` 覆盖：

- `--quiet` 模式只输出 summary / aggregate，不输出 progress 噪声。
- 多 client 流式 stdout 保留行边界，避免多条 `phase=measure summary` 粘连后只解析到第一条。

真实 RDMA 数据面至少跑一个最小 benchmark 闭环，推荐先用 `6.3` 的单 client 命令，再用 2-client `--quiet` 命令确认聚合表中的 `clients=2`。

## 12. 排障顺序

1. 确认二进制是最新构建的目标，尤其是 `petps_server`、`ps_transport_benchmark`、`petps_integration_test` 和 `recstore_torch_ops`。
2. 确认参数传到了正确入口：runner 用 RDMA 专项参数，C++ binary 读对应 gflags，op-layer 读 `RECSTORE_CONFIG` 和相关环境变量。
3. 确认 RDMA 设备和 memcached 可用：检查 `/dev/infiniband`、`ibv_devices`、`ss -ltnp | grep ':21211'`。
4. 如果 runner 卡在 `memcached-wait` 或 `startup-wait`，先看 runner 捕获的 server 日志。
5. 如果看到 `unknown command line flag 'rdma_transport_mode'`，通常是跑到了旧 binary，或者当前目标没有链接 RDMA client 相关对象。
6. 如果看到 `message size too large`，先把 `batch-keys` 降到 500 或更小建立稳定基线。
7. 如果 RDMA 路径卡住，重点检查 raw verbs buffer 是否注册、QP metadata 是否按 shard/lane 匹配、CQ 是否被错误线程消费、server/client mode 是否一致。

最小日常验证顺序：

```bash
cmake --build ./build --target ps_transport_benchmark rdma_rc_transport_benchmark petps_server recstore_torch_ops -j
python3 src/test/scripts/run_petps_integration.py \
  --server-count 1 \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.UpdateGetRoundTripSingleShard \
  --use-local-memcached=auto
ctest --test-dir ./build -R "^pytorch_client_test_rdma_basic$" -VV
```
