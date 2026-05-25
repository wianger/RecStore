# RDMA 模块运行手册

本文档只记录当前 RDMA 路径的边界、构建、运行和验证入口。默认工作目录为仓库根目录：

```bash
cd /app/RecStore
```

## 当前边界

RecStore 目前有两层 RDMA 入口：

| 层级 | 入口 | 用途 |
|------|------|------|
| PetPS RDMA | `petps_server` + `PetPSClient` | RDMA 数据面、协议验证、transport benchmark |
| Op-layer RDMA | `RDMAPSClientAdapter` + `KVClientOp` | 通过统一 op 接口验证 RDMA 后端 |

两层复用 PetPS/RDMA 数据面，但初始化方式不同：

- PetPS integration 和 benchmark 主要通过 C++ gflags 传参。
- Op-layer / Python client 主要通过环境变量和测试配置传参。
- 不要把脚本参数、C++ gflag 和环境变量混用到错误入口。

Op-layer RDMA 还不是 gRPC/bRPC 的完整替代：`AsyncGetParameter` 和 `Command` 未实现，`UpdateParameter` 走同步 read-modify-write，当前更适合作为 correctness / integration 路径。

## Transport Mode

当前 PetPS RDMA 入口使用 RC-write slot transport：

- `RequestDescriptor + payload` 先写入 server request slot。
- `CommitWord` 最后写入，server 通过轮询 commit word 发现请求。
- server 处理后先写 client response payload。
- `StatusWord` 最后写回，client 通过轮询 status word 判断完成。

当前 `src/ps/rdma/rc_transport.*` 已经不是 `shm_open + mmap` baseline，而是直接复用 `raw_verbs_transport.*` 的真实 verbs RC 写入路径。`RawVerbsTransport` 负责设备打开、MR 注册、QP 建立、memcached metadata 交换和 RDMA write completion 轮询；`rc_transport.*` 负责固定 slot 布局、shard/client/lane offset 计算和 request/response 提交流程。

PetPS server、PetPS client 和 op-layer RDMA 目前不再暴露可切换的 transport mode 配置。旧文档或旧路径里的 `raw_message` / `RECSTORE_RDMA_TRANSPORT_MODE` 不代表当前 PetPS RC write transport。

## 构建

常用 RDMA 目标：

```bash
cmake --build ./build --target \
  ps_transport_benchmark \
  petps_server \
  petps_integration_test \
  recstore_torch_ops \
  test_allshards_ps_client \
  -j
```

如果刚改过 `src/ps/rdma/*`、`src/test/scripts/*rdma*` 或 op-layer 相关代码，先重编对应目标再判断行为。旧的 `petps_server` / `recstore_torch_ops` 二进制很容易造成“源码已改但测试仍卡住”的假象。

## Benchmark

推荐先跑 PetPS RC-write correctness 基线，再做 benchmark：

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

summary 表中的 `put_v2` 列用于确认 PUT-v2 payload transfer mode；`read` 和 `push` 的结果不能直接混比。benchmark 仍然需要按同口径参数解释，不能只凭控制面差异推断收益。

真实 RDMA 路径通常会输出类似：

```text
I open mlx5_0 :)
I connect server 0
transport=RDMA op=put phase=measure ...
transport=RDMA op=get phase=measure ...
```

当前 RC write slot path 的 correctness 验证优先级高于 benchmark。性能数据需要单独记录硬件、QP 数、batch size、value size 和 server poll thread 数。

## PetPS Integration

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

多分片排障时优先检查 `distributed_client.num_shards`、`distributed_client.servers`、`server-count`、`num_server_processes` 和 key 到 shard 的路由是否一致。

最近一次真实 verbs 验证结果：

- `PutGetRoundTripSingleShard` 通过。
- `UpdateGetRoundTripSingleShard` 通过。
- `PutGetRoundTripMultiShard` 通过。

## Op-layer RDMA

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

最近一次 op-layer RDMA 验证结果：

- `test_op_runtime_support` 通过。
- `test_op` 通过。
- `pytorch_client_test_rdma_basic` 通过，覆盖 PyTorch custom op 到 RDMA backend 的 init/write/read。
- `pytorch_client_test_rdma` 通过，覆盖 PyTorch custom op 到 `RDMAPSClientAdapter -> PetPSClient -> verbs RC` 的 init/write/read、prefetch 和 table-aware update roundtrip。

## Unit 和脚本测试

协议 helper 和 wrapper：

```bash
ctest --test-dir ./build -R "^test_allshards_ps_client$" -VV
```

runner 参数拼接：

```bash
python3 -m unittest src/test/scripts/test_petps_cluster_runner.py
python3 -m unittest src/test/scripts/test_run_rdma_transport_benchmarks.py
```

这些测试不证明 RDMA 数据面可用，只证明协议编码、分片 wrapper 和脚本 plumbing 没有明显回归。

## memcached

RDMA 脚本通过 memcached 交换 Mayfly/DSM 元数据。推荐使用：

```bash
--use-local-memcached auto
```

| 值 | 行为 |
|----|------|
| `auto` | 优先复用外部 memcached；不可用时启动本地 memcached |
| `always` | 总是启动本地 memcached |
| `never` | 只使用已经存在的外部 memcached |

手工启动：

```bash
memcached -u root -l 127.0.0.1 -p 21211 -c 10000
```

检查端口：

```bash
ss -ltnp | grep ':21211'
```

## 排障顺序

1. 确认二进制是最新构建的目标，尤其是 `petps_server`、`ps_transport_benchmark`、`petps_integration_test` 和 `recstore_torch_ops`。
2. 确认参数传到了正确入口：runner 用 RDMA 专项参数，C++ binary 读对应 gflags，op-layer 读 `RECSTORE_CONFIG` 和相关环境变量。
3. 确认 RDMA 设备和 memcached 可用：检查 `/dev/infiniband`、`ibv_devices`、`ss -ltnp | grep ':21211'`。
4. 如果 runner 卡在 `memcached-wait` 或 `startup-wait`，先看 runner 捕获的 server 日志。
5. 如果看到 `unknown command line flag 'rdma_transport_mode'`，通常是跑到了旧 binary，或者当前目标没有链接 RDMA client 相关对象。
6. 如果看到 `messeage size too large`，先把 `batch-keys` 降到 500 或更小建立稳定基线。
7. 如果 RDMA 路径卡住，重点检查 raw verbs buffer 是否注册、QP metadata 是否按 shard/lane 匹配、CQ 是否被错误线程消费、server/client mode 是否一致。

最小日常验证顺序：

```bash
cmake --build ./build --target ps_transport_benchmark petps_server recstore_torch_ops -j
python3 src/test/scripts/run_petps_integration.py \
  --server-count 1 \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.UpdateGetRoundTripSingleShard \
  --use-local-memcached=auto
ctest --test-dir ./build -R "^pytorch_client_test_rdma_basic$" -VV
```
