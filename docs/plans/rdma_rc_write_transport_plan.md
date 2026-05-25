# RDMA RC Write Transport 实现方案

## 当前落地状态

截至当前仓库实现，`src/ps/rdma` 已经完成一版真实 verbs RC write transport MVP。当前重要边界如下：

1. 这一版没有复用旧的 Mayfly `RawMessage` 数据面，也没有继续沿用旧的高层 `verbs/qp` 数据面封装。
2. `rc_transport.*` 已经从早期 `shm_open + mmap` correctness baseline 切换到复用 `raw_verbs_transport.*` 的真实 verbs RC 写入路径。
3. 协议仍保持固定 slot 语义：client 写 `RequestDescriptor + payload`，再写 `CommitWord`；server 写 response payload，再写 `StatusWord`。
4. 当前仍是 correctness-first 实现：每条 lane 单 in-flight，提交和完成路径等待本地 send completion，还没有做深度 async overlap。

当前已完成：

- `RequestDescriptor` / `CommitWord` / `StatusWord` 协议定义。
- 基于 `RawVerbsTransport` 的 request/response slot transport。
- 单 shard `GET` / `PUT` / `UPDATE` 数据面。
- 多 shard wrapper 和显式 `servers[].shard` 路由。
- `PetPSClient` / `petps_server` / `RDMAPSClientAdapter` 接入。
- 单元测试、单 shard 集成测试、多 shard 集成测试、op-layer RDMA 测试。

当前未完成：

- 更强的 async/prefetch 语义。
- 控制面和运维指标补齐。
- 性能 benchmark、QP 数和 server poll thread 调优。

### 当前阶段目标

当前阶段目标已经从早期 shm baseline 推进为：

1. 固定 `RC-write-like` 协议、slot、seq、status 和显式 shard 路由语义。
2. 用真实 verbs RC 替换 `shm_open + mmap` transport，且不改变协议和测试语义。
3. 在真实 verbs RC 上覆盖：
   - 单 shard roundtrip
   - 多 shard roundtrip
   - adapter 基本接线
   - `Update -> Get` roundtrip
   - op-layer init/write/read/prefetch/update roundtrip

这意味着当前 `src/ps/rdma` 的阶段产物应被视为：

```text
Phase 2: RC protocol baseline on real verbs RC transport
```

早期的 shm baseline 只用于固定协议语义，当前不再是主实现。

### 下一阶段目标

下一阶段目标是在不改协议和测试语义的前提下，提高真实 verbs RC 路径的工程完整度：

- 补齐性能 benchmark 和调参记录。
- 降低同步等待带来的开销，逐步实现真正的 async overlap。
- 补充控制面 ready/health/debug metadata。
- 增加更明确的错误日志，包括 shard、client、lane、seq、op、remote node。
- 评估多 client、多 shard、多 lane 下的 QP 资源预算。

### 当前代码范围

本轮已经实际改写或新增的核心文件：

- `src/ps/rdma/rdma_protocol.h`
- `src/ps/rdma/rc_transport.h`
- `src/ps/rdma/rc_transport.cc`
- `src/ps/rdma/rc_options.h`
- `src/ps/rdma/rc_options.cc`
- `src/ps/rdma/petps_client.h`
- `src/ps/rdma/petps_client.cc`
- `src/ps/rdma/allshards_ps_client.h`
- `src/ps/rdma/allshards_ps_client.cc`
- `src/ps/rdma/petps_server.cc`
- `src/ps/rdma/rdma_ps_client_adapter.cc`
- `src/ps/rdma/CMakeLists.txt`
- `src/test/test_rdma_rc_protocol.cpp`
- `src/test/test_allshards_ps_client.cpp`
- `src/test/ps/rdma/petps_integration_test.cpp`
- `src/test/CMakeLists.txt`

### 当前实现语义

当前版本的 client/server 语义已经固定为：

- client 通过 RDMA write 向 shard-local server `request slot` 写入 `RequestDescriptor + payload`
- client 最后通过 RDMA write 写 `CommitWord`
- server 轮询 request slot，按 `seq + READY` 发现新请求
- server 执行 `CachePS::GetParameterFlat`、`PutSingleParameter` 或 table-aware `UpdateParameter`
- server 通过 RDMA write 向 client `response slot` 写入 response payload
- server 最后通过 RDMA write 写 `StatusWord`
- client 轮询 `StatusWord` 判断完成

当前版本已经满足：

- 单 shard `PUT -> GET` round trip
- 多 shard `PUT -> GET` round trip
- 单 shard `UPDATE -> GET` round trip
- op-layer RDMA init/write/read/prefetch/update round trip
- missing key 返回零值
- 显式 `servers[].shard` 路由，不依赖 `servers` 数组顺序
- 单 lane 单 in-flight；资源耗尽时立即失败，而不是静默阻塞
- `UpdateParameter` 语义固定为 table-aware update，而不是本地 `Get + Put` 退化路径

### 本轮修复记录

在当前 MVP 基础上，本轮额外完成：

1. `rc_transport.*` 从 shm slot transport 切换到 `RawVerbsTransport` backed 的真实 verbs RC slot transport。
2. client submit 边界改为 `RcShardClientTransport::SubmitRequest`，由 transport 完成 descriptor/payload 和 commit 的 RDMA write。
3. server complete 边界改为 `RcShardServerTransport::CompleteResponse`，由 transport 完成 response payload 和 status 的 RDMA write。
4. client 侧 registered region 按 shard 分段，避免多 shard client response slot 覆盖。
5. `RawVerbsConfig` 增加 `only_node_id` 过滤，让每个 shard-local client lane 只连接目标 server，避免多 shard metadata 冲突。
6. `petps_server` 复用 `Postoffice` 已定义的 `global_id` / `num_server_processes` / `num_client_processes` gflags，避免链接 `RawVerbsTransport` 后重复定义。

早期已修复的同步等待重入锁问题仍然保持有效，当前同步 `PUT/GET/UPDATE` 路径可正常完成。

### 当前验证结果

已完成并通过的验证：

```text
ctest --test-dir build -R "test_rdma_rc_protocol|test_allshards_ps_client" -VV
```

```text
python3 src/test/scripts/run_petps_integration.py \
  --server-count 1 \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.MissingKeysReturnZeroSlots:PetPSIntegrationTest.ExhaustedQpPoolFailsLoudly \
  --use-local-memcached=auto \
  --show-runner-logs \
  --client-timeout=12 \
  --cluster-timeout=25
```

```text
python3 src/test/scripts/run_petps_integration.py \
  --server-count 2 \
  --config-path ./src/test/configs/recstore_config.rdma_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripMultiShard \
  --use-local-memcached=auto \
  --show-runner-logs \
  --client-timeout=12 \
  --cluster-timeout=25
```

当前新增测试覆盖：

- 协议 budget/对齐/seq 判定
- 多 shard 显式 shard id 路由
- 单 shard `PUT/GET`
- missing key
- 多 shard `PUT/GET`
- QP 池耗尽时 fail loudly

本轮真实 verbs 和 op-layer 额外验证：

```text
cmake --build build --target petps_integration_test -j4
```

```text
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

结果：`PutGetRoundTripSingleShard` 和 `UpdateGetRoundTripSingleShard` 通过。

```text
python3 src/test/scripts/run_petps_integration.py \
  --server-count 2 \
  --config-path ./src/test/configs/recstore_config.rdma_multishard_test.json \
  --test-binary ./build/bin/petps_integration_test \
  --gtest-filter=PetPSIntegrationTest.PutGetRoundTripMultiShard \
  --use-local-memcached=auto \
  --show-runner-logs \
  --client-timeout=25 \
  --cluster-timeout=45
```

结果：`PutGetRoundTripMultiShard` 通过。

```text
cmake --build build --target test_op test_op_runtime_support recstore_torch_ops -j4
ctest --test-dir build -R "test_op_runtime_support|test_op$" -VV
ctest --test-dir build -R "pytorch_client_test_rdma_basic" -VV
ctest --test-dir build -R "pytorch_client_test_rdma$" -VV
```

结果：

- `test_op_runtime_support` 通过。
- `test_op` 通过。
- `pytorch_client_test_rdma_basic` 通过。
- `pytorch_client_test_rdma` 通过，覆盖 op-layer 到 `RDMAPSClientAdapter -> PetPSClient -> verbs RC` 的 init/write/read、prefetch 和 table-aware update roundtrip。

### 当前已知边界

当前实现仍然有这些明确边界：

- `GetParameter` 的 shard-local 单个 RPC 仍受 response budget 约束，超出时依赖 wrapper 分片/分批
- `PutParameter` 当前仍按 correctness-first 路径逐批同步提交
- `Barrier` 仍是 no-op
- 还没有真正的异步 overlap 保证
- 当前 correctness 路径等待每次 RDMA write 的本地 send completion，吞吐还没有调优
- 多 client / 高 QP 数 / 长时间压力测试尚未覆盖

### 下一步建议

如果继续推进，推荐顺序是：

1. 保持当前 verbs RC 协议和 correctness 测试不变，补 multi-client 和更长时间压力测试。
2. 增加 benchmark 记录，明确 `qps_per_client_per_shard`、value size、batch size、poll thread 数对吞吐和延迟的影响。
3. 优化 write completion 策略，减少每个 RPC 固定两次同步等待。
4. 补控制面 ready/health 和更清晰的错误日志。
5. 再评估是否需要 tensor registered buffer 或真正的 async/future API。

以下计划保留目标态设计，阶段状态以当前实现为准。

## 背景

当前 `src/ps/rdma` 仍以 Mayfly `RawMessage` 为主要请求通道。该路径受 `MESSAGE_SIZE=4096` 约束，PUT/GET 的 batch size 很容易被 MTU 级消息大小限制，无法支撑更大的训练 batch 吞吐。

本方案目标是抛弃 Mayfly RawMessage/UD 请求语义，改为 RC write 模型下的固定 slot 请求/响应协议。gRPC/bRPC 版本只作为参数服务器语义参考，不作为网络实现参考。

## 目标

- 使用 shard-local RC write transport 替代 Mayfly RawMessage。
- 请求和响应均使用 descriptor/payload + commit/status 的单边写入模型。
- 第一版采用一个 transport lane 同一时间只承载一个 in-flight RPC。
- 每个 lane 固定绑定一个 server request slot 和一个 client response slot。
- server 不依赖 recv CQ 和 SEND/WRITE_WITH_IMM doorbell；server 通过扫描 request slot 的 commit word 发现请求。
- 按 response budget 控制 GET batch size，避免返回 payload 过大成为主要瓶颈。
- 第一版优先实现 `GetParameter` 和 `PutParameter`，为后续 `UpdateParameter` 预留协议字段。

## 非目标

- 不兼容 Mayfly RawMessage 数据面。
- 不做每 RPC 创建/销毁连接上下文。
- 不在一个 lane 内维护多个 outstanding RPC。
- 不引入 SEND/WRITE_WITH_IMM 通知语义。
- 不做 memcached key-position 缓存。
- 不追求第一版 tensor 直接注册零拷贝；允许 client 复制到 registered staging buffer。
- 不优先支持完整 async/future API。

## 现有代码依据

- `FLAGS_value_size` 全局默认是 128B，但主配置和 op RDMA 配置通常使用 `default_value_size_hint=512B`。
- 当前 RDMA GET response 近似为 `key_count * value_size + sizeof(int32_t)`。
- 当前 GET request 是连续 `uint64_t keys`。
- gRPC/bRPC PUT/UPDATE 使用 `ParameterCompressReader` 语义。
- 当前 Mayfly RawMessage 路径会因为 `MESSAGE_SIZE` 限制大 batch，不再作为新方案基础。

## 第一版参数

默认按主配置 `value_size=512B` 设计。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `mtu_bytes` | 4096 | 计算 batch 预算使用的逻辑 MTU |
| `target_response_mtu` | 200 | GET response 目标大小 |
| `response_budget_bytes` | 819200 | `mtu_bytes * target_response_mtu` |
| `max_get_keys_per_rpc` | 1600 | `response_budget_bytes / value_size` |
| `target_request_mtu` | 200 | PUT request 目标大小 |
| `max_put_keys_per_rpc` | 约 1550-1575 | 取决于使用 dense payload 还是 `ParameterCompressReader` |
| `qps_per_client_per_shard` | 32 | 第一版建议值，需 benchmark 调整 |
| `request_slot_bytes` | 1 MiB | 覆盖 200 MTU 级 PUT/GET request |
| `response_slot_bytes` | 1 MiB | 覆盖 200 MTU 级 GET response |
| `poll_threads` | 配置化 | server 扫描 slot 的线程数 |

对于 `value_size=512B`：

```text
GET keys/request = floor(819200 / 512) = 1600
GET request bytes = 1600 * 8 = 12800B
GET response bytes = 1600 * 512 = 819200B
```

如果 `value_size` 改变，`max_get_keys_per_rpc` 必须动态重算：

```text
max_get_keys_per_rpc = floor(target_response_mtu * mtu_bytes / value_size)
```

## 总体架构

```text
Distributed client / op-layer adapter
  -> partition keys by distributed_client hash semantics
  -> submit one shard-local RPC per shard chunk

Shard-local client thread
  -> choose shard
  -> acquire idle QP context
  -> write request descriptor + payload to server request slot
  -> write commit word last
  -> wait local response status word
  -> release QP context

Shard-local server poll thread
  -> scan assigned request slots
  -> detect new commit seq
  -> parse descriptor and payload from local registered memory
  -> call CachePS
  -> RDMA_WRITE response body to client response slot
  -> RDMA_WRITE response status word last
  -> mark slot consumed locally
```

## 分片和分布式语义

新 RC path 必须保留现有 gRPC/bRPC 和 RDMA wrapper 的分片语义。不能假设 `shard == servers 数组下标`，也不能用单 shard 设计覆盖多 shard 场景。

### 配置来源

client 侧必须从 `distributed_client` 读取：

- `num_shards`
- `hash_method`
- `servers`
- `max_keys_per_request`，或 RC 专用 batch 参数

server 侧必须从 `cache_ps` 读取：

- `num_shards`
- `servers`
- 当前 server 的 `shard`
- 当前 shard 的 `base_kv_config`

如果 RC 专用配置缺失，允许 field-by-field 回退到现有字段，但不能整块替换配置。例如：

```text
num_shards: distributed_client.num_shards
servers: distributed_client.servers
hash_method: distributed_client.hash_method
max_keys_per_request: min(existing max_keys_per_request, rc computed max keys)
```

### 路由规则

必须复用现有 `distributed_client` 的 hash 语义。历史教训是：不要假设 `shard == sorted_index`。推荐实现：

1. 根据 `hash_method` 和 `num_shards` 计算 logical shard id。
2. 通过 `servers[].shard` 找到对应 server endpoint。
3. 每个 shard endpoint 维护独立 RC QP pool。
4. 每个 shard chunk 单独提交 shard-local RPC。
5. GET 返回后按原始 key 位置 merge 回用户输出 buffer。

### 多 shard GET

```text
input keys[0..N)
  -> PartitionKeys(keys) produces shard chunks:
       shard_id, keys_for_shard, original_positions
  -> each shard chunk split by rc max_get_keys_per_rpc
  -> submit to that shard's QP pool
  -> wait all shard-local RPCs
  -> copy each shard response row back to original_positions
```

GET response 是 dense rows，shard-local response 内部顺序必须等于 shard-local request key 顺序。跨 shard merge 由 wrapper 根据 `original_positions` 完成。

### 多 shard PUT / UPDATE

PUT/UPDATE 同样先按 shard 拆分：

```text
keys + values/grads
  -> partition by shard
  -> preserve per-shard row order
  -> build shard-local payload
  -> submit per-shard RPC
  -> any shard failure means global operation failure
```

PUT/UPDATE 不需要按原始顺序回填 response，但错误状态必须带上 shard id、seq 和 op，方便定位失败 shard。

### QP pool 与 shard 的关系

QP pool 是 shard-local 的：

```text
client process
  shard 0 -> qps_per_client_per_shard QPs
  shard 1 -> qps_per_client_per_shard QPs
  ...
```

多 shard 情况下总 QP 数：

```text
num_shards * qps_per_client_per_shard * client_threads_or_processes
```

实现时必须限制总 QP 数，避免多 shard 配置下资源爆炸。

## QP 模型

每个 client 到每个 server shard 建立一个 QP pool。

第一版约束：

- 一个 QP 同一时间最多一个 in-flight RPC。
- 一个 QP 固定对应一个 server request slot。
- 一个 QP 固定对应一个 client response slot。
- `QP busy` 等价于 `RPC in flight`。
- QP 归属固定线程，避免多个线程同时对同一个 QP post send。

建议数据结构：

```cpp
struct RcQpContext {
  int qp_index;
  ibv_qp* qp;
  ibv_cq* send_cq;
  RequestSlotRemote server_request_slot;
  ResponseSlotLocal client_response_slot;
  uint64_t next_seq;
  bool busy;
};
```

## 内存布局

### Server Request Slot

每个 QP 固定一个 server request slot。

```text
RequestSlot
  RequestDescriptor descriptor
  uint8_t payload[request_slot_payload_bytes]
  padding to cacheline
  CommitWord commit
```

`commit` 必须单独 cacheline 对齐，避免 server 扫描时反复拉取大 payload 所在 cacheline。

### Client Response Slot

每个 QP 固定一个 client response slot。

```text
ResponseSlot
  uint8_t response[response_slot_payload_bytes]
  padding to cacheline
  StatusWord status
```

server 必须先写 response body，再写 status word。client 只以 status word 判断完成。

## 协议结构

### RequestDescriptor

第一版建议固定大小并 cacheline 对齐。

```cpp
enum class RcOp : uint16_t {
  kGet = 1,
  kPut = 2,
  kUpdate = 3,
};

struct RequestDescriptor {
  uint32_t magic;
  uint16_t version;
  uint16_t op;
  uint64_t seq;
  uint32_t key_count;
  uint32_t value_size;
  uint32_t embedding_dim;
  uint32_t payload_offset;
  uint32_t payload_bytes;
  uint64_t client_response_addr;
  uint32_t client_response_rkey;
  uint32_t client_response_bytes;
  uint64_t client_status_addr;
  uint32_t client_status_rkey;
  uint32_t flags;
  char table_name[64];
};
```

### CommitWord

client 最后写入。

```cpp
struct CommitWord {
  uint64_t seq;
  uint32_t state;  // READY
  uint32_t checksum_or_reserved;
};
```

### StatusWord

server 最后写回。

```cpp
struct StatusWord {
  uint64_t seq;
  uint32_t state;  // DONE
  int32_t status;  // RpcStatus
  uint32_t response_bytes;
  uint32_t reserved;
};
```

## 写入顺序和内存可见性

client 提交顺序：

```text
1. 清空本地 response status 为 PENDING(seq)。
2. 准备 request descriptor 和 payload。
3. RDMA_WRITE descriptor + payload 到 server request slot。
4. 等待第 3 步本地 send completion。
5. RDMA_WRITE commit word 到 server request slot。
6. 等待 commit write 本地 send completion。
7. 轮询 client response status。
```

server 处理顺序：

```text
1. load_acquire commit word。
2. commit.seq != last_seq 且 state == READY 时读取 descriptor。
3. 校验 magic/version/op/value_size/payload_bytes/slot capacity。
4. 调用 CachePS。
5. RDMA_WRITE response body 到 client response slot。
6. 等待 response body write 本地 completion。
7. RDMA_WRITE status word 到 client status addr。
8. 等待 status write 本地 completion。
9. 更新 last_seq。
```

关键约束：

- commit word 必须最后写。
- status word 必须最后写。
- 每个 QP 一个 in-flight RPC，避免同一 slot 被覆盖。
- seq 单调递增，避免 client/server 误读旧状态。

## GET 流程

### Client

1. 按 shard 路由 keys。
2. 按 `max_get_keys_per_rpc` 切分。
3. 从目标 shard QP pool 获取 idle QP。
4. 将连续 `uint64_t keys` 写入 request slot payload。
5. 填写 descriptor：
   - `op = kGet`
   - `key_count`
   - `payload_bytes = key_count * sizeof(uint64_t)`
   - `client_response_addr/rkey`
   - `client_status_addr/rkey`
6. RDMA_WRITE descriptor + payload。
7. RDMA_WRITE commit。
8. 等待 response status。
9. 将 dense response 拷贝到用户输出 buffer。

### Server

1. poller 发现 READY commit。
2. 将 payload 解释为连续 `uint64_t keys`。
3. 调用 `CachePS::GetParameterFlat` 或等价 flat get 路径。
4. 将 dense float response 写回 client response slot。
5. 写回 status。

GET response 格式：

```text
float values[key_count][embedding_dim]
StatusWord status
```

## PUT 流程

第一版建议沿用 `ParameterCompressReader` payload，减少与 gRPC/bRPC PUT 语义差异。

### Client

1. 校验 `keys.size() == values.size()`。
2. 按 `max_put_keys_per_rpc` 切分。
3. 使用 `ParameterCompressor` 构造 payload 到 registered staging buffer。
4. 填写 descriptor：
   - `op = kPut`
   - `key_count`
   - `payload_bytes`
5. RDMA_WRITE descriptor + payload。
6. RDMA_WRITE commit。
7. 等待 status。

### Server

1. poller 发现 READY commit。
2. 将 payload 解释为 `ParameterCompressReader`。
3. 调用 `reader->Valid(payload_bytes)`。
4. 循环 `cache_ps_->PutSingleParameter(reader->item(i), tid)`。
5. 写回 status。

PUT 第一版不需要 response body，只需要 status。

## UPDATE 流程

UPDATE 可以复用 PUT payload 格式：

- descriptor `op = kUpdate`
- `table_name` 必填
- payload 为 `ParameterCompressReader`
- server 调用 `cache_ps_->UpdateParameter(table_name, reader, tid)`

当前实现已经接入 table-aware update，并通过单 shard integration 和 op-layer roundtrip 验证。

## Server Poller 分配

server 启动时将所有 request slots 分配给 poll threads。

```text
poll_thread_id = slot_index % poll_threads
```

第一版 poller 可以直接处理请求，不额外引入 worker queue。后续如果发现 `CachePS` 调用阻塞扫描，可改成：

```text
poller -> lock-free queue -> worker -> RDMA_WRITE response
```

但第一版不建议增加该复杂度。

## Server shard 生命周期

RC server 必须支持现有两类部署形态：

- 单进程单 shard。
- 多 shard 配置下，每个 shard 一个 server endpoint 或一个进程内多 shard 服务。

第一版建议优先做单进程单 shard，与当前 `petps_server` 测试路径对齐；但协议和 client wrapper 必须按多 shard 设计，不要在接口中写死 shard 0。

每个 shard server 启动流程：

```text
1. 读取 config。
2. 确定本 server 的 explicit shard id。
3. 构造 shard-local CachePS。
4. 初始化 RC listener/control metadata。
5. 为每个 client/QP 分配 request slots。
6. 建立 RC QP。
7. 发布 shard ready。
8. 启动 poll threads。
```

ready key / control metadata 也必须包含 shard id，避免多 shard 时互相覆盖：

```text
rc-ready-shard-{shard_id}
rc-qp-meta-shard-{shard_id}-client-{client_id}-qp-{qp_index}
```

## 内存预算

以 `value_size=512B`、`qps_per_client_per_shard=32`、单 client 单 shard为例：

```text
request region  = 32 * 1 MiB = 32 MiB
response region = 32 * 1 MiB = 32 MiB
```

如果 8 clients、单 shard：

```text
server request region = 8 * 32 * 1 MiB = 256 MiB
client response region per client = 32 MiB
```

如果将 response slot 提高到 2 MiB，则上述内存翻倍。第一版应避免 50 MiB 级 response slot。

## 文件级实现结果

当前已新增或重写以下文件：

| 文件 | 作用 |
|---|---|
| `src/ps/rdma/rc_transport.h` | request/response slot transport 封装 |
| `src/ps/rdma/rc_transport.cc` | 基于 `RawVerbsTransport` 的真实 verbs RC slot transport 实现 |
| `src/ps/rdma/rdma_protocol.h` | descriptor、commit、status、常量和 size helper |
| `src/ps/rdma/raw_verbs_transport.h/.cc` | verbs 设备、MR、QP、metadata 交换和 RDMA read/write primitive |
| `src/ps/rdma/petps_client.h` | 单 shard client |
| `src/ps/rdma/petps_client.cc` | Get/Put/Update submit/wait 实现 |
| `src/ps/rdma/allshards_ps_client.h` | 多 shard wrapper |
| `src/ps/rdma/allshards_ps_client.cc` | PartitionKeys、并发 shard submit、GET merge |
| `src/ps/rdma/petps_server.cc` | slot 扫描、CachePS 调用、response write |
| `src/ps/rdma/rdma_ps_client_adapter.cc` | 增加 RC backend 初始化和配置解析 |
| `src/ps/rdma/CMakeLists.txt` | 新目标接入 |
| `src/test/ps/rdma/*` | 协议 helper 和单机集成测试 |

保留复用：

- `rdma_status.h` 中已有状态码。
- `AllShardsParameterClientWrapper` 的 shard 拆分思路，但已经改成显式 shard id 映射，避免假设 `shard == client index`。

## 配置项

建议新增 RDMA RC 专用配置或 gflags：

```text
--rdma_rc_qps_per_client_per_shard=32
--rdma_rc_mtu_bytes=4096
--rdma_rc_target_response_mtu=200
--rdma_rc_target_request_mtu=200
--rdma_rc_request_slot_bytes=1048576
--rdma_rc_response_slot_bytes=1048576
--rdma_rc_poll_threads=8
--rdma_rc_wait_timeout_ms=60000
```

启动时校验：

```text
request_slot_bytes >= sizeof(RequestDescriptor) + max_put_payload_bytes + sizeof(CommitWord)
response_slot_bytes >= max_get_keys_per_rpc * value_size + sizeof(StatusWord)
```

## 分阶段实施

本计划按可交付能力分阶段。每一阶段都应能独立验证，不把功能等价、异步能力和性能调优混在同一个 patch 中。

### 阶段 0：边界确认和配置对齐

状态：已完成。

目标：确认 RC path 的配置、shard 语义和 `value_size` 来源，避免后续实现出现 server/client 不一致。

- 明确 `ps_type` 或环境开关，例如 `RDMA_RC` / `RECSTORE_RDMA_RC_WRITE=1`。
- 明确 client 从 `distributed_client` 读取 shard 路由配置。
- 明确 server 从 `cache_ps.servers[].shard` 或进程参数确定 explicit shard id。
- 明确 `value_size` 的优先级：
  - RC 专用配置。
  - `cache_ps.base_kv_config.value.default_value_size_hint`。
  - `FLAGS_value_size`。
- 明确 `max_get_keys_per_rpc` 由 response budget 动态计算。
- 明确旧 Mayfly RawMessage path 不再作为新实现依赖。

验收：

- 单元测试覆盖 `value_size=512B` 时 `max_get_keys_per_rpc=1600`。
- 多 shard 配置中 `servers` 顺序打乱时仍能解析 explicit shard id。

### 阶段 1：协议和纯内存测试

状态：已完成。

- 新增 `rdma_protocol.h`。
- 添加 size helper：
  - `GetKeysPerRpcByResponseBudget(value_size, mtu, response_mtu)`
  - `GetRequestBytes(key_count)`
  - `GetResponseBytes(key_count, value_size)`
  - `PutPayloadBudget(...)`
- 单元测试覆盖 512B value 下 1600 keys 计算。
- 单元测试覆盖 descriptor、commit、status 的对齐和大小。
- 单元测试覆盖 seq 递增、旧 seq 不重复消费。

### 阶段 2：RC transport smoke

状态：已完成，当前 transport 是真实 verbs RC slot transport。

- 建立 client/server shard-local slot region。
- client 提交 fixed slot request 到 server。
- server 扫描 commit。
- server 回写 response/status 到 client response slot。
- 不接 CachePS。
- 验证一个 QP 同一时间只能提交一个 RPC。
- 验证 descriptor/payload 发布后再写 commit。
- 验证 response body 发布后再写 status。

### 阶段 3：MVP 数据面 GET

状态：已完成。

- client 写 keys payload。
- server 调 `GetParameterFlat`。
- server 写 dense response。
- 验证 1、1600、超过 1600 自动切分。
- 验证 response bytes 不超过 `response_slot_bytes`。
- 验证 `value_size=512B` 下 1600 keys response 为 819200B。
- 验证 flat `float*` 输出接口兼容 `RDMAPSClientAdapter`。

### 阶段 4：MVP 数据面 PUT

状态：已完成。

- client 构造 `ParameterCompressReader` payload。
- server 校验并调用 `PutSingleParameter`。
- PUT 后 GET 验证回读正确。
- 验证 PUT payload 超过 `request_slot_bytes` 时 client 自动切分。
- 验证 shard-local PUT 失败会向上返回错误。

### 阶段 5：MVP 多 QP 和多 shard

状态：已完成第一版。

- QP pool 并发。
- 单 QP 单 in-flight。
- shard wrapper 适配。
- 验证 `distributed_client.hash_method`、`num_shards`、`servers[].shard` 路由一致。
- 验证不同 shard 的 key 路由和结果合并。
- 验证 `servers` 数组顺序打乱时仍按 explicit shard id 路由。
- 验证多 shard GET 按原始 key 顺序回填。
- 验证多 shard PUT 任一 shard 失败时整体失败。

完成阶段 5 后，MVP 交付范围为：

- RC WRITE transport。
- GET 数据面。
- PUT 数据面。
- UPDATE 数据面。
- shard wrapper。
- flat buffer 兼容路径。
- response-budget driven batch 切分。

当前 MVP 不包含：

- `Command` 控制面。
- 真正的 async overlap。
- metrics/reflection/复杂运维能力。

### 阶段 6：MVP benchmark

状态：未完成。

- 对比：
  - Mayfly RawMessage RDMA old path
  - gRPC/bRPC
  - RC WRITE path
- 分别测：
  - GET 1600 keys
  - PUT 约 1550 keys
  - QP 数 1/4/8/16/32
  - poll threads 1/2/4/8
- response budget 200/400/800 MTU
- 单 shard 和多 shard 场景

### 阶段 7：MVP+ UpdateParameter

状态：已完成第一版。

目标：接入训练更新数据面，但不改变 MVP 的 QP/slot/scan 模型。

- 复用 PUT 的 `ParameterCompressReader` payload。
- descriptor `op = kUpdate`。
- descriptor `table_name` 必填，或后续优化为 table id。
- server 调用 `cache_ps_->UpdateParameter(table_name, reader, tid)`。
- 多 shard UPDATE 按 key 拆分 gradients，保持 row 对齐。
- 任一 shard UPDATE 失败时整体返回失败。

验收：

- 单 shard UPDATE 后 GET 结果符合 optimizer 语义。
- 多 shard UPDATE 路由和 row 顺序正确。
- malformed `ParameterCompressReader` 返回错误，不崩溃 server。

### 阶段 8：控制面补齐

状态：未完成。

目标：补齐功能等价所需的非热点控制接口，但不阻塞 RC 数据面 MVP。

优先级：

1. `Command(CLEAR_PS)`
2. `Command(RELOAD_PS)`
3. `LOAD_FAKE_DATA` / `DUMP_FAKE_DATA` benchmark 辅助命令

`InitEmbeddingTable` 已作为 RC control op 接入当前路径，并被 update roundtrip 和 op-layer RDMA 测试覆盖。

建议实现策略：

- 第一选择：继续复用现有 gRPC/bRPC/control path。
- 第二选择：在 RC protocol 中增加 small control op。

不要在阶段 8 之前为了控制面能力改变 RC GET/PUT/UPDATE 数据面。

### 阶段 9：Async / Prefetch

状态：未完成。

目标：在同步路径正确后，再提供可等待句柄和 pipeline overlap。

约束：

- 不复用 backend-local handle 作为全局 handle。
- wrapper 必须生成自己的 opaque handle。
- 一个 QP 单 in-flight 的限制如果要放宽，必须重新设计 slot/ring 和 completion ownership。

第一步建议只做：

```text
submit -> returns opaque handle
wait(handle) -> blocks until status done
consume(handle) -> copies response and releases resource
```

不在第一步承诺真实 overlap 性能。

### 阶段 10：运维、指标和调试

状态：部分完成。

- QP pool 资源统计。
- slot 扫描命中率。
- response bytes histogram。
- timeout 和 seq mismatch 日志。
- shard id、client id、qp index、seq 的错误定位日志。
- benchmark summary 输出。
- 健康检查和启动自检。

## 验证计划

当前最小测试：

```text
ctest --test-dir ./build -R "rc_protocol|rc_transport" -VV
```

当前集成测试：

```text
python3 src/test/scripts/run_petps_integration.py --server-count 1 --config-path ./src/test/configs/recstore_config.rdma_test.json --test-binary ./build/bin/petps_integration_test --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.MissingKeysReturnZeroSlots --use-local-memcached=auto --show-runner-logs --client-timeout=12 --cluster-timeout=25

python3 src/test/scripts/run_petps_integration.py --server-count 2 --config-path ./src/test/configs/recstore_config.rdma_test.json --test-binary ./build/bin/petps_integration_test --gtest-filter=PetPSIntegrationTest.PutGetRoundTripMultiShard:PetPSIntegrationTest.ExhaustedQpPoolFailsLoudly --use-local-memcached=auto --show-runner-logs --client-timeout=12 --cluster-timeout=25
```

手工 smoke：

```text
server: petps_server --value_size=512 --rdma_rc_qps_per_client_per_shard=32
client: petps_integration_test --gtest_filter=PetPSIntegrationTest.PutGetRoundTripSingleShard
```

验收条件：

- 1600 keys GET response 正确。
- 1600 keys GET response bytes 为 819200B。
- PUT 后 GET 数据一致。
- 超过 `max_get_keys_per_rpc` 时 client 自动切分。
- 单 lane 不允许重复 submit，必须 fail loudly。
- server slot 处理 seq 单调递增，不重复处理旧请求。
- 多 shard 下按 `servers[].shard` 路由，不依赖 servers 数组下标。
- 多 shard GET 按原始 key 顺序回填输出。
- 任一 shard PUT/UPDATE 失败时 global operation 返回失败。

## 风险和处理

### Server 扫描开销

风险：QP 数过多时 poller 扫描所有 slot 会消耗 CPU。

处理：

- slot 按 poll thread 分片。
- commit word cacheline 对齐。
- 从 32 QP 起步，不直接上百上千。
- benchmark 后决定是否需要 active list 或 doorbell。

### Response 仍是瓶颈

风险：即使限制到 200 MTU response，server->client bandwidth 仍主导延迟。

处理：

- 将 `target_response_mtu` 配置化。
- benchmark 200/400/800 MTU。
- 保持 client 自动切分，避免 50 MiB response。

### 内存排序错误

风险：server 看到 READY 时 descriptor/payload 尚未完全可见。

处理：

- descriptor/payload write 和 commit write 分离。
- client 等 descriptor/payload write completion 后再写 commit。
- server 只以 commit word 判断请求可处理。

### 真实 verbs RC 尚未接入

风险：当前 transport 先用 shared memory 固定了协议和语义，但还没有覆盖真实 NIC、MR、CQ、completion 相关问题。

处理：

- 保持 `rc_transport.*` 为独立层，后续仅替换这一层。
- 继续保留 descriptor/commit/status 协议不变。
- 在切换到 verbs RC 前，先用当前 MVP 稳定上层 client/server/shard 语义和测试面。

### QP/CQ 线程归属混乱

风险：多个线程 poll 或 post 同一个 QP/CQ 导致 completion 被错误消费。

处理：

- QP 固定 owner thread。
- 第一版每个 worker/poller 管理自己的 QP 集合和 CQ。
- 禁止跨线程复用 busy QP。

## 回滚方案

新 RC path 已以独立 transport/config 接入。当前实现不再依赖旧 RDMA RawMessage 数据面；是否继续清理旧文件，取决于后续是否还需要保留旧 benchmark 或对照路径。

建议开关：

```text
RECSTORE_RDMA_RC_WRITE=1
```

或配置：

```json
{
  "cache_ps": {
    "ps_type": "RDMA_RC"
  }
}
```

如果 RC path 不稳定，可以直接切回现有 gRPC/bRPC 或旧 RDMA 测试路径。

---
