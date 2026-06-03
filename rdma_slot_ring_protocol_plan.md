# RDMA PS Slot Ring 协议实验计划

日期：2026-06-03

## 背景

当前 RC-write slot 协议中，一个 request slot 在逻辑上只承载一个 in-flight request。client 写入 `RequestDescriptor + payload`，最后写 `CommitWord`；server poller 扫描 commit word，发现 `READY && seq > last_seq` 后处理请求。

跨机测试显示，client pending 能打满，但 server poller 经常扫到旧 seq，fresh request 密度偏低。单纯增加 QP 数、slot 数或 prefetch depth 没有改善，反而扩大了 server 每轮扫描范围。

本方案不引入 SRQ，也不改变当前 RDMA WRITE 发布请求的基本语义，而是在每个原有 slot/lane 内部扩展一个小型循环队列，使一条 lane 可以缓存多个 request entry。

## 核心想法

把当前：

```text
lane -> single request slot
```

改成：

```text
lane -> request ring[ring_depth]
```

每个 ring entry 都包含独立的：

- `RequestDescriptor`
- request payload
- `CommitWord`

client 在同一条 lane 上按 `head % ring_depth` 写入多个 request entry；server 按 `tail % ring_depth` 顺序消费。这样 server 扫到某条 lane 时，可以连续处理多个 entry，而不是每次只判断一个 slot 是否产生了新 seq。

## 预期收益

1. 降低 server poller 对 QP/slot 数量的依赖。
2. 提高单次命中后的连续处理机会，改善 cache locality。
3. 在跨机 RTT 较长时，让 client 可以提前把多个请求排到同一 lane ring 中。
4. 避免继续扩大 QP 数导致的扫描范围膨胀。
5. 保留现有 RDMA WRITE request 发布方式，改动面小于 SRQ/SEND 协议。

## 协议结构

### Ring 布局

每个 lane 分配固定大小的 ring：

```text
LaneRequestRegion
  RingControl
    client_head_seq
    server_tail_seq
    ring_depth
  Entry[0]
    RequestDescriptor
    payload
    CommitWord
  Entry[1]
    RequestDescriptor
    payload
    CommitWord
  ...
```

第一版可以不让 server 写回 `server_tail_seq`，仍由 client 通过 response status/revoke 判断 slot 是否可复用。后续如果要更明确 backpressure，再考虑 server tail 可见化。

### Client 提交

client 对每条 lane 维护：

- `next_seq`
- `head`
- `inflight_count`
- `ring_depth`

提交步骤：

1. 确认 `inflight_count < ring_depth`。
2. 选择 `entry = head % ring_depth`。
3. 写 `RequestDescriptor`。
4. 写 request payload。
5. 最后写 `CommitWord{seq, READY}`。
6. `head++`，`inflight_count++`。

response 完成并 revoke 后，client 降低对应 lane 的 `inflight_count`。

### Server 扫描

server 对每条 lane 维护：

- `tail_seq`
- `tail_index`

扫描步骤：

1. poller 扫到 lane。
2. 从 `tail_index` 开始检查 entry commit word。
3. 如果 `state == READY && seq == tail_seq + 1`，处理该 entry。
4. 处理完成后 `tail_seq++`，`tail_index = tail_seq % ring_depth`。
5. 在同一 lane 内继续尝试消费下一个 entry，直到遇到未 ready 或达到 per-lane budget。

这样 server 一旦命中某条 lane，可以批量 drain 一小段连续 ready entry。

## 关键参数

建议新增或复用参数：

- `--rdma-rc-request-ring-depth`
  - 默认 `1`，表示保持当前行为。
  - 实验值：`2`、`4`、`8`。
- `--rdma-rc-lane-drain-budget`
  - 每次 poll 命中后最多连续处理多少 entry。
  - 默认可以等于 ring depth，也可以先设为 `4`。

不要把 ring depth 和 QP 数同时拉大。第一轮建议固定 q16，只扫：

```text
q16/ring1/depth16  baseline
q16/ring2/depth32
q16/ring4/depth64
q16/ring8/depth128
```

## 与现有 slots_per_qp 的关系

当前 `slots_per_qp` 更像是在每条 QP lane 上增加多个独立逻辑 slot；server 扫描范围会随着 slot 数扩张。

slot ring 的目标不同：

- 尽量保持 lane 数不变；
- 在 lane 内增加连续 request entry；
- 命中 lane 后顺序消费；
- 避免 server 全局扫描面线性扩大。

因此第一版建议不要直接把它实现成现有 `slots_per_qp` 的别名，而是明确区分：

```text
qps_per_client_per_shard: lane/QP 数量
slots_per_qp: 现有逻辑 slot 扩展
request_ring_depth: 每个 lane 内的连续 request entry 深度
```

如果后续发现二者可以统一，再做结构收敛。

## Profile 字段

需要新增或复用这些字段判断是否有效：

- server:
  - `ring_ready_entries`
  - `ring_empty_lanes`
  - `ring_drained_entries`
  - `ring_drain_avg`
  - `ring_drain_max`
  - `ring_out_of_order_seq`
  - `poll_loop_avg_ns`
- client:
  - `ring_inflight_peak`
  - `ring_inflight_avg`
  - `ring_full_count`
  - `submit_avg_ns`
  - `wait_status_avg_ns`

最关键的判断是：`ring_drain_avg` 是否大于 `1`，以及 `poll_loop_avg_ns` 是否没有像 q32/q64 那样上涨。

## 实验矩阵

### 本机 smoke

```text
p1/q16/ring1/depth16
p1/q16/ring2/depth32
p1/q16/ring4/depth64
```

目标是 correctness 和基础 profile。

### 跨机主实验

```text
p6/q16/ring1/depth16
p6/q16/ring2/depth32
p6/q16/ring4/depth64
p6/q16/ring8/depth128
```

固定：

- `DRAM_PET_HASH`
- `rdma_get_response_mode=auto`
- `get_workers=0`
- `server_rdm_threads=16`
- `client_threads_per_process=1`

### 对照实验

保留已有对照：

- q32/slots1/depth32
- q32/slots2/depth64
- q64/slots1/depth64
- status_only
- payload_memset

## 成功标准

slot ring 方案如果有效，应至少看到：

1. 跨机 full GET 高于 q16/ring1 baseline `16.5 M keys/s`。
2. 增大 ring depth 后，吞吐上升或稳定，而不是像 q32/q64 那样下降。
3. `poll_loop_avg_ns` 不随 ring depth 明显上涨。
4. server 单次 lane 命中后能 drain 多个 ready entry。
5. `duplicate_seq_ready` 占比下降，fresh entry 密度上升。

## 主要风险

1. 如果 client 无法在同一 lane 上提前填充多个 request，ring 仍然不会变密。
2. 如果 response/status 闭环才是主要瓶颈，ring depth 增大可能只增加排队，不提升吞吐。
3. 每个 entry 都有 payload，ring depth 会显著增加 server request buffer 占用。
4. ring 内 entry 顺序消费可能遇到 head-of-line blocking：前一个 entry 未 ready 时，后面的 ready entry 暂时不能处理。
5. 如果允许乱序消费，协议复杂度会明显上升，需要 per-entry last_seq/processed 状态。

## 第一版建议

先做严格顺序 ring，不做乱序：

- 每条 lane 一个 ring；
- ring depth 从 `2` 和 `4` 开始；
- server 每次命中后最多 drain `ring_depth` 个 entry；
- response path 不变；
- `request_ring_depth=1` 必须完全等价于现有协议。

如果严格顺序 ring 已经能提升，就继续优化 drain budget 和布局；如果没有提升，再判断是否需要乱序消费或转向 SRQ/SEND completion-driven 方案。
