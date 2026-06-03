# RDMA PS 协议改造计划

日期：2026-06-03

状态更新：本计划中的 SRQ/SEND request-discovery 方向已被
`send_doorbell` 最小实验部分证伪。`send_doorbell` 将 request discovery
改为 completion-driven 后，跨机 full GET 仍约 `16.6 M keys/s`，说明
server blind scan 低命中率更像症状而不是根因。随后修复多 logical
`PetPSClient` 后，跨机 `p4/t3/q16/depth16` 达到 `39.087 M keys/s`。
因此短期不建议优先投入完整 SRQ/SEND descriptor 协议；若继续改协议，
更应关注 client request loop、slot reuse、per-lane queue/ring 和
logical client / raw lane 映射。

## 背景

当前 RecStore PetPS RDMA 主路径使用 RC-write slot transport：

- client 将 `RequestDescriptor + payload` 写入 server request slot；
- client 最后写 `CommitWord`；
- server poller 轮询所有 request slot 的 commit word，发现 `READY && seq > last_seq` 后处理请求；
- server 写 response payload，最后写 `StatusWord`；
- client 轮询 status 后回收 slot 并提交下一批请求。

跨机 benchmark 最初显示，该协议在本机可以达到约 `41.5 M keys/s`，但 node190/node191 跨机 full GET 约 `16.5 M keys/s`。`status_only` 可到约 `28.4 M keys/s`，说明 server 处理能力和短路径能力没有完全丢失；当时的主要怀疑是跨机 request/status 闭环变长后，server poller 需要扫描大量旧 seq 或非新请求 slot，导致有效发现新请求的效率下降。

后续实验修正了这个判断：`send_doorbell` 可以消除 blind scan 旧 seq，但没有提升跨机 full GET；多 logical client 修复则可以把跨机吞吐提升到 `39.087 M keys/s`。因此，request discovery 不是单独根因，client 请求源密度和 slot reuse/提交组织方式才是更高优先级方向。

## 已确认现象

1. `DRAM_PET_HASH + auto/staging_copy + get_workers=0` 是当前跨机测试主路径。
2. 更换 node190 server verbs device 不能恢复吞吐。
3. 本机和跨机 full GET 的 `handle_get_avg_ns` 都在约 `180 us` 量级，PET_HASH 查找不是主要跨机退化点。
4. 跨机 `scan_hit_pct` 明显低于本机。
5. 新增 profile 显示，低命中主要表现为大量 `duplicate_seq_ready`，即 poller 看到 `READY` 但 seq 没变。
6. clear-ready 可以消除 `duplicate_seq_ready`，但 full GET 只从 `16.488 M keys/s` 到 `16.830 M keys/s`，不是根因；在 `status_only` 下还会把吞吐从 `28.380 M keys/s` 降到 `19.049 M keys/s`。
7. 增加 QP 数、slot 数和 prefetch depth 没有提升：
   - q16/slots1/depth16：`16.488 M keys/s`
   - q32/slots1/depth32：`15.967 M keys/s`
   - q32/slots2/depth64：`13.567 M keys/s`
   - q64/slots1/depth64：`14.339 M keys/s`
8. 更大 outstanding 会扩大 server 每轮扫描范围，将 `poll_loop_avg_ns` 从约 `10 us` 拉高到约 `21-23 us`。

## 当前判断

瓶颈不是 server 真正执行 GET 的计算能力。早期 profile 中的低
`scan_hit_pct` 主要反映 client 新请求到达密度不足，而不是 server
request discovery 本身一定需要替换。跨机闭环变长后，每个 slot 产生新
seq 的频率降低；继续增加 QP/slot/depth 会让 poller 扫描更多旧状态，
反而增加 poll loop 成本。

因此下一版协议不应只围绕 completion/event-driven discovery 展开。
更合适的验证顺序是：先优化 client-side request loop 和 slot/ring
复用，再评估是否还需要 SRQ/SEND。

## 改造目标

1. 将 request discovery 从 `server scan all slots` 改成 `server poll completion` 或至少接近 completion-driven。
2. 保持 response path 尽量不变，降低实验改动面。
3. 先服务 benchmark 验证，不急于替换 Python/op-layer 全路径。
4. 保留现有 RC-write slot transport 作为 baseline 和 fallback。
5. 用 profile 证明收益来自 request discovery，而不是混入其他优化。

## 历史方案：SRQ/SEND 请求描述符实验

该方案作为历史设计记录保留，但已降为低优先级。原因是
`send_doorbell` 已经以最小代价验证了 completion-driven request
discovery，跨机 full GET 没有随之提升。

### 协议形态

新增一个实验 request path，例如 `rc_send_srq`：

- client 使用 RC SEND 发送小 request descriptor；
- server 使用 SRQ 或 per-QP recv queue 接收请求；
- server poll CQ 得到 recv completion 后直接定位 descriptor 并进入现有 `HandleGet`；
- response payload/status 暂时继续使用现有 response slot + RDMA WRITE；
- client 仍按现有 pending RPC/revoke 逻辑等待 status。

这样可以先把最大问题从“server 如何发现请求”中隔离出来，而不同时重写 response/status 协议。

### 请求 descriptor

最小 descriptor 可以包含：

- magic/version
- op
- shard/table id
- client id
- rpc id
- seq
- key count
- request payload length
- response slot/lane 标识，或沿用现有 client/QP/slot 映射
- 可选：client request payload remote address，用于大 payload 时让 server RDMA READ

GET 的 key payload 有两个选择：

1. 小 batch 直接随 SEND inline 或普通 send buffer 发送；
2. SEND 只发 descriptor，key payload 仍由 client RDMA WRITE 到 request buffer，或由 server RDMA READ。

第一版建议先支持当前 benchmark 的 key payload，优先做 correctness 和 profile。若 SEND payload 压力太高，再拆成 `SEND descriptor + RDMA READ/WRITE payload`。

### Server buffer 管理

server 启动时：

- 创建 SRQ；
- 注册 recv buffer ring；
- post 足够数量的 recv WR；
- 每个 recv buffer 对齐 cacheline，避免多个 poller 写同一 cacheline；
- CQ completion 后将 buffer 转换为 request view；
- request 处理完成后 repost recv buffer。

需要显式 backpressure：

- recv buffer 不足时 client SEND 会受到 RNR 或 completion 失败风险；
- 第一版应设置足够保守的 recv depth，并在 profile 中输出 recv repost、RNR、CQ poll 空轮数、recv buffer peak。

### CQ 和线程模型

第一版建议从简单模型开始：

- 每个 server poll thread 拥有一个 CQ 或一组 CQ；
- SRQ 可以共享，但 completion 分发应避免多线程抢同一个 CQ；
- 如果先用单 CQ 验证 request discovery 能否提升，再做 CQ 分片扩展性实验；
- 后续可以按 client id、QP id 或 completion queue 分区到不同 poller。

### 与现有代码的边界

建议新增实验路径，而不是直接改坏现有路径：

- `src/ps/rdma/rdma_protocol.h`：新增 SEND/SRQ descriptor 或 version 字段。
- `src/ps/rdma/raw_verbs_transport.*`：补齐 SRQ 创建、recv buffer、SEND/RECV completion 支持。
- `src/ps/rdma/rc_transport.*`：新增 request transport mode，保留现有 RC-write slot mode。
- `src/ps/rdma/petps_client.*`：新增 submit path；response wait/revoke 尽量复用。
- `src/ps/rdma/petps_server.cc`：新增 completion-driven request loop；`HandleGet` 复用。
- `src/test/scripts/run_benchmark_ps.py`：新增参数，例如 `--rdma-rc-request-mode=slot_write|send_srq`。

## 分阶段计划

### Phase 0：补充 profile，确认闭环

目的：在不改协议前，把闭环耗时拆清楚。

新增或确认字段：

- client submit 内部 descriptor write、payload write、commit write、drain completion 时间；
- client wait status 首次可见时间；
- client revoke 到下一次 submit 的间隔；
- server poll CQ/scan 时间；
- server complete response 的 payload/status write 时间；
- server per-poller request discovery hit rate。

验收：能解释 q16/q32/q64 下 poll loop 增长和 fresh seq 密度变化。

### Phase 1：最小 SRQ/SEND smoke

目的：验证 server 能通过 recv completion 发现 GET 请求。

范围：

- 单 server、单 client；
- GET only；
- 小 record count；
- response 沿用现有 slot/status；
- 不追求性能，只追 correctness。

验证：

- `ps_transport_benchmark --help` 正常；
- 单 client 本机 smoke；
- `ctest` 中新增最小 protocol test 或 transport test。

### Phase 2：跨机 p1/p6 benchmark

目的：判断 request discovery 是否是主瓶颈。

对比：

- slot_write q16/slots1/depth16；
- send_srq 同等 p6/depth16；
- status_only；
- payload_memset；
- 本机和跨机各一组。

关键判断：

- 若 send_srq 跨机 full GET 明显接近 status_only 或本机比例改善，说明 server blind scan 是核心瓶颈；
- 若 send_srq 仍约 `16M`，则继续排查 response/status 闭环或 client reuse。

### Phase 3：扩展性实验

目的：判断 SRQ 是否不仅提升命中率，也改善扩展性。

矩阵：

- server poll threads：1、4、8、16；
- client processes：1、2、4、6；
- request mode：slot_write、send_srq；
- CQ 模型：单 CQ、多 CQ；
- recv depth：保守值、较大值。

重点看：

- 每 poller completion 数是否均衡；
- CQ poll 空轮比例；
- recv repost 成本；
- RNR 或 retry；
- CPU 使用率；
- full GET 吞吐是否随 poller/client 扩展。

## 风险和注意事项

1. SRQ 不是免费优化。SEND/RECV completion 会增加 NIC/CQ 事件和 recv buffer 管理成本。
2. 纯 SEND 大 payload 可能不适合当前 500 keys batch，需要小心 payload 设计。
3. SRQ 共享 recv queue 可能改善 recv buffer 管理，但不自动保证 CQ/poller 扩展性。
4. 如果多个 poller 抢同一个 CQ，可能出现新的锁竞争或 cacheline 抖动。
5. response path 仍是 RDMA WRITE/status polling；如果最终瓶颈在 response/status 可见性，SRQ 只能部分改善。
6. 需要保留现有 slot_write 路径，避免影响当前已可复现的 baseline。

## 暂不建议的方向

1. 继续单纯增加 QP 数、slot 数或 prefetch depth：已有结果显示无效且会增加 poll loop 成本。
2. server 端每次处理后 clear-ready：能改变 profile 形态，但不是根因，还会伤害 `status_only`。
3. 直接大改全链路 request/response/status：改动面太大，不利于定位收益来源。

## 最小成功标准

第一版协议实验成功不要求直接打满 RNIC，但至少应满足：

- correctness 稳定；
- 跨机 p6 full GET 明显高于 slot_write baseline `16.5 M keys/s`；
- server request discovery 不再主要表现为扫描旧 seq；
- 增加 QP/slot/depth 不再线性放大 poll loop 成本；
- profile 能明确说明收益来自 completion-driven request discovery。
