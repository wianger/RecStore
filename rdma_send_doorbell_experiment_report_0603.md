# RDMA send_doorbell 请求发现实验报告

日期：2026-06-03

## 背景

此前跨机 RDMA PS benchmark 中，`slot_write` 协议在 node190/node191 上的 full GET 吞吐约为 `16.5 M keys/s`，明显低于本机约 `39-41 M keys/s` 的水平。

当时 server profile 中有大量 `duplicate_seq_ready`，即 server poller 反复看到 `READY` 状态但 seq 没变。因此一个直接假设是：跨机 RTT 拉长后，server blind scan 大量旧 slot，request discovery 命中率低，拖低了吞吐。

本次实验实现并验证了一个更小的 completion-driven request discovery 方案，用来检验这个假设。

## 实现内容

实验分支：

```text
feat/rdma-send-doorbell-experiment
```

实验提交：

```text
ca0233d3 perf(rdma): add send doorbell request mode
```

新增参数：

```text
--rdma-rc-request-mode=slot_write|send_doorbell
```

默认仍为 `slot_write`，保持原协议不变。

`send_doorbell` 模式的协议形态：

1. client 仍然用 RDMA WRITE 写入原来的 request slot，包括 `RequestDescriptor + payload`。
2. client 仍然写 commit word。
3. client 额外发一个 `SEND_WITH_IMM` doorbell。
4. server poll CQ completion，通过 imm data 定位 `client_id + qp_index + slot_in_qp`。
5. server 后续处理 GET 和写 response/status 的路径保持不变。

这个方案不是完整 SRQ/SEND descriptor 协议；它只替换 request discovery，目的是最小化改动并验证“server blind scan 是否是跨机根因”。

## 本机验证结果

本机固定同 NUMA 和绑核：

```text
server_numa=0
client_numa=0
server_core_offset=0
client_core_offset=16
client_core_stride=2
DRAM_PET_HASH
rdma_get_response_mode=auto
rdma_rc_server_get_workers=0
```

主要结果：

| 配置 | 吞吐 |
|---|---:|
| `p6 t16 q8 d8` | `25.297 M keys/s` |
| `p6 t16 q16 d16` | `39.036 M keys/s` |
| `p6 t16 q32 d32` | `37.103 M keys/s` |
| `p6 t16 q64 d64` | `36.311 M keys/s` |
| `p4 t16 q16 d16` | `36.365 M keys/s` |
| `p8 t16 q16 d16` | `30.095 M keys/s` |
| `p6 t8 q16 d16` | `23.590 M keys/s` |
| `p6 t24 q16 d16` | `28.505 M keys/s` |

本机最佳点为：

```text
p6 / server_threads=16 / qps=16 / slots=1 / prefetch_depth=16
```

结果目录：

```text
results/rdma_ps_tune_0603/
```

本机 profile 现象：

- `duplicate_seq_ready=0`
- `scan_hit_pct≈100%`
- `pending_rpc_peak=16`
- `pending_rpc_avg≈15`

说明 `send_doorbell` 确实消除了 blind scan 旧 seq 的问题，并且本机能够接近原先 slot-write 的高吞吐水平。

## 跨机验证结果

跨机环境：

```text
server: node190 / 10.0.2.190
client: node191 / 10.0.2.191
container: recstore
repo: /app/RecStore
```

跨机命令关键参数：

```text
client_processes=6
server_threads=16
qps_per_client_per_shard=16
slots_per_qp=1
prefetch_depth=16
DRAM_PET_HASH
rdma_get_response_mode=auto
rdma_rc_server_get_workers=0
rdma_rc_request_mode=send_doorbell
```

跨机结果：

```text
send_doorbell cross-host: 16.627 M keys/s
```

结果目录：

```text
results/rdma_ps_cross_send_p6_t16_q16_d16_0603
```

对比此前基线：

```text
slot_write cross-host:    about 16.5 M keys/s
send_doorbell cross-host: 16.627 M keys/s
```

吞吐基本没有改善。

## 关键 profile 观察

跨机 `send_doorbell` server profile：

```text
duplicate_seq_ready=0
scan_hit_pct≈100%
handle_get_avg_ns≈175-178 us
complete_response_avg_ns≈3.1-3.5 us
poller_active=16
```

这说明 server 已经不再因为 blind scan 旧 seq 浪费大量发现请求的时间。server 能通过 completion 准确发现新请求。

但是跨机 client profile 显示：

```text
pending_rpc_peak=16
pending_rpc_avg≈15
submit_avg_ns≈10-32 us
drain_submit_avg_ns 部分 client 达到 7-18 us
```

本机最佳配置中，client `submit_avg_ns` 通常约为 `4 us` 量级。跨机下 submit/drain 明显变重。

## 结论

本次实验排除了一个重要假设：

```text
跨机吞吐低的根因不是 server request discovery 的 blind scan 命中率低。
```

`send_doorbell` 已经把 request discovery 改成 completion-driven，并且 profile 中 `duplicate_seq_ready` 被消除、`scan_hit_pct` 接近 100%，但跨机 full GET 仍然只有约 `16.6 M keys/s`。

因此，原先 slot-write 中的低 `scan_hit_pct` 更像是症状，而不是根因。它反映的是跨机闭环变长后，新请求到达 server 的频率低；server 扫到的大量旧 seq 只是这个现象的表现。

当前更可能的瓶颈在 client 侧闭环：

```text
submit request
drain previous submit completion
wait response status
revoke slot
reuse slot
```

跨机 RTT 和 verbs completion 可见性拉长后，每个 slot 的 reuse 频率下降。client 虽然能把 `prefetch_depth=16` 打满，但打满后下一轮提交受 submit/drain 和 status/revoke 闭环约束。

## 对后续方向的影响

完整 SRQ/SEND descriptor 协议主要解决的是：

```text
server 如何发现请求
```

但本次 `send_doorbell` 已经证明，仅解决 request discovery 不足以提升跨机 full GET。因此继续把大量精力投入 SRQ/SEND descriptor，短期性价比可能不高。

更值得优先验证的方向是减少 client slot reuse 闭环成本：

1. 减少每个 RPC 的 WR 数量和 signaled completion 依赖。
2. 避免每次复用 slot 前等待上一轮 submit completion。
3. 改为更大的 per-QP request ring，让 client 可以推进 tail，而不是依赖短 slot 闭环。
4. 将 request publish 设计成批量化或队列化，降低跨机 RTT 对单个 slot reuse 的影响。
5. 增加 profile 字段，继续拆分 client submit、drain submit、status visibility、revoke 到下一次 submit 的耗时。

## 当前建议

保留 `feat/rdma-send-doorbell-experiment` 分支作为反证实验和后续参考，但不要直接把 `send_doorbell` 合入主线作为跨机性能优化。

下一版协议改造应从 client-side request ring / slot reuse 机制入手，而不是继续只改 server-side request discovery。
