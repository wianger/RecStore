# 跨机 RDMA PS Benchmark 诊断记录

日期：2026-06-02

## 摘要

- 当前推荐跨机 baseline：`p4/t3/q16/d16`，12 logical clients，`get_workers=0`。
- Debug clean 单次为 `39.384 M keys/s`；Release/O3 单次为 `45.238-45.523 M keys/s`。
- 旧 16M 级低分主要来自 client 请求源密度不足；多 logical client 后已恢复。
- server poll 低命中率和 SEND/SRQ 方向实验都没有单独解释吞吐下降。
- round-robin acquire 无收益；额外 hot-path profile 字段会明显扰动结果。

## 背景

本次诊断目标是解释 RecStore PS/network 层跨机 RDMA benchmark 为什么明显低于本机 RDMA benchmark。

测试拓扑：

- server：`node190` / `10.0.2.190`
- client：`node191` / `10.0.2.191`
- 两端容器名：`recstore`
- 容器内仓库路径：`/app/RecStore`
- server 端挂载目录：`/home/xieminhui/jkt/RecStore`
- client 端挂载目录：`/home/xieminhui/jkt/recstore`

主要 benchmark 约束：

- 使用 `$benchmark-ps` 工作流，即 `tools/benchmarks/run_benchmark_ps.py`
- transport：`rdma`
- index：`DRAM_PET_HASH`
- value size：`512`
- record count：`1000000`
- batch keys：`500`
- client processes：`6`
- 每个 client process 线程数：`1`
- server RDMA poll threads：`16`
- `--rdma-get-response-mode auto`
- 对 `DRAM_PET_HASH + auto/staging_copy`，`--rdma-rc-server-get-workers` 必须为 `0`

## 已确认环境现象

node190 容器内有 3 个 verbs device：

| host | device | NUMA | rate |
|-|-|-|-|
| node190 | mlx5_0 | 1 | 200 Gb/sec |
| node190 | mlx5_1 | 1 | 200 Gb/sec |
| node190 | mlx5_2 | 1 | 200 Gb/sec |
| node191 | mlx5_0 | 0 | 200 Gb/sec |

代码中的 `--rdma-rc-server-numa-id` 当前会通过 `SelectRawVerbsDeviceIndex` 选择 verbs device index，而不是严格按真实 NUMA node 查找设备。因此在 node190 上：

- `--rdma-rc-server-numa-id 0` 选择 `mlx5_0`
- `--rdma-rc-server-numa-id 1` 选择 `mlx5_1`
- `--rdma-rc-server-numa-id 2` 选择 `mlx5_2`

## 关键结果

结果路径均在仓库根目录 `results/` 下。

| run | 结果目录 | run 吞吐 |
|-|-|-:|
| 本机 p6/q16 | `results/benchmark_ps_local_rdma_p6_profile_compare` | 41.534 M keys/s |
| 跨机 p6/q16, server device 0 | `results/benchmark_ps_cross_host_rdma_p6_clean_profile` | 16.527 M keys/s |
| 跨机 p6/q16, server device 1 | `results/benchmark_ps_cross_host_rdma_p6_server_dev1_profile` | 16.386 M keys/s |
| 跨机 p6/q16, server device 2 | `results/benchmark_ps_cross_host_rdma_p6_server_dev2_profile` | 16.406 M keys/s |
| 跨机 p6/q32 | `results/benchmark_ps_cross_host_rdma_p6_q32_profile` | 11.866 M keys/s |
| 跨机 p6/slots2 | `results/benchmark_ps_cross_host_rdma_p6_slots2_profile` | 16.110 M keys/s |
| 跨机 p6/status_only | `results/benchmark_ps_cross_host_rdma_p6_status_only_clean` | 28.380 M keys/s |
| 跨机 p6/payload_memset | `results/benchmark_ps_cross_host_rdma_p6_payload_memset_clean` | 17.274 M keys/s |
| 跨机 p6/skip_client_copy | `results/benchmark_ps_cross_host_rdma_p6_skip_client_copy_clean` | 16.266 M keys/s |
| 跨机 p6/q16, 增加 server clear-ready 实验 | `results/benchmark_ps_cross_host_rdma_p6_clear_ready_0603` | 16.830 M keys/s |
| 跨机 p6/status_only, 增加 server clear-ready 实验 | `results/benchmark_ps_cross_host_rdma_p6_status_only_clear_ready_0603` | 19.049 M keys/s |
| 跨机 p6/q32/slots1/depth32 | `results/benchmark_ps_cross_host_rdma_p6_q32_s1_d32_0603` | 15.967 M keys/s |
| 跨机 p6/q32/slots2/depth64 | `results/benchmark_ps_cross_host_rdma_p6_q32_s2_d64_0603` | 13.567 M keys/s |
| 跨机 p6/q64/slots1/depth64 | `results/benchmark_ps_cross_host_rdma_p6_q64_s1_d64_0603` | 14.339 M keys/s |
| 跨机 p6/q2/depth2 | `results/benchmark_ps_cross_host_rdma_p6_q2_d2_0603` | 6.762 M keys/s |
| 跨机 p6/q4/depth4 | `results/benchmark_ps_cross_host_rdma_p6_q4_d4_0603` | 11.032 M keys/s |
| 跨机 p8/q2/depth2 | `results/benchmark_ps_cross_host_rdma_p8_q2_d2_0603` | 6.772 M keys/s |
| 跨机 p8/q4/depth4 | `results/benchmark_ps_cross_host_rdma_p8_q4_d4_0603` | 12.939 M keys/s |
| 跨机 p8/q8/depth8 | `results/benchmark_ps_cross_host_rdma_p8_q8_d8_0603` | 16.487 M keys/s |
| 跨机 p8/q16/depth16 | `results/benchmark_ps_cross_host_rdma_p8_q16_d16_0603` | 16.571 M keys/s |
| 跨机 p6/t2/q8/depth8, 多 logical client | `results/benchmark_ps_cross_host_rdma_p6t2_q8_d8_0603` | 21.328 M keys/s |
| 跨机 p6/t2/q12/depth12, 多 logical client | `results/benchmark_ps_cross_host_rdma_p6t2_q12_d12_0603` | 26.652 M keys/s |
| 跨机 p6/t2/q16/depth16, 多 logical client | `results/benchmark_ps_cross_host_rdma_p6t2_q16_d16_0603` | 31.639 M keys/s |
| 跨机 p6/t2/q18/depth18, 多 logical client | `results/benchmark_ps_cross_host_rdma_p6t2_q18_d18_0603` | 22.638 M keys/s |
| 跨机 p6/t2/q20/depth20, 多 logical client | `results/benchmark_ps_cross_host_rdma_p6t2_q20_d20_0603` | 25.311 M keys/s |
| 跨机 p8/t2/q16/depth16, 多 logical client | `results/benchmark_ps_cross_host_rdma_p8t2_q16_d16_0603` | 31.270 M keys/s |
| 跨机 p4/t3/q16/depth16, 多 logical client | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_0603` | 39.087 M keys/s |
| 跨机 p3/t4/q16/depth16, 多 logical client | `results/benchmark_ps_cross_host_rdma_p3t4_q16_d16_0603` | 35.733 M keys/s |
| 跨机 p4/t4/q16/depth16, 多 logical client | `results/benchmark_ps_cross_host_rdma_p4t4_q16_d16_0603` | 38.861 M keys/s |
| 跨机 p4/t3/q16/depth16 repeat3, 多 logical client | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_repeat3_0603` | mean 34.750 M keys/s |
| 本机 p4/t3/q16/depth16 repeat3, 多 logical client | `results/benchmark_ps_local_rdma_p4t3_q16_d16_repeat3_0603` | mean 30.683 M keys/s |
| 跨机 send_doorbell request discovery 实验 | `results/rdma_ps_cross_send_p6_t16_q16_d16_0603` | 16.627 M keys/s |
| 跨机 p4/t3/q16/depth16, runner SSH/endpoint split 修复后干净重跑 | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_runner_fix_only_0603` | 39.384 M keys/s |
| 跨机 p4/t3/q16/depth16, Release/O3 + profile=1000ms | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_release_0603` | 45.238 M keys/s |
| 跨机 p4/t3/q16/depth16, Release/O3 + profile=0 | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_release_noprofile_0603` | 45.523 M keys/s |

## 稳定性结果

`p4/t3/q16/depth16` 是当前最有价值的跨机 baseline，但单次
`39.087 M keys/s` 更像偏高峰值。repeat3 后更可信的稳定吞吐约为
`35 M keys/s`。

| 环境 | 结果目录 | repeat totals | mean | stdev | cv |
|-|-|-|-:|-:|-:|
| 跨机 node190/node191 | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_repeat3_0603` | 37.130 / 32.024 / 35.096 M keys/s | 34.750 M keys/s | 2.099 | 6.04% |
| 本机 local | `results/benchmark_ps_local_rdma_p4t3_q16_d16_repeat3_0603` | 28.061 / 30.457 / 33.531 M keys/s | 30.683 M keys/s | 2.239 | 7.30% |

本机同配置低于跨机同配置，因此它不是一个更干净的上限对照。更可能的解释是
local 模式把 server/client 都放在 node190 上，CPU、cache、NUMA 或 RNIC
资源竞争更重；跨机模式把 client CPU 压力放到 node191，反而更接近实际部署。

稳定性 profile 对比：

| 环境 | client submit_avg_ns | client wait_status_avg_ns | server handle_get_avg_ns | 说明 |
|-|-:|-:|-:|-|
| 跨机 repeat3 | 约 3.3-3.6 us | 约 121-153 us | 约 203-242 us | submit 稳定，波动主要来自 wait/status 和 server GET |
| 本机 repeat3 | 约 4.7-5.8 us | 约 132-170 us | 约 222-277 us | submit 和 server GET 都比跨机更慢 |
| 跨机 Release/O3 | 约 2.4-2.9 us | 约 108-115 us | 约 89-99 us | 编译优化显著压缩 client submit/revoke 和 server GET 固定开销 |

## Profile 对比

下面是从最后几个有效 profile interval 汇总出的关键字段。该表用于定位方向，不应替代完整原始日志。

| run | run M keys/s | server scan_hit_pct | handle_get_avg_ns | complete_response_avg_ns | client submit_avg_ns | client wait_status_avg_ns | pending peak |
|-|-:|-:|-:|-:|-:|-:|-:|
| 本机 p6/q16 | 41.534 | 5.824 | 179787 | 3443 | 4061 | 110724 | 16 |
| 跨机 p6/q16 device0 | 16.527 | 0.256 | 180069 | 4187 | 9688 | 5241 | 16 |
| 跨机 p6/q16 device1 | 16.386 | 0.247 | 163953 | 3545 | 24915 | 10304 | 16 |
| 跨机 p6/q16 device2 | 16.406 | 0.424 | 180755 | 4009 | 13990 | 26342 | 16 |
| 跨机 p6/q32 | 11.866 | 0.105 | 211259 | 4252 | 5618 | 248978 | 32 |
| 跨机 p6/slots2 | 16.110 | 0.135 | 194621 | 4280 | 7001 | 129928 | 16 |
| 跨机 status_only | 28.380 | 0.529 | 140 | 2494 | 6894 | 74396 | 16 |
| 跨机 payload_memset | 17.274 | 0.285 | 22101 | 3689 | 14479 | 39899 | 16 |
| 跨机 skip_client_copy | 16.266 | 0.269 | 200716 | 4308 | 15152 | 13150 | 16 |

## 主要判断

1. 当前跨机 benchmark 确认使用的是 `RDMA + DRAM_PET_HASH + auto/staging_copy + get_workers=0`。

2. 单纯更换 node190 上的 server verbs device 不能恢复吞吐。device 0/1/2 都在 `16.4 M keys/s` 左右，因此这不像是简单选错某张 server RNIC。

3. `DRAM_PET_HASH` 查找本身不是跨机掉速的主因。本机和跨机默认配置下 `handle_get_avg_ns` 都在约 `180 us` 量级。

4. 跨机时 server poller 的 ready slot 命中率显著下降。本机 `scan_hit_pct` 约 `5.8%`，跨机默认只有约 `0.25%`。这说明服务端大部分 poll round 扫不到 ready request，server poller 没有被持续喂满。

5. 增大 outstanding 没有改善。`prefetch/qps` 从 `16` 提到 `32` 后，吞吐从 `16.527 M keys/s` 降到 `11.866 M keys/s`，并且 `wait_status_avg_ns` 明显变差。

6. 增加 `slots_per_qp` 也没有改善。`slots_per_qp=2` 后吞吐约 `16.110 M keys/s`。

7. client response copy 不是主要瓶颈。`skip_client_copy` 后吞吐仍约 `16.266 M keys/s`。

8. `status_only` 能达到 `28.380 M keys/s`，说明纯 request/status 路径仍明显低于本机 full GET，但高于跨机 full GET。payload 路径仍有成本，但不是唯一瓶颈。

9. server 端 clear-ready 实验不是根因。它能把 `duplicate_seq_ready` 从百万/千万级降到 `0`，但 full GET 吞吐只从 `16.488 M keys/s` 到 `16.830 M keys/s`，提升约 `2%`。同时在 `status_only` 下，clear-ready 会把吞吐从 `28.380 M keys/s` 降到 `19.049 M keys/s`，说明额外清状态写入会伤害极短路径。因此该优化不适合作为默认修复，也不能解释跨机吞吐下降。

10. 单纯提升 QP 数、slot 数和 prefetch depth 没有改善跨机 full GET。`q32/slots1/depth32` 为 `15.967 M keys/s`，`q32/slots2/depth64` 为 `13.567 M keys/s`，`q64/slots1/depth64` 为 `14.339 M keys/s`，都低于 q16/slots1/depth16 基线。对应 profile 中 client pending 仍打满，但 server 仍主要扫到旧 seq；同时扫描范围增大后 `poll_loop_avg_ns` 从约 `10 us` 上升到约 `21-23 us`，说明简单堆 outstanding 会增加 poll/slot 状态面成本。

11. 降低每 client 的 QP/depth 到 `2` 或 `4` 会明显降低吞吐。`p6/q2/depth2` 约 `6.762 M keys/s`，`p6/q4/depth4` 约 `11.032 M keys/s`；增加到 `p8` 后，`q2` 仍约 `6.772 M keys/s`，`q4` 提升到 `12.939 M keys/s`，但仍低于 p6/q16 基线。对应 client profile 中 `q2` 的 `wait_status_avg_ns` 约 `400-540 us`，`q4` 约 `227-275 us`，说明过浅的 per-client outstanding 会拉长 client 请求闭环。

12. `p8/q8/depth8` 基本追平 p6/q16，吞吐约 `16.487 M keys/s`；`p8/q16/depth16` 为 `16.571 M keys/s`，提升很小。这说明 `q8` 已足够接近旧实现的平台，但单纯增加 OS client process 或继续使用 q16 不能突破约 `16.5 M keys/s`。

13. 修复 benchmark 内多 logical `PetPSClient` 后，跨机吞吐显著提升。`p6/t2/q8` 为 `21.328 M keys/s`，`p6/t2/q12` 为 `26.652 M keys/s`，`p6/t2/q16` 为 `31.639 M keys/s`。这说明旧低分的主因不是 server 处理能力，而是单进程/单 logical client 形态下 client 请求源密度不足。

14. 继续增大 q/depth 并不稳定。`p6/t2/q18` 降到 `22.638 M keys/s`，`p6/t2/q20` 为 `25.311 M keys/s`；这些配置的 lane 数不能被 16 个 server poll thread 均匀分配，profile 中 poller work distribution 明显不均，说明简单加深度会被扫描面和映射不均反噬。

15. 当前最优跨机单次配置是 `p4/t3/q16/depth16`，最高达到 `39.087 M keys/s`。repeat3 后均值为 `34.750 M keys/s`，更适合作为当前稳定 baseline。对比 `p6/t2/q16` 的 `31.639 M keys/s` 和 `p8/t2/q16` 的 `31.270 M keys/s`，更少 OS process、每进程承载 3 个 logical client 的请求闭环更稳；`p3/t4/q16` 降到 `35.733 M keys/s`，`p4/t4/q16` 为 `38.861 M keys/s`，说明每进程 3 个 logical client 是当前更好的甜点。

16. 本机同配置 `p4/t3/q16/depth16 repeat3` 的均值只有 `30.683 M keys/s`，低于跨机同配置的 `34.750 M keys/s`。因此后续不应把该 local 对照当成上限；它更像是同机资源竞争更重的 lane observation。

17. `send_doorbell` 最小实验把 request discovery 改成 completion-driven，消除了 blind scan 旧 seq，但跨机 full GET 仍约 `16.627 M keys/s`。因此 server request discovery 低命中率更像症状，不是单独根因；短期不建议继续优先投入完整 SRQ/SEND descriptor 协议。

18. runner 已经区分 SSH target 和 RecStore/RDMA endpoint host。`server-plan` /
`client-plan` 可以写 `xieminhui@10.0.2.xxx` 供 SSH 使用，但生成的 RecStore
配置会自动使用纯 endpoint IP，避免把 `user@host` 写进 RDMA 控制面导致解析失败。

19. Release/O3 是目前最明确的硬性优化手段。当前历史 `build/` 为 Debug/O0，
干净重跑为 `39.384 M keys/s`；同配置切到 `build_release` 后达到
`45.238 M keys/s`，关闭周期性 RDMA profile 后为 `45.523 M keys/s`。因此
主要收益来自编译优化，而不是关闭 profile。

20. client slot acquire round-robin 实验没有收益。同配置 round-robin acquire
为 `38.403 M keys/s`，低于干净基线，因此已撤回，不应把从头线性扫描视为当前
主要瓶颈。

## 当前最可能的瓶颈方向

旧实现中的主要问题在跨机 RDMA request/status 闭环、slot 复用节奏和 server poller 的有效新请求密度，而不是 PET_HASH 查询本身，也不是旧 `READY` 状态未清除本身。修复多 logical `PetPSClient` 后，跨机吞吐已经恢复到接近本机水平，因此当前更准确的判断是：client 请求源密度和 OS process / logical client 的组织方式是关键调优面。

具体表现是：

- client 侧 `pending_rpc_peak` 已经达到配置上限；
- server 侧却只有很低的 `scan_hit_pct`，而新增 profile 显示低命中主要来自旧 seq 的 `duplicate_seq_ready` 或 clear-ready 后的 `not_ready_slots`，不是 RDMA WRITE 完全不可见；
- 增大 `prefetch/qps` 或 `slots_per_qp` 没有让 server poller 看到更高密度的新 seq，反而扩大扫描范围并增加 poll loop 成本；
- `handle_get_avg_ns` 与本机接近，说明真正进入 GET handler 后的处理成本没有大幅恶化。
- 多 logical client 修复后，在相同 12 个 logical client / q16 下，`p4/t3` 明显优于 `p6/t2`，说明 OS process 数、每进程 logical client 数和 raw lane 映射会影响 request loop 的稳定性。

因此下一步应重点检查：

- RDMA request slot 的状态写入和状态可见性；
- server poller 扫描 slot 的策略是否适合跨机 RTT；
- client 对 slot 的 acquire/revoke/wait 逻辑是否在跨机下形成额外等待；
- benchmark 已支持一个 OS client process 内多个 logical `PetPSClient`，后续跨机 profile 推荐从 `p4/t3/q16/depth16` 作为新 baseline；
- request/status 写入是否存在 cacheline、memory ordering、poll 粒度或 slot 复用问题；
- 是否需要为跨机场景调整 QP/slot 映射，而不是单纯提高 qps 数量；
- 是否需要让 logical client / raw lane 到 server poller 的映射保持均匀，避免 `q18/q20` 这类不整除布局造成 poller 负载偏斜。

## 当前推荐 baseline

跨机 RDMA PET_HASH profile 建议优先使用：

```text
client OS processes = 4
client_threads_per_process = 3
logical clients = 12
qps_per_client_per_shard = 16
prefetch_depth = 16
slots_per_qp = 1
server RDMA poll threads = 16
rdma_get_response_mode = auto
rdma_rc_server_get_workers = 0
```

该配置对应结果目录：

```text
results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_repeat3_0603
```

当前稳定吞吐应记为约 `35 M keys/s`；单次 `39.087 M keys/s` 可作为已观察到的高点，但不应作为 repeat baseline。

如果目标是测当前上限，应使用 Release/O3 构建：

```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --target ps_transport_benchmark petps_server -j
```

并在 ssh runner 中加入：

```text
--build-dir build_release
--remote-build-dir build_release
```

当前 Release/O3 单次高点为 `45.523 M keys/s`，结果目录为：

```text
results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_release_noprofile_0603
```

## 文档状态

当前保留的主要资料：

- `cross_host_rdma_benchmark_diagnosis_0602.md`：本文件，作为跨机 RDMA PS 诊断和结论的主记录。
- `.agents/skills/cross-host-rdma-ps/SKILL.md`：当前跨机 benchmark 执行方式和推荐 baseline。
- `ps_rdma_benchmark_report_0531.md`：较早的本机 RDMA benchmark 历史记录。

已删除的临时计划/草稿文档，其结论已合并到本文件：

- `rdma_multi_pet_client_plan.md`
- `rdma_protocol_redesign_plan.md`
- `rdma_send_doorbell_experiment_report_0603.md`
- `rdma_slot_ring_protocol_plan.md`
- `todo.md`

## 复现命令示例

跨机默认 p6/q16 profile：

```bash
python3 tools/benchmarks/run_benchmark_ps.py \
  --transports rdma \
  --server-plan 0:xieminhui@10.0.2.190:25000:0 \
  --client-plan 0:xieminhui@10.0.2.191,1:xieminhui@10.0.2.191,2:xieminhui@10.0.2.191,3:xieminhui@10.0.2.191,4:xieminhui@10.0.2.191,5:xieminhui@10.0.2.191 \
  --record-count 1000000 \
  --value-size 512 \
  --batch-keys 500 \
  --index-type DRAM_PET_HASH \
  --client-threads-per-process 1 \
  --client-load-threads-per-process 1 \
  --runtime-seconds 5 \
  --repeat 1 \
  --execution-backend ssh \
  --remote-sync check \
  --remote-repo /app/RecStore \
  --remote-container recstore \
  --prefetch-depth 16 \
  --rdma-rc-qps-per-client-per-shard 16 \
  --rdma-rc-slots-per-qp 1 \
  --server-rdma-threads 16 \
  --rdma-rc-server-get-workers 0 \
  --rdma-rc-server-coroutines-per-thread 1 \
  --rdma-get-response-mode auto \
  --rdma-rc-profile-interval-ms 1000 \
  --rdma-rc-server-numa-id 0 \
  --rdma-rc-client-numa-id 0 \
  --rdma-server-bind-core-offset 0 \
  --rdma-client-bind-core-offset 16 \
  --rdma-client-bind-core-stride 2 \
  --rdma-control-plane-port 25350 \
  --output-dir results/benchmark_ps_cross_host_rdma_p6_profile_next \
  --show-runner-logs
```
