# 当前状态记录

日期：2026-06-03

## 分支状态

- 当前分支：`feat/newrdma`
- 已 rebase 到包含 benchmark 目录迁移的 `origin/main`
- benchmark 正式入口：`tools/benchmarks/run_benchmark_ps.py`
- 跨机执行说明：`.agents/skills/cross-host-rdma-ps/SKILL.md`
- 详细诊断记录：`cross_host_rdma_benchmark_diagnosis_0602.md`

未纳入本轮提交的本地脏项：`third_party/*`、`dockerfiles/init_env.log`、构建产物。

## 关键结论

- `DRAM_PET_HASH + auto/staging_copy` 必须保持 `--rdma-rc-server-get-workers 0`。
- 旧跨机低吞吐主因不是 PET_HASH，也不是 server poll 命中率本身，而是旧 benchmark 形态下 client 请求源密度不足。
- 多 logical `PetPSClient` 后，`p4/t3/q16/d16` 已恢复到 39M+；Release/O3 后可到 45M+。
- `AcquireIdleSlot` round-robin 无收益，已撤回。
- 额外 profile 字段会污染热路径；周期性 RDMA profile 的开销较小。

## 关键结果

| 配置 | 结果目录 | 吞吐 |
|-|-|-:|
| Debug clean | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_runner_fix_only_0603` | 39.384 M keys/s |
| Debug round-robin acquire | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_round_robin_0603` | 38.403 M keys/s |
| Release/O3 + profile=1000ms | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_release_0603` | 45.238 M keys/s |
| Release/O3 + profile=0 | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_release_noprofile_0603` | 45.523 M keys/s |
| Debug repeat3 | `results/benchmark_ps_cross_host_rdma_p4t3_q16_d16_repeat3_0603` | mean 34.750 M keys/s |
| Local Debug repeat3 | `results/benchmark_ps_local_rdma_p4t3_q16_d16_repeat3_0603` | mean 30.683 M keys/s |

## 推荐运行方式

吞吐测试优先使用 Release/O3：

```bash
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release
cmake --build build_release --target ps_transport_benchmark petps_server -j
```

ssh runner 增加：

```text
--build-dir build_release
--remote-build-dir build_release
```

推荐 baseline：

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

## 已验证

```bash
cmake --build build_release --target ps_transport_benchmark petps_server -j
python3 -m unittest src/test/scripts/test_run_benchmark_ps.py
git diff --check
```

## 下一步

1. 用 Release/O3 对 `p4/t3/q16/d16` 做 repeat3 稳定性测试。
2. 再看 client wait/status 轮询策略，例如 `rdma_rc_wait_spin_iterations`。
3. 合入前保持 runner、skill、报告中的命令入口一致。
