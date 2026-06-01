# rs_demo

用于在本地快速模拟较大数据量的 RecStore 训练读写更新压力，并导出结构化性能数据。
该 demo 默认复用 DLRM 同源数据入口和组织方式（`processed_day_0_data` + custom dataloader + KJT）。

## 1. 功能

- 使用 DLRM 相同数据来源：`model_zoo/torchrec_dlrm/processed_day_0_data`
- 使用 DLRM 相同稀疏组织：26 特征 -> KJT -> 拼接 ids
- 更新使用与 DLRM 融合模式一致的 fused id：`(table_idx << fuse_k) + id`
- 内部采用模块化结构：`config / data / runtime / runners / cli`
- 执行批量 `emb_read` + `emb_update_table` 循环（可调 steps/batch）
- 可选自动启动/停止 `ps_server`
- 强制开启本地结构化上报（JSONL）
- 自动调用 `analyze_embupdate_stages.py` 导出 CSV
- `read_before_update` 默认走 `emb_prefetch + emb_wait_result` 稳定读路径（避免同步读路径在部分环境下崩溃）
- 可用 `--prefetch-depth` 控制 fused embedding lookahead 预取幅度；`0` 保持同 batch issue+wait 路径，`1+` 会提前发起未来 batch 的预取并在后续 batch 消费。
- 当 `--prefetch-depth > 0` 与 `--enable-gpu-cache` 同时开启，RecStore 会在消费 batch 前等待对应 fused prefetch handle，并把返回 embedding 通过 `prefill_gpu_cache` 写入 GPU cache；随后本 batch 的 local lookup 可从 GPU cache 查询，后续 batch 也能通过 profile 观察 cache request/hit/miss。

## 2. 快速运行

默认输出目录已迁移到共享挂载盘：

- `/nas/home/shq/docker/rs_demo/outputs/<run_id>`
- `/nas/home/shq/docker/rs_demo/logs/<run_id>`
- `/nas/home/shq/docker/rs_demo/runtime/<run_id>`

在仓库根目录执行：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --steps 60 \
  --batch-size 4096 \
  --num-embeddings 200000 \
  --embedding-dim 128 \
  --run-id rs-demo-recstore-local \
  --output-root /nas/home/shq/docker/rs_demo
```

单机 distributed TorchRec：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 1 \
  --node-rank 0 \
  --nproc-per-node 4 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  --rdzv-id rs-demo-local \
  --run-id rs-demo-local \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server
```

## 3. 预取幅度实验

用于验证当前预取是否真正产生 overlap，并统计不同 lookahead 幅度的收益：

```bash
for depth in 0 1 2 4 8 16; do
  python3 model_zoo/rs_demo/run_mock_stress.py \
    --backend recstore \
    --steps 80 \
    --warmup-steps 5 \
    --batch-size 4096 \
    --num-embeddings 200000 \
    --embedding-dim 128 \
    --read-mode prefetch \
    --prefetch-depth "${depth}" \
    --run-id "recstore-prefetch-depth-${depth}" \
    --output-root /nas/home/shq/docker/rs_demo
done
```

重点看 `recstore_main.csv` 和 `recstore_main_agg.csv` 中这些列：

- `prefetch_depth`
- `prefetch_issued_batches`
- `prefetch_consumed_batches`
- `prefetch_pending_batches`
- `prefetch_total_ids`
- `prefetch_issue_ms`
- `lookup_wait_ms`
- `embed_lookup_local_ms`
- `step_total_ms`

判断方式：

- `--prefetch-depth 0` 是 legacy 同 batch issue+wait，对 overlap 没有证明力。
- `--prefetch-depth 1+` 如果实现有效，`prefetch_consumed_batches` 应从第 `depth` 个 batch 后变为 1，`lookup_wait_ms` 应随 depth 增大下降，直到平台期。
- 同时开启 `--enable-gpu-cache --gpu-cache-capacity <rows>` 时，对比 `lookup_gpu_cache_*` 字段，判断 GPU cache 查询开销是否抵消了 lookahead 预取收益。

GPU cache 与预取结合的推荐解释是：lookahead 负责把未来 batch 的 embedding 提前拉近，GPU cache 负责保留未来会复用的 embedding。当前实现是 BagPipe 思路的 RecStore 增量版本：不实现 oracle server、LRPP 或 TTL cache，但会复用 `prefetch_depth`、fused id、prefetch handle、GPU cache prefill/profile，在 batch 消费点把 lookahead prefetch 结果写入 GPU cache。若 CUDA、GPU cache prefill API 或 wait 结果不可用，会明确计入 fallback 字段并回到已有 local lookup/pull 路径。

四种 lane 的 smoke/perf 对比可用：

```bash
python3 model_zoo/rs_demo/run_lookahead_gpu_cache_lanes.py \
  --steps 20 \
  --warmup-steps 2 \
  --batch-size 512 \
  --num-embeddings 20000 \
  --gpu-cache-capacity 8192 \
  --output-root /tmp/recstore_lookahead_gpu_cache_lanes
```

该脚本依次运行：

- `baseline`：`--prefetch-depth 0`，不开 GPU cache。
- `prefetch_only`：`--prefetch-depth 2`，不开 GPU cache。
- `gpu_cache_only`：`--prefetch-depth 0 --enable-gpu-cache`。
- `prefetch_gpu_cache`：`--prefetch-depth 2 --enable-gpu-cache`。

输出汇总：`<output_root>/lookahead_gpu_cache_lane_summary.csv`。如已有外部 server，可追加 `--no-start-server --library-path <lib_recstore_ops.so>`。

关键 CSV 字段：

- 端到端训练：`step_total_ms`、`samples_per_sec`、`batches_per_sec`。
- embedding 访问分解：`lookup_total_ms`、`embed_lookup_local_ms`、`prefetch_issue_ms`、`lookup_wait_ms`、`lookup_fallback_pull_ms`。
- dense/overlap：`dense_compute_ms`、`dense_fwd_ms`、`backward_ms`、`optimizer_ms`、`prefetch_queue_residence_ms`、`prefetch_issue_to_consume_ms`、`prefetch_wait_share_of_lookup`、`prefetch_network_wait_ms`、`prefetch_exposed_network_ms`、`prefetch_dense_cover_ratio`、`prefetch_issue_to_consume_cover_ratio`。
- GPU cache：`lookup_gpu_cache_request_count`、`lookup_gpu_cache_hit_count`、`lookup_gpu_cache_miss_count`、`lookup_gpu_cache_hit_rate`、`lookup_gpu_cache_query_ms`、`lookup_gpu_cache_fill_ms`、`update_gpu_cache_invalidate_ms`。
- prefetch-to-cache：`planned_gpu_cache_prefill_batches`、`planned_gpu_cache_prefill_wait_ms`、`planned_gpu_cache_prefill_ids`、`planned_gpu_cache_prefill_successes`、`planned_gpu_cache_prefill_fallbacks`、`planned_gpu_cache_prefill_wait_failures`、`planned_gpu_cache_prefill_result_size_mismatches`、`planned_gpu_cache_prefill_no_cuda`、`planned_gpu_cache_prefill_no_api`。
- 数据规模与窗口足迹：`batch_raw_ids`、`batch_unique_ids`、`batch_dedup_ratio`、`gpu_cache_capacity`、`prefetch_depth`、`prefetch_window_live_ids`、`prefetch_window_live_bytes`、`prefetch_window_peak_live_ids`、`prefetch_window_peak_live_bytes`、`prefetch_window_live_cache_capacity_ratio`、`prefetch_window_peak_cache_capacity_ratio`。
- 正确性相关：`gpu_cache_clear_count`、`update_gpu_cache_invalidate_ms`、`planned_gpu_cache_prefill_fallbacks`、`planned_gpu_cache_prefill_wait_failures`、`planned_gpu_cache_prefill_result_size_mismatches`。

可信结果应同时满足：

- `prefetch_gpu_cache` lane 中 `planned_gpu_cache_prefill_successes > 0`，且 fallback/mismatch 不是主导。
- `lookup_gpu_cache_request_count > 0`，`lookup_gpu_cache_hit_count` 或 `lookup_gpu_cache_hit_rate` 能随 batch 复用提升；若始终为 0，需要检查 cache capacity、lookup bypass、数据复用率和 CUDA fast path。
- prefetch only 相比 baseline 应主要体现 `lookup_wait_ms` 或 `prefetch_wait_share_of_lookup` 下降；如果没有下降，瓶颈可能在 issue 队列驻留不足、server 端 prefetch 实际延迟或 batch 准备无法覆盖 wait。
- GPU cache only 如果 `lookup_gpu_cache_query_ms + lookup_gpu_cache_fill_ms` 高于节省的 backend lookup，则可能出现收益被 prefill/fill 开销抵消。
- update 后应看到 `update_gpu_cache_invalidate_ms` 或 `gpu_cache_clear_count` 增长；后续 lookup miss/refresh 是正确性信号，不应为了 hit rate 破坏 read-after-write。

根据 BagPipe/NestPipe 类工作，预取窗口实验不应只看 cache hit rate，还要判断通信是否被 dense 计算窗口隐藏。`dense_compute_ms = dense_fwd_ms + backward_ms + optimizer_ms`，`prefetch_network_wait_ms` 记录消费点暴露出的预取等待，包含普通 wait 路径的 `lookup_wait_ms` 和 prefill-to-cache 路径的 `planned_gpu_cache_prefill_wait_ms`；`prefetch_exposed_network_ms = max(0, prefetch_network_wait_ms - dense_compute_ms)`。如果增加 depth 只增加 `prefetch_window_peak_live_bytes`，但 `prefetch_exposed_network_ms` 和 `step_total_ms` 不下降，说明窗口已经过大或 cache/prefill 开销抵消收益。

查找最佳预取窗口和 GPU cache 容量可运行：

```bash
python3 model_zoo/rs_demo/run_prefetch_window_sweep.py \
  --depths 0,1,2,4,8 \
  --gpu-cache-capacities 0,4096,8192,16384 \
  --steps 30 \
  --warmup-steps 3 \
  --batch-size 512 \
  --num-embeddings 20000 \
  --embedding-dim 128 \
  --output-root /tmp/recstore_prefetch_window_sweep
```

输出汇总：`<output_root>/prefetch_window_sweep_summary.csv`。推荐窗口必须同时满足：

- `step_total_ms_mean` 在可比组合中最低或处于平台期。
- `prefetch_dense_cover_ratio_mean` 接近 1，或 `prefetch_exposed_network_ms_mean` 明显低于小窗口。
- `prefetch_window_peak_live_bytes_mean` 不超过 GPU cache 容量可接受范围，`prefetch_window_peak_cache_capacity_ratio_mean` 不长期大于 1。
- `planned_gpu_cache_prefill_fallbacks_mean` 和 `planned_gpu_cache_prefill_result_size_mismatches_mean` 不主导。
- update 后 cache invalidation/clear 指标正常，不能为了命中率牺牲 read-after-write。

单机单进程 TorchRec UVM caching（embedding 主存放在 host/UVM，GPU 侧使用 TorchRec/FBGEMM cache）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 1 \
  --node-rank 0 \
  --nproc-per-node 1 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  --rdzv-id rs-demo-uvm \
  --run-id rs-demo-uvm \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --torchrec-memory-mode uvm_caching \
  --no-start-server
```

该 lane 依赖当前环境中的 TorchRec/FBGEMM 支持 `FUSED_UVM_CACHING`。它更接近 TorchRec 原生的 DRAM/UVM 路径，但不是纯 CPU gather + GPU copy 的 staging baseline。报告时应和默认 `hbm` lane 分开标注资源模型。

双机手工启动 distributed TorchRec：

机器 A（`node-rank 0`）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 2 \
  --node-rank 0 \
  --nproc-per-node 4 \
  --master-addr <machine-a-ip> \
  --master-port 29500 \
  --rdzv-id rs-demo-2node \
  --run-id rs-demo-2node \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server
```

双机公平对齐 lane（单 trainer + 远端 embedding worker）：

机器 A（`node-rank 0`，trainer）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 2 \
  --node-rank 0 \
  --nproc-per-node 1 \
  --master-addr <machine-a-ip> \
  --master-port 29500 \
  --rdzv-id rs-demo-fair \
  --run-id rs-demo-fair \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --torchrec-dist-mode fair_remote \
  --no-start-server
```

机器 B（`node-rank 1`，embedding worker）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 2 \
  --node-rank 1 \
  --nproc-per-node 1 \
  --master-addr <machine-a-ip> \
  --master-port 29500 \
  --rdzv-id rs-demo-fair \
  --run-id rs-demo-fair \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --torchrec-dist-mode fair_remote \
  --no-start-server
```

该 lane 会让非 `rank0` 只保留 embedding worker 角色；主 `torchrec_main.csv` 只汇总 trainer 行，便于和单 trainer 的远端 RecStore lane 做公平主结论对比。
为保证 sparse update 语义正确，`fair_remote` 会让所有 rank 使用相同 batch 顺序；它不是多 trainer 吞吐测试语义。

机器 B（`node-rank 1`）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 2 \
  --node-rank 1 \
  --nproc-per-node 4 \
  --master-addr <machine-a-ip> \
  --master-port 29500 \
  --rdzv-id rs-demo-2node \
  --run-id rs-demo-2node \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server
```

如需 profiler trace（每次运行会在 trace dir 下生成多个 trace 文件，并聚合到 trace csv）：

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 1 \
  --node-rank 0 \
  --nproc-per-node 4 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  --rdzv-id rs-demo-profiler \
  --run-id rs-demo-profiler \
  --output-root /nas/home/shq/docker/rs_demo \
  --steps 60 \
  --batch-size 4096 \
  --no-start-server \
  --torchrec-profiler
```

## 3. 常用参数

- `--num-embeddings`：表大小
- `--embedding-dim`：向量维度
- `--batch-size`：每步 keys 数
- `--steps`：总迭代数
- `--warmup-steps`：预热步数（不计入脚本内 read/update 统计）
- `--output-root`：输出根目录，默认 `/nas/home/shq/docker/rs_demo`
- `--run-id`：本次运行标识；未指定时自动生成
- `--data-dir`：DLRM processed day0 目录（默认 `model_zoo/torchrec_dlrm/processed_day_0_data`）
- `--fuse-k`：与 DLRM 相同的融合位移参数（默认 `30`）
- `--read-before-update/--no-read-before-update`：是否每步先读后更
  - 开启时：读路径采用 `prefetch/wait`，并统计 `emb_read` 耗时
- `--start-server/--no-start-server`：是否自动起停 `ps_server`
- `--server-port0/--server-port1`：server 端口（默认读取 `recstore_config.json`）
- `--allocator`：value 内存管理器（默认 `R2ShmMalloc`，更适合压测）
- `--nnodes`：TorchRec 分布式节点数
- `--node-rank`：当前节点编号
- `--nproc-per-node`：每个节点的进程数
- `--master-addr`：TorchRec rendezvous master 地址
- `--master-port`：TorchRec rendezvous master 端口
- `--rdzv-backend`：TorchRec rendezvous backend，默认 `c10d`
- `--rdzv-id`：TorchRec rendezvous 标识；多机手工启动时两端必须一致
- `--torchrec-main-csv`：TorchRec 主报表 CSV 路径
- `--torchrec-main-agg-csv`：TorchRec 主报表聚合 CSV 路径（mean/p50/p95/max）
- `--torchrec-profiler`：启用 Torch profiler 并导出 trace 聚合 CSV
- `--torchrec-dist-mode`：TorchRec distributed 运行语义，默认 `replicated`
  - `replicated`：保留当前 distributed training 观测语义
  - `fair_remote`：单 trainer + 远端 embedding worker 的公平对齐语义
  - `fair_remote` 要求 `world_size > 1`
- `--torchrec-memory-mode`：TorchRec embedding 内存模式，默认 `hbm`；`uvm_caching` 使用 TorchRec/FBGEMM fused UVM caching，需要对应依赖支持
- `--torchrec-trace-dir`：Torch profiler trace 输出目录
- `--torchrec-trace-csv`：Torch profiler trace 聚合 CSV 路径
- `--torchrec-compare-recstore-csv`：可选，指定 RecStore CSV 以导出对照差值表
- `--torchrec-compare-csv`：RecStore vs TorchRec 对照差值 CSV 路径

## 4. 结果文件

- RecStore JSONL：`<output_root>/outputs/<run_id>/recstore_events.jsonl`
- RecStore CSV：`<output_root>/outputs/<run_id>/recstore_embupdate.csv`
- Server 日志：`<output_root>/logs/<run_id>/ps_server.log`
- TorchRec rank CSV：`<output_root>/outputs/<run_id>/torchrec_ranks/rank*.csv`
- TorchRec 主报表 CSV：`<output_root>/outputs/<run_id>/torchrec_main.csv`
- TorchRec 主报表聚合 CSV：`<output_root>/outputs/<run_id>/torchrec_main_agg.csv`
- TorchRec profiler trace 目录：`<output_root>/outputs/<run_id>/torchrec_traces`
- TorchRec profiler trace CSV：`<output_root>/outputs/<run_id>/torchrec_trace.csv`
- RecStore vs TorchRec 对照 CSV：`<output_root>/outputs/<run_id>/recstore_torchrec_compare.csv`
- Runtime 配置与 KV 数据：`<output_root>/runtime/<run_id>/...`

TorchRec 主报表（`--torchrec-main-csv`）关键列：

- `embed_transport_ms`：用于和远端 RecStore lane 对齐的归一 transport 列；当前等于 `collective_total_ms`
- `collective_total_ms`：collective launch + wait 的总耗时
- `kv_local_only_ms`：本地 embedding lookup + pool 的耗时（不含 pack/unpack）
- `kv_extended_ms`：输入打包 + 本地 lookup/pool + 输出解包的总耗时
- `network_proxy_torchrec_extended_ms`：`collective_total + input_pack + output_unpack` 的扩展通信代理项

TorchRec 主报表聚合 CSV（`--torchrec-main-agg-csv`）会对每个 `*_ms` 列导出：

- `*_mean`
- `*_p50`
- `*_p95`
- `*_max`

对照差值 CSV（`--torchrec-compare-csv`）默认导出以下口径：

- `network_main`：`RecStore(network_transport)` vs `TorchRec(collective_total)`
- `network_extended`：`RecStore(network_transport)` vs `TorchRec(collective + pack + unpack)`
- `kv_strict`：`RecStore(storage_backend_update)` vs `TorchRec(kv_local_only)`
- `server_vs_extended`：`RecStore(server_total)` vs `TorchRec(kv_extended)`

`dist_mode=single_node` 表示当前为单机 distributed；`dist_mode=multi_node` 表示当前为多机 distributed。
