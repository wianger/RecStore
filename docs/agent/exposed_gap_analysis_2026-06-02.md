# RecStore vs TorchRec HBM 暴露时间分析, 2026-06-02

## 目标

本轮回答两个问题：

1. 是否能像 BagPipe/NestPipe 类论文一样直接统计暴露时间。
2. 在 RecStore 开启 lookahead + GPU cache 后，相比 TorchRec HBM 显存直读，每个环节的差距在哪里。

## 方法

新增脚手架：

```bash
python3 model_zoo/rs_demo/analyze_torchrec_gap.py \
  --recstore-csv <recstore_main.csv> \
  --torchrec-csv <torchrec_main.csv> \
  --output-csv <exposed_gap.csv> \
  --output-md <exposed_gap.md>
```

暴露时间定义：

```text
prefetch_network_exposed_ms = max(0, prefetch_network_wait_ms - dense_compute_ms)
dense_compute_ms = dense_fwd_ms + backward_ms + optimizer_ms
```

该脚手架同时输出 raw time 和 exposed time。raw time 用于看系统实际付出的成本；exposed time 用于判断它是否真的暴露到 step 上。

## Smoke 设置

RecStore:

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend recstore \
  --steps 12 \
  --warmup-steps 2 \
  --batch-size 256 \
  --num-embeddings 10000 \
  --embedding-dim 128 \
  --read-mode prefetch \
  --prefetch-depth 1 \
  --enable-gpu-cache \
  --gpu-cache-capacity 16384 \
  --disable-gpu-cache-lookup-bypass \
  --run-id recstore-gap-prefetch-gpucache \
  --output-root /tmp/recstore_exposed_gap_smoke
```

TorchRec HBM:

```bash
python3 model_zoo/rs_demo/run_mock_stress.py \
  --backend torchrec \
  --nnodes 1 \
  --node-rank 0 \
  --nproc-per-node 1 \
  --master-addr 127.0.0.1 \
  --master-port 29500 \
  --rdzv-id exposed-gap-smoke \
  --steps 12 \
  --warmup-steps 2 \
  --batch-size 256 \
  --num-embeddings 10000 \
  --embedding-dim 128 \
  --torchrec-memory-mode hbm \
  --no-start-server \
  --run-id torchrec-gap-hbm \
  --output-root /tmp/recstore_exposed_gap_smoke
```

Gap:

```bash
python3 model_zoo/rs_demo/analyze_torchrec_gap.py \
  --recstore-csv /tmp/recstore_exposed_gap_smoke/outputs/recstore-gap-prefetch-gpucache/recstore_main.csv \
  --torchrec-csv /tmp/recstore_exposed_gap_smoke/outputs/torchrec-gap-hbm/torchrec_main.csv \
  --output-csv /tmp/recstore_exposed_gap_smoke/exposed_gap.csv \
  --output-md /tmp/recstore_exposed_gap_smoke/exposed_gap.md
```

## 结果

| metric | RecStore raw ms | RecStore exposed ms | TorchRec raw ms | TorchRec exposed ms | delta raw ms | delta exposed ms | bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| step_total | 19.144973 | 19.144973 | 15.071686 | 15.071686 | 4.073286 | 4.073286 | exposed |
| embedding_stage | 3.958518 | 3.958518 | 2.496853 | 2.496853 | 1.461665 | 1.461665 | exposed |
| embedding_lookup | 2.927056 | 2.927056 | 1.413389 | 1.413389 | 1.513667 | 1.513667 | exposed |
| prefetch_network | 1.442617 | 0.000000 | 0.000000 | 0.000000 | 1.442617 | 0.000000 | raw_only |
| gpu_cache_query | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | hidden |
| gpu_cache_fill | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | hidden |
| gpu_cache_prefill | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | hidden |
| sparse_update | 12.116674 | 12.116674 | 5.151055 | 5.151055 | 6.965620 | 6.965620 | exposed |
| gpu_cache_invalidate | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | hidden |
| dense_compute | 3.414472 | 3.414472 | 3.521343 | 3.521343 | -0.106870 | -0.106870 | exposed |

## 解释

本次 smoke 不能证明 GPU cache 已经追平 TorchRec HBM。相反，gap 表说明：

- `prefetch_network` raw wait 为 `1.442617 ms`，但 exposed 为 `0`，说明这部分等待已被 dense compute 覆盖。继续增加 lookahead 深度不会直接减少 step，只会增加窗口足迹。
- `embedding_lookup` 仍比 TorchRec HBM 慢 `1.513667 ms`。在网络等待已隐藏时，剩余读路径差距来自 RecStore lookup 本地路径、Python/Tensor 包装、pooling 或 cache lane 是否真的启用。
- 本次 `gpu_cache_query/prefill/invalidate` 为 0，不能解读为 GPU cache 免费；需要结合 `planned_gpu_cache_prefill_successes/fallback/no_cuda/no_api/mismatch`。如果 success 为 0，这就是 fallback/未启用路径，不是 GPU cache 正常快路径。
- 最大端到端差距来自 `sparse_update`，RecStore 比 TorchRec HBM 多 `6.965620 ms`。GPU cache 主要优化读路径，无法单独消除 sparse update 的暴露成本。

因此，当前更准确的结论是：lookahead 已经把本地 smoke 的预取网络等待隐藏掉，但 RecStore 与 TorchRec HBM 的剩余 gap 主要在 sparse update 和 embedding lookup raw path，而不是 exposed prefetch wait。

## 后续定位工具

CSV 级 exposed gap 用于先定位大类瓶颈。下一步建议：

- PyTorch profiler：使用已有 `--torchrec-profiler` 生成 trace 和 `torchrec_trace.csv`，定位 PyTorch operator 与 CPU/CUDA activity。
- Nsight Systems：用 `nsys profile --trace=cuda,nvtx,osrt -o <report> <python command>` 包住同一条 RecStore/TorchRec 命令，确认 CUDA kernel、memcpy、runtime API 和 CPU 调度是否真正 overlap。

## 限制

本次只是 `steps=12`、`batch_size=256` 的 smoke。由于 TorchRec 和 RecStore 都有明显 step 波动，论文级结论需要同参数 repeat、固定 GPU/CPU 亲和性，并在 CUDA GPU-cache prefill success 非零的环境中复测。
