# RecStore Prefetch Window Evaluation, 2026-06-02

## Setup

Layer: PyTorch/model.

Command:

```bash
python3 model_zoo/rs_demo/run_prefetch_window_sweep.py \
  --depths 0,1,2,4 \
  --gpu-cache-capacities 0,1024,4096 \
  --steps 12 \
  --warmup-steps 2 \
  --batch-size 256 \
  --num-embeddings 10000 \
  --embedding-dim 128 \
  --output-root /tmp/recstore_prefetch_window_eval
```

Raw summary: `/tmp/recstore_prefetch_window_eval/prefetch_window_sweep_summary.csv`.

## Result

| depth | cache rows | step ms | network wait ms | dense cover | peak live rows | peak live bytes | peak/cache | hit rate | prefill success | sparse update ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 9.318617 | 0.161234 | 1.000000 | 12646.4 | 6474956.8 | 0.000000 | 0.000000 | 0.000000 | 3.994451 |
| 2 | 0 | 9.760258 | 0.163311 | 1.000000 | 17971.2 | 9201254.4 | 0.000000 | 0.000000 | 0.000000 | 4.298937 |
| 4 | 0 | 9.904445 | 0.174939 | 1.000000 | 26624.0 | 13631488.0 | 0.000000 | 0.000000 | 0.000000 | 4.718257 |
| 2 | 4096 | 10.740164 | 0.151092 | 1.000000 | 17971.2 | 9201254.4 | 4.387500 | 1.000000 | 1.000000 | 4.783996 |
| 1 | 1024 | 11.384772 | 0.138323 | 1.000000 | 12646.4 | 6474956.8 | 12.350000 | 1.000000 | 1.000000 | 4.928504 |
| 2 | 1024 | 11.657181 | 0.159514 | 1.000000 | 17971.2 | 9201254.4 | 17.550000 | 1.000000 | 1.000000 | 4.561100 |
| 4 | 1024 | 12.264584 | 0.163477 | 1.000000 | 26624.0 | 13631488.0 | 26.000000 | 1.000000 | 1.000000 | 5.281583 |
| 1 | 4096 | 12.347457 | 0.147684 | 1.000000 | 12646.4 | 6474956.8 | 3.087500 | 1.000000 | 1.000000 | 4.758275 |
| 0 | 1024 | 12.739753 | 0.000000 | 1.000000 | 0.0 | 0.0 | 0.000000 | 0.000000 | 0.000000 | 4.188301 |
| 0 | 0 | 12.943851 | 1.525884 | 1.000000 | 0.0 | 0.0 | 0.000000 | 0.000000 | 0.000000 | 5.158361 |
| 4 | 4096 | 14.739098 | 0.270492 | 1.000000 | 26624.0 | 13631488.0 | 6.500000 | 1.000000 | 1.000000 | 5.735783 |
| 0 | 4096 | 15.340609 | 0.000000 | 1.000000 | 0.0 | 0.0 | 0.000000 | 0.000000 | 0.000000 | 5.373417 |

## Interpretation

The best setting in this run is `prefetch_depth=1` with GPU cache disabled. It reduces mean step time from the no-prefetch baseline `12.943851 ms` to `9.318617 ms`. The measured network wait at consumption is only `0.161234 ms`, and `prefetch_dense_cover_ratio=1.0`, so the current dense compute window is already sufficient to hide the exposed prefetch wait at depth 1.

Increasing depth to 2 or 4 does not improve step time. It increases peak live prefetched rows from `12646.4` to `17971.2` and `26624.0`, which raises the live footprint from about `6.47 MB` to `9.20 MB` and `13.63 MB`. This matches the BagPipe/NestPipe-style expectation: once exposed communication is covered, a deeper window mostly increases cache pressure.

GPU cache lanes reach `lookup_gpu_cache_hit_rate=1.0` and `planned_gpu_cache_prefill_successes=1.0`, proving the prefetch-to-cache path is active. However, this run's cache capacity is smaller than the live window (`peak/cache` ranges from `3.0875` to `26.0`), and `lookup_gpu_cache_query_ms` plus prefill/update overhead offset the wait savings. The best GPU cache lane is `depth=2, cache=4096`, but its `10.740164 ms` step time is still slower than prefetch-only depth 1.

## Next Experiments

Run larger capacity sweeps where `prefetch_window_peak_cache_capacity_ratio <= 1`, for example capacities `16384,32768,65536` at batch size 256. Also repeat each point at least three times to separate sparse update variance from true prefetch-window effects.
