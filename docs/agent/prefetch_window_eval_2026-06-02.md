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

## Large Capacity Sweep

Command:

```bash
python3 model_zoo/rs_demo/run_prefetch_window_sweep.py \
  --depths 1,2,4 \
  --gpu-cache-capacities 8192,16384,32768 \
  --steps 12 \
  --warmup-steps 2 \
  --batch-size 256 \
  --num-embeddings 10000 \
  --embedding-dim 128 \
  --output-root /tmp/recstore_prefetch_window_capacity_eval
```

Raw summary: `/tmp/recstore_prefetch_window_capacity_eval/prefetch_window_sweep_summary.csv`.

| depth | cache rows | step ms | dense ms | prefill wait ms | network wait ms | dense cover | peak live rows | peak/cache | hit rate | query ms | prefill success | fallback | mismatch | sparse update ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8192 | 8.951713 | 2.941056 | 0.145927 | 0.145927 | 1.000000 | 12646.4 | 1.543750 | 1.000000 | 1.016809 | 1.000000 | 0.000000 | 0.000000 | 3.050682 |
| 1 | 16384 | 9.296622 | 2.791577 | 0.139726 | 0.139726 | 1.000000 | 12646.4 | 0.771875 | 1.000000 | 1.007790 | 1.000000 | 0.000000 | 0.000000 | 3.400767 |
| 4 | 8192 | 9.425468 | 2.776016 | 0.165456 | 0.165456 | 1.000000 | 26624.0 | 3.250000 | 1.000000 | 1.037305 | 1.000000 | 0.000000 | 0.000000 | 3.412031 |
| 2 | 32768 | 9.449564 | 2.765788 | 0.153851 | 0.153851 | 1.000000 | 17971.2 | 0.548438 | 1.000000 | 1.012629 | 1.000000 | 0.000000 | 0.000000 | 3.525222 |
| 2 | 16384 | 9.570394 | 2.907897 | 0.159468 | 0.159468 | 1.000000 | 17971.2 | 1.096875 | 1.000000 | 1.006593 | 1.000000 | 0.000000 | 0.000000 | 3.609736 |
| 4 | 32768 | 9.662194 | 2.780805 | 0.156996 | 0.156996 | 1.000000 | 26624.0 | 0.812500 | 1.000000 | 1.011624 | 1.000000 | 0.000000 | 0.000000 | 3.612245 |
| 1 | 32768 | 10.910065 | 2.861836 | 0.143640 | 0.143640 | 1.000000 | 12646.4 | 0.385938 | 1.000000 | 1.007268 | 1.000000 | 0.000000 | 0.000000 | 4.865243 |
| 2 | 8192 | 12.199883 | 3.588097 | 0.148952 | 0.148952 | 1.000000 | 17971.2 | 2.193750 | 1.000000 | 1.012098 | 1.000000 | 0.000000 | 0.000000 | 5.634383 |
| 4 | 16384 | 18.700816 | 5.961897 | 0.312888 | 0.312888 | 1.000000 | 26624.0 | 1.625000 | 1.000000 | 1.080717 | 1.000000 | 0.000000 | 0.000000 | 7.909860 |

The large-capacity run confirms two useful properties. First, prefetch-to-cache still succeeds across all tested capacities: `planned_gpu_cache_prefill_successes_mean=1.0`, fallback and result-size mismatch are zero, and lookup cache hit rate is 1.0. Second, cache capacity alone is not the bottleneck once the live window fits. `depth=1, cache=16384` and `depth=2/4, cache=32768` keep `prefetch_window_peak_cache_capacity_ratio_mean <= 1`, but they are not consistently faster than `depth=1, cache=8192`. The measured exposed network wait is already covered by dense compute (`prefetch_dense_cover_ratio_mean=1.0`), while query time is about `1.0 ms` per measured step and sparse update time varies from `3.05 ms` to `7.91 ms`.

The best point in this run is `prefetch_depth=1, gpu_cache_capacity=8192`, with `8.951713 ms` mean step time. It improves over the small-capacity prefetch-only best (`depth=1, cache=0`, `9.318617 ms`) but still carries `peak/cache=1.54375`, so it is a fast point rather than a clean capacity-safe recommendation. The cleaner capacity-safe candidates are `depth=1, cache=16384` (`9.296622 ms`, `peak/cache=0.771875`) and `depth=2, cache=32768` (`9.449564 ms`, `peak/cache=0.548438`). For a paper-style comparison, report both: the fastest observed lane and the best capacity-safe lane.

## Current Recommendation

For the current local PyTorch/model setup, use `prefetch_depth=1` as the default lookahead window. It gives the lowest or near-lowest step time in both sweeps, keeps live rows at about `12646` for this workload, and fully hides the measured prefetch network wait behind dense compute. Larger windows raise the live prefetch footprint to about `17971` rows at depth 2 and `26624` rows at depth 4 without reducing exposed wait.

GPU cache should be evaluated as a separate lane rather than assumed beneficial. The prefill path is now functionally active, but the current implementation still spends about `1 ms` in cache query and adds update/invalidate work; depending on sparse update variance, that overhead can offset the hidden wait. A trustworthy performance claim should therefore include all four lanes (`baseline`, `prefetch_only`, `gpu_cache_only`, `prefetch_gpu_cache`) plus at least one capacity-safe sweep point where `prefetch_window_peak_cache_capacity_ratio_mean <= 1`.

## Remaining Experiments

Repeat each point at least three times to separate sparse update variance from true prefetch-window effects. The next network-sensitive run should use a remote PS/RDMA setting because this local PyTorch/model setup already hides measured prefetch wait behind dense compute; a higher-latency path is needed to decide whether deeper windows help before the cache footprint dominates.
