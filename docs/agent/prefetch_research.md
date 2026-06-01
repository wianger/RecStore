# Prefetch Research Notes for RecStore Agents

## Paper Takeaways

BagPipe-style prefetching is not just "load the next batch into cache". The useful control variables are the future-access window, the communication latency exposed at the consumption point, the dense computation window that can hide this latency, and the cache footprint required to keep prefetched rows live.

NestPipe highlights the same boundary in a different system: communication should launch as early as correctness permits, computation should consume ready micro-batches without waiting for unrelated communication, and the exposed communication ratio matters more than raw communication time alone. Its frozen-window argument is important for RecStore because sparse updates must remain read-after-write visible; prefetch cannot use values made stale by an update that has already committed.

Herald frames cache usefulness around predictability and current cache snapshots. For the current RecStore rs_demo, the dataloader already exposes future sparse ids, so a practical first step is measuring future-window footprint and hit/miss behavior before attempting sample reordering or worker placement.

AdaEmbed and related embedding-cache work reinforce that cache capacity must be reported with the access pattern. A larger lookahead can improve overlap while also increasing live rows in GPU cache. A window is only useful when the saved wait time exceeds prefill/query/fill overhead.

## Mapping to Current RecStore

The current implementation keeps BSP-like sparse update semantics: each batch still runs lookup, dense forward/backward, optimizer, sparse update, and cache invalidate/clear in order. Lookahead prefetch may issue reads for future batches, but consumption still falls back to stable lookup paths if CUDA, GPU cache prefill API, wait result, or size validation fails.

The `model_zoo/rs_demo/runtime/prefetch.py` scheduler owns lookahead handles and reports:

- `prefetch_issue_to_consume_ms`: time between issuing a future read and attaching it to the consuming batch.
- `prefetch_network_wait_ms`: wait exposed when the consuming batch needs prefetched values.
- `prefetch_dense_compute_ms`: dense forward + backward + dense optimizer time available to hide communication.
- `prefetch_exposed_network_ms`: wait not covered by dense compute.
- `prefetch_window_live_ids` and `prefetch_window_peak_live_bytes`: current and peak live prefetched row footprint.

`run_prefetch_window_sweep.py` sweeps depth and GPU cache capacity. Use it to find a window that lowers `step_total_ms` without pushing fallback/mismatch counts up or requiring a cache footprint larger than the configured GPU cache.

## Evaluation Rules

Report every result as PyTorch/model-layer unless lower-layer PS or storage benchmarks are also run. Do not claim architecture-level wins from a single rs_demo lane.

A credible lookahead result should show:

- lower `lookup_wait_ms` or `prefetch_exposed_network_ms`,
- stable or lower `step_total_ms`,
- bounded `prefetch_window_peak_live_bytes`,
- nonzero GPU cache requests and hits in cache lanes,
- low fallback and result-size mismatch counts,
- visible update invalidation or clear behavior after sparse updates.

If `prefetch_dense_cover_ratio` is already 1 and step time does not improve with deeper windows, the best depth is the shallowest depth with equivalent step time and lower live bytes.
