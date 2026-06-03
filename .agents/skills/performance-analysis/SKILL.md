---
name: performance-analysis
description: Use when coordinating RecStore paper-style performance analysis across existing storage, PS/RDMA, and TorchRec end-to-end results, especially when aggregating outputs, producing figures, or writing a Chinese benchmark report.
---

# Performance Analysis

Top-level coordinator for paper-style performance analysis. Delegate details to:

- `kvengine-ycsb`: storage-only KVEngine/YCSB and storage reports.
- `benchmark-ps`: PS/network transport benchmarks.
- `rdma-module`: RDMA correctness, RDMA code paths, and RDMA bottleneck diagnosis.

This skill only covers cross-layer organization, TorchRec E2E reporting, and final artifact structure.

## Rules

- Reply in Chinese; keep project reports in Chinese.
- Do not run benchmark jobs concurrently.
- Keep layers separate: `storage-only`, `PS/network`, and `PyTorch/model`.
- Do not turn PS/network RDMA throughput into end-to-end training throughput. RecStore-RDMA must be measured through `model_zoo/rs_demo` lanes to support E2E claims.
- Preserve failed, skipped, OOM, timeout, startup failure, and missing-output rows.

## E2E Script

```bash
python3 tools/benchmarks/run_paper_e2e.py
```

- `--only-lanes`: choose TorchRec/RecStore lanes.
- `--include-ablation-lanes`: enable backend/transport/prefetch/RDMA ablations.
- `--aggregate-only`: regenerate summaries and report from existing outputs.
- `--combine-roots <roots...>`: merge existing roots.
- `--skip-rdma-ps`: skip PS/network calibration.

Main lanes: `torchrec-hbm-1p`, `torchrec-uvm-1p`, `recstore-rdma-pet-1p`, `recstore-rdma-eh-1p`, `recstore-rdma-map-1p`, `recstore-brpc-*`, `recstore-grpc-*`, `recstore-local-shm-*`.

## RecStore-RDMA E2E Template

```bash
python3 tools/benchmarks/run_paper_e2e.py \
  --profile smoke \
  --output-root <output_root> \
  --input-file /nas/home/shq/RecStore_/model_zoo/torchrec_dlrm/partial_data/day_0.bak \
  --data-rows 131072 \
  --batch-sizes 256,1024,4096 \
  --num-embeddings 200000,2000000,3000000 \
  --embedding-dims 128 \
  --steps 30 \
  --warmup-steps 5 \
  --repeat 3 \
  --include-ablation-lanes \
  --only-lanes torchrec-hbm-1p,torchrec-uvm-1p,recstore-rdma-pet-1p,recstore-rdma-eh-1p,recstore-rdma-map-1p \
  --skip-rdma-ps
```

For long jobs, run TorchRec-only and RecStore-RDMA-only roots separately, then combine roots.

## Result Files

Use the `run_paper_e2e.py` output directory as the authoritative result root.

- `manifest.csv`: one row per attempted run, including status, command, log path, and output CSV path.
- `summary_e2e.csv`: PyTorch/model summary rows for the completed matrix.
- `summary_gap.csv`: paired RecStore/TorchRec comparison rows when baseline pairs exist.
- `paper_e2e_report.tex`: Chinese report generated from the current result root.
- `metadata.json`: environment, workload, and lane metadata for the run.
- `logs/e2e/*.log`: per-run logs.
- `outputs/<run_id>/*_main.csv`: raw per-step metrics.

Keep storage-only, PS/network, and RDMA-diagnostic results in their own roots defined by `kvengine-ycsb`, `benchmark-ps`, and `rdma-module`. Do not rename those outputs into a new cross-layer format.
