---
name: performance-analysis
description: Use when planning, running, stopping, aggregating, or reporting RecStore performance experiments, especially paper-oriented RecStore vs TorchRec end-to-end, RecStore-RDMA, PS/network, storage-only, SVG/LaTeX reports, or benchmark verification.
---

# Performance Analysis

Use this skill from the RecStore repository root. Reply in Chinese. Keep benchmark layers separate and do not run benchmark jobs concurrently.

## First Checks

1. Stop or audit active experiments before starting new work:

```bash
ps -eo pid,ppid,etime,stat,cmd | rg 'run_paper_e2e|run_mock_stress|run_benchmark_ps|ps_transport_benchmark|rdma_rc_transport_benchmark|ps_server|petps_server'
```

If any benchmark or server process is still running, wait for it to finish or terminate it only when the user asked to stop experiments. Do not start another benchmark while one is active.

2. Record repository and environment state:

```bash
git status --short
git branch --show-current
nvidia-smi
ls /dev/infiniband/uverbs*
```

If RDMA verbs or enough GPUs are missing, record `skipped` instead of fabricating throughput. Single-GPU hosts cannot prove multi-GPU scalability.

## Layer Boundaries

- `storage-only`: direct KVEngine/YCSB/backend measurements. Use `tools/benchmarks/run_ycsb_compare.py`.
- `PS/network`: RecStore parameter-server transport path. Use `tools/benchmarks/run_benchmark_ps.py` or RDMA RC transport scripts.
- `PyTorch/model`: end-to-end model loop through `model_zoo/rs_demo`. Use `tools/benchmarks/run_paper_e2e.py`.

Never convert one layer into another. RDMA PS throughput can explain a bottleneck, but it is not end-to-end training throughput unless measured through `model_zoo/rs_demo` with an RDMA lane.

## Paper E2E Runner

Main script:

```text
tools/benchmarks/run_paper_e2e.py
```

Important lanes:

- `torchrec-hbm-1p`: TorchRec HBM baseline.
- `torchrec-uvm-1p`: TorchRec UVM cache baseline.
- `recstore-rdma-pet-1p`: RecStore-RDMA PET main innovation path.
- `recstore-rdma-eh-1p`, `recstore-rdma-map-1p`: RDMA backend ablations, enabled with `--include-ablation-lanes`.
- `recstore-brpc-*`, `recstore-grpc-*`, `recstore-local-shm-*`: non-RDMA transport and fast-path ablations.

Key outputs:

- `manifest.csv`: every attempted, skipped, failed, or successful run.
- `summary_e2e.csv`: PyTorch/model metrics, including `samples_per_sec`, `mean_step_total_ms`, `p95_step_total_ms`, `lookup_mrows_per_sec`, and `update_mrows_per_sec`.
- `summary_gap.csv`: paired RecStore/TorchRec ratios.
- `summary_ps_network.csv`: PS/network rows only.
- `figures/*.svg`: scenario-specific plots.
- `paper_e2e_report.tex`: Chinese LaTeX report embedding the SVG figures.
- `metadata.json`: git, GPU, RDMA, workload, and lane metadata.

## End-to-End RecStore-RDMA

Use this when the user asks for the main RDMA innovation point:

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

Use `--skip-rdma-ps` when the goal is pure PyTorch/model E2E and PS/network calibration is already available. Run TorchRec and RecStore-RDMA in separate roots when long jobs need tighter failure isolation, then combine roots.

## Aggregation And Report Regeneration

Combine existing runs without launching new benchmarks:

```bash
python3 tools/benchmarks/run_paper_e2e.py \
  --profile full \
  --output-root <combined_report_root> \
  --input-file /nas/home/shq/RecStore_/model_zoo/torchrec_dlrm/partial_data/day_0.bak \
  --combine-roots <root_a> <root_b> <root_c> \
  --aggregate-only \
  --skip-rdma-ps
```

Regenerate summaries and figures from one existing root:

```bash
python3 tools/benchmarks/run_paper_e2e.py \
  --profile full \
  --output-root <existing_root> \
  --aggregate-only \
  --skip-rdma-ps
```

Prefer separate SVG plots by question:

- E2E absolute throughput: `e2e_batch.svg`, `e2e_capacity.svg`, `e2e_dim.svg`.
- RDMA E2E: `e2e_rdma_batch.svg`, `e2e_rdma_capacity.svg`.
- Relative speedup: `speedup_batch.svg`, `speedup_capacity.svg`.
- RDMA coverage and PS/network: `rdma_failure_capacity.svg`, `rdma_ps_clients.svg`.

Do not replace these with one wide table. If LaTeX cannot include SVG, convert `figures/*.svg` to PDF and update the paths.

## PS/Network Calibration

For RecStore PS transport and RDMA GET calibration, use the `benchmark-ps` skill and `tools/benchmarks/run_benchmark_ps.py`. Keep RPC and RDMA runs separate unless client thread settings are valid for every transport. For RDMA, usually scale with client processes and keep per-process client threads at `1`.

Use PS/network results to explain RecStore-RDMA behavior, not as a replacement for `recstore-rdma-*` PyTorch/model lanes.

## Storage-Only

For KVEngine, YCSB, and storage-only batch lookup, use the `kvengine-ycsb` skill. When aligning storage limits with PS RDMA GET, use `read_mode=batch_get_flat --batch-keys 500`. Label operation mismatches explicitly.

## Verification

Before claiming script or report changes are complete, run the narrowest relevant checks:

```bash
python3 -m unittest model_zoo.rs_demo.tests.test_paper_e2e_benchmark -v
python3 -m unittest src.test.scripts.test_run_benchmark_ps -v
```

For RDMA code-path changes, also use the `rdma-module` skill and run:

```bash
ctest --test-dir build -R 'test_rdmaps_client_adapter|test_allshards_ps_client|petps_hashed_value_transfer_test' -VV
```

Report tests that could not run and why. Do not say a benchmark succeeded unless the command completed and its expected CSV exists.

## Reporting Rules

- Always report exact command, output root, data rows, batch size, embedding rows, embedding dim, warmup, measured steps, repeat count, lane list, GPU count, RDMA availability, and failed/skipped counts.
- Use median and CV across repeats for paper claims; keep raw per-run rows in `summary_e2e.csv`.
- Treat OOM, timeout, startup failure, missing output, and hardware skip as results. Do not drop them from the denominator silently.
- For RecStore vs TorchRec claims, state whether the comparison is against HBM or UVM cache and whether it is absolute throughput or speedup.
- For RecStore-RDMA, include both E2E plots and PS/network calibration only when both layers were actually measured.
