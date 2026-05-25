---
name: kvengine-ycsb
description: Run RecStore KVEngine correctness and YCSB benchmark workflows. Use when Codex needs to validate src/test/test_kvengine.cpp, run tools/benchmarks/run_ycsb_compare.py, prompt for thread count, SSD benchmark path, and results directory, then generate a three-table Chinese summary.md with workload descriptions, Run throughput, and Load throughput.
---

# KVEngine YCSB

## Workflow

Use this skill from a RecStore checkout. Do not run helper scripts from this skill directory; call the project script directly.

1. Confirm the current directory is the RecStore repo root, or pass `--repo`.
2. Prompt the user for:
   - thread count (default = 16)
   - SSD root path for benchmark data (default = /mnt/nvme1n1_recstore/recstore)
   - result output directory (default = results/kvengine_ycsb_$(date +%m%d%H%M))
   - workloads to run (default = a b c)
   - repeat count for YCSB (default = 3, ask if user wants 1)
   - distributions to use (default = uniform and zipfian)
   - record-count (default=10M)
3. Run:
   - `cmake -S . -B build`
   - `cmake --build build --target test_kvengine -j`
   - `ctest -R '^test_kvengine$' -VV`
   - `cmake --build build --target benchmark_kv_engine -j`
   - `tools/benchmarks/run_ycsb_compare.py`
4. Save logs and CSV/SVG artifacts under the chosen result directory.
5. Write `summary.md` as exactly three report tables, with the benchmark hyperparameters recorded as Chinese prose under `Workload 说明` before the first table:
   - Workload description
   - Run throughput
   - Load throughput

## Command Template

Ask the user for `threads`, `ssd_root`, and `output_dir`. Use defaults only when the user accepts them.

```bash
cmake -S . -B build
cmake --build build --target test_kvengine -j
ctest -R '^test_kvengine$' -VV
cmake --build build --target benchmark_kv_engine -j
```

```bash
python3 tools/benchmarks/run_ycsb_compare.py \
  --output-dir <output_dir> \
  --workloads a b c \
  --distributions uniform \
  --record-count 10000000 \
  --runtime-seconds 3 \
  --threads <threads> \
  --load-threads <threads> \
  --repeat 1 \
  --value-size 128 \
  --read-mode get
```

`tools/benchmarks/run_ycsb_compare.py` currently uses `/mnt/nvme1n1_recstore/recstore` internally for SSD data. If the user provides a different SSD path, create a temporary symlink or patch the command wrapper only after making that choice explicit.

If the user asks for "3 次平均", pass `--repeat 3`; otherwise preserve the requested repeat count in the `summary.md` heading.

## Summary Format

Generate `<output_dir>/summary.md` from `kvengine_workload_summary.csv` after YCSB finishes. Keep only these three sections:

1. `Workload 说明`
2. `Run 吞吐（ops/s，...）`
3. `Load 吞吐（ops/s，...）`

Under `Workload 说明`, before the workload table, record the benchmark hyperparameters in Chinese prose. Include at least: `threads`, `load_threads`, `record_count`, `runtime_seconds`, `repeat`, `value_size`, `read_mode`, `distributions`, `workloads`, SSD root path, output directory, `ssd_io_backend`, `ssd_queue_depth`, and allocator choices that affect the benchmark.

Use `M` for values >= 1,000,000 and `K` for values >= 1,000. Include the `三 workload 平均` column only in the Run table.

## Reporting Rules

- Do not claim tests pass unless the script completed successfully.
- If `test_kvengine` fails, stop before YCSB and report the log path.
- If any YCSB row exits nonzero, still write `summary.md`, but state failures in the final response and point to `summary.csv`.
- Keep generated project-facing report text in Chinese.
