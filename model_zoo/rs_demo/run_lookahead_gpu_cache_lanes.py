from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


LANES = (
    ("baseline", 0, False),
    ("prefetch_only", 2, False),
    ("gpu_cache_only", 0, True),
    ("prefetch_gpu_cache", 2, True),
)

SUMMARY_FIELDS = (
    "step_total_ms",
    "samples_per_sec",
    "batches_per_sec",
    "embed_lookup_local_ms",
    "lookup_total_ms",
    "lookup_wait_ms",
    "lookup_fallback_pull_ms",
    "prefetch_issue_ms",
    "prefetch_queue_residence_ms",
    "prefetch_issue_to_consume_ms",
    "prefetch_wait_share_of_lookup",
    "lookup_gpu_cache_request_count",
    "lookup_gpu_cache_hit_count",
    "lookup_gpu_cache_miss_count",
    "lookup_gpu_cache_hit_rate",
    "lookup_gpu_cache_query_ms",
    "lookup_gpu_cache_fill_ms",
    "gpu_cache_clear_count",
    "update_gpu_cache_invalidate_ms",
    "planned_gpu_cache_prefill_batches",
    "planned_gpu_cache_prefill_ids",
    "planned_gpu_cache_prefill_successes",
    "planned_gpu_cache_prefill_fallbacks",
    "planned_gpu_cache_prefill_wait_failures",
    "planned_gpu_cache_prefill_result_size_mismatches",
    "batch_raw_ids",
    "batch_unique_ids",
    "batch_dedup_ratio",
)


def _mean_numeric(rows: list[dict[str, str]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return sum(values) / len(values) if values else 0.0


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["lane", "run_id", "csv"] + [f"{field}_mean" for field in SUMMARY_FIELDS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _lane_cmd(args: argparse.Namespace, lane: str, depth: int, enable_gpu_cache: bool) -> tuple[list[str], str]:
    if depth > 0:
        depth = int(args.prefetch_depth)
    run_id = f"{args.run_id_prefix}-{lane}"
    cmd = [
        sys.executable,
        str(args.repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
        "--backend",
        "recstore",
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--batch-size",
        str(args.batch_size),
        "--num-embeddings",
        str(args.num_embeddings),
        "--embedding-dim",
        str(args.embedding_dim),
        "--read-mode",
        "prefetch" if depth > 0 or enable_gpu_cache else "direct",
        "--prefetch-depth",
        str(depth),
        "--run-id",
        run_id,
        "--output-root",
        str(args.output_root),
        "--ps-type",
        str(args.ps_type),
        "--ps-kv-backend",
        str(args.ps_kv_backend),
        "--recstore-index-type",
        str(args.recstore_index_type),
    ]
    if args.no_start_server:
        cmd.append("--no-start-server")
    if args.library_path:
        cmd.extend(["--library-path", str(args.library_path)])
    if args.data_dir:
        cmd.extend(["--data-dir", str(args.data_dir)])
    if enable_gpu_cache:
        cmd.extend(
            [
                "--enable-gpu-cache",
                "--gpu-cache-capacity",
                str(args.gpu_cache_capacity),
                "--disable-gpu-cache-lookup-bypass",
            ]
        )
    if lane == "gpu_cache_only":
        cmd.append("--no-read-before-update")
    return cmd, run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RecStore baseline/prefetch/GPU-cache/prefetch+GPU-cache lanes."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/recstore_lookahead_gpu_cache_lanes"))
    parser.add_argument("--run-id-prefix", type=str, default="lookahead-gpu-cache")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-embeddings", type=int, default=20000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--gpu-cache-capacity", type=int, default=8192)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--ps-type", type=str, default="LOCAL_SHM")
    parser.add_argument("--ps-kv-backend", type=str, default="recstore_dram")
    parser.add_argument("--recstore-index-type", type=str, default="DRAM_EXTENDIBLE_HASH")
    parser.add_argument("--library-path", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--no-start-server", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    summary_rows: list[dict[str, object]] = []
    for lane, depth, enable_gpu_cache in LANES:
        cmd, run_id = _lane_cmd(args, lane, depth, enable_gpu_cache)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(args.repo_root), check=True)
        csv_path = args.output_root / "outputs" / run_id / "recstore_main.csv"
        rows = _load_csv(csv_path)
        summary: dict[str, object] = {
            "lane": lane,
            "run_id": run_id,
            "csv": str(csv_path),
        }
        for field in SUMMARY_FIELDS:
            summary[f"{field}_mean"] = _mean_numeric(rows, field)
        summary_rows.append(summary)

    if not args.dry_run:
        summary_path = args.output_root / "lookahead_gpu_cache_lane_summary.csv"
        _write_summary(summary_path, summary_rows)
        print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
