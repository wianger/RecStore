from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from types import SimpleNamespace

from tools.benchmarks import run_hps_backend_compare as hps_compare

SUMMARY_FIELDS = [
    "comparison_group",
    "target_dram_fraction",
    "target_ssd_fraction",
    "configured_dram_capacity_bytes",
    "configured_ssd_capacity_bytes",
    "configured_high_watermark_ratio",
    *hps_compare.SUMMARY_FIELDS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RecStore TIERED_VALUE_STORE against HPS HashMap/RocksDB "
            "while sweeping target DRAM resident fractions."
        )
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-jobs", type=int, default=0)
    parser.add_argument(
        "--dram-fractions",
        nargs="+",
        type=float,
        default=[1.0, 0.75, 0.5, 0.25, 0.1],
        help=(
            "Target fraction of records that should fit in RecStore DRAM. "
            "The script compensates for the tiered high watermark."
        ),
    )
    parser.add_argument(
        "--include-hps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run hps_hash_map and hps_rocksdb endpoint baselines.",
    )
    parser.add_argument(
        "--include-hps-native-tiered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also sweep HugeCTR HPS native volatile+persistent DB storage.",
    )
    parser.add_argument("--mode", choices=["fetch", "insert", "mixed", "fetch_insert"], default="fetch")
    parser.add_argument("--read-ratio", type=int, default=100)
    parser.add_argument("--record-count", type=int, default=1_000_000)
    parser.add_argument("--runtime-seconds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--load-threads", type=int, default=0)
    parser.add_argument("--hps-rocksdb-load-threads", type=int, default=1)
    parser.add_argument("--hps-rocksdb-db-threads", type=int, default=1)
    parser.add_argument(
        "--hps-native-cache-missed-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass through to HPS VolatileDatabaseParams.cache_missed_embeddings.",
    )
    parser.add_argument(
        "--hps-native-overflow-policy",
        choices=["evict_random", "evict_least_used", "evict_oldest"],
        default="evict_random",
        help="Pass through to HPS VolatileDatabaseParams.overflow_policy.",
    )
    parser.add_argument(
        "--hps-native-overflow-resolution-target",
        type=float,
        default=0.8,
        help="Pass through to HPS VolatileDatabaseParams.overflow_resolution_target.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--value-size", type=int, default=512)
    parser.add_argument("--distribution", choices=["uniform", "zipfian"], default="uniform")
    parser.add_argument("--zipfian-alpha", type=float, default=0.9)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dram-allocator", default="PERSIST_LOOP_SLAB")
    parser.add_argument("--recstore-index-type", default="DRAM_EXTENDIBLE_HASH")
    parser.add_argument("--ssd-io-backend", default="IOURING")
    parser.add_argument("--ssd-queue-depth", type=int, default=512)
    parser.add_argument("--ssd-capacity-multiplier", type=float, default=2.0)
    parser.add_argument("--tiered-high-watermark-ratio", type=float, default=0.85)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.build_jobs < 0:
        raise ValueError("--build-jobs must be non-negative")
    if args.record_count <= 0:
        raise ValueError("--record-count must be positive")
    if args.value_size <= 0:
        raise ValueError("--value-size must be positive")
    if args.tiered_high_watermark_ratio <= 0.0 or args.tiered_high_watermark_ratio > 1.0:
        raise ValueError("--tiered-high-watermark-ratio must be in (0, 1]")
    if args.ssd_capacity_multiplier <= 0.0:
        raise ValueError("--ssd-capacity-multiplier must be positive")
    for fraction in args.dram_fractions:
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError("--dram-fractions values must be in [0, 1]")


def configured_dram_capacity_bytes(args: argparse.Namespace, dram_fraction: float) -> int:
    if dram_fraction <= 0.0:
        return 1
    value_bytes = args.record_count * args.value_size * dram_fraction
    capacity = int(value_bytes / args.tiered_high_watermark_ratio)
    return max(capacity, 1)


def configured_ssd_capacity_bytes(args: argparse.Namespace) -> int:
    slot_size = max(args.value_size + 8, 128)
    return max(int(args.record_count * slot_size * args.ssd_capacity_multiplier), 1)


def base_run_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        mode=args.mode,
        read_ratio=args.read_ratio,
        record_count=args.record_count,
        runtime_seconds=args.runtime_seconds,
        threads=args.threads,
        load_threads=args.load_threads,
        hps_rocksdb_load_threads=args.hps_rocksdb_load_threads,
        hps_rocksdb_db_threads=args.hps_rocksdb_db_threads,
        hps_native_dram_fraction=1.0,
        hps_native_cache_missed_embeddings=getattr(
            args, "hps_native_cache_missed_embeddings", False
        ),
        hps_native_overflow_policy=getattr(
            args, "hps_native_overflow_policy", "evict_random"
        ),
        hps_native_overflow_resolution_target=getattr(
            args, "hps_native_overflow_resolution_target", 0.8
        ),
        batch_size=args.batch_size,
        value_size=args.value_size,
        distribution=args.distribution,
        zipfian_alpha=args.zipfian_alpha,
        dram_allocator=args.dram_allocator,
        dram_capacity_bytes=0,
        ssd_io_backend=args.ssd_io_backend,
        ssd_queue_depth=args.ssd_queue_depth,
        ssd_capacity_bytes=0,
        tiered_high_watermark_ratio=0.0,
        extra_arg=list(args.extra_arg),
        keep_data=args.keep_data,
        output_dir=args.output_dir,
    )


def annotate(
    rows: list[dict[str, object]],
    *,
    comparison_group: str,
    dram_fraction: str,
    dram_capacity: str,
    ssd_capacity: str,
    high_watermark: str,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        annotated = {
            "comparison_group": comparison_group,
            "target_dram_fraction": dram_fraction,
            "target_ssd_fraction": "" if dram_fraction == "" else f"{1.0 - float(dram_fraction):.6f}",
            "configured_dram_capacity_bytes": dram_capacity,
            "configured_ssd_capacity_bytes": ssd_capacity,
            "configured_high_watermark_ratio": high_watermark,
        }
        annotated.update(row)
        out.append(annotated)
    return out


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_recstore_tiered(args: argparse.Namespace, repeat: int, dram_fraction: float) -> list[dict[str, object]]:
    run_args = base_run_args(args)
    dram_capacity = configured_dram_capacity_bytes(args, dram_fraction)
    ssd_capacity = configured_ssd_capacity_bytes(args)
    run_args.dram_capacity_bytes = dram_capacity
    run_args.ssd_capacity_bytes = ssd_capacity
    run_args.tiered_high_watermark_ratio = args.tiered_high_watermark_ratio

    alias = f"recstore_tiered_dram{dram_fraction:.6f}"
    spec = hps_compare.BackendSpec(
        "recstore",
        args.recstore_index_type,
        "TIERED_VALUE_STORE",
    )
    rows = hps_compare.run_one(alias, repeat, run_args, spec)
    return annotate(
        rows,
        comparison_group="recstore_tiered",
        dram_fraction=f"{dram_fraction:.6f}",
        dram_capacity=str(dram_capacity),
        ssd_capacity=str(ssd_capacity),
        high_watermark=f"{args.tiered_high_watermark_ratio:.6f}",
    )


def run_hps_baseline(args: argparse.Namespace, repeat: int, alias: str) -> list[dict[str, object]]:
    run_args = base_run_args(args)
    rows = hps_compare.run_one(alias, repeat, run_args)
    return annotate(
        rows,
        comparison_group=alias,
        dram_fraction="",
        dram_capacity="",
        ssd_capacity="",
        high_watermark="",
    )


def run_hps_native_tiered(
    args: argparse.Namespace, repeat: int, dram_fraction: float
) -> list[dict[str, object]]:
    run_args = base_run_args(args)
    run_args.hps_native_dram_fraction = dram_fraction

    alias = f"hps_native_tiered_dram{dram_fraction:.6f}"
    spec = hps_compare.BACKEND_ALIASES["hps_native_tiered"]
    rows = hps_compare.run_one(alias, repeat, run_args, spec)
    return annotate(
        rows,
        comparison_group="hps_native_tiered",
        dram_fraction=f"{dram_fraction:.6f}",
        dram_capacity="",
        ssd_capacity="",
        high_watermark="",
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.build:
        hps_compare.ensure_build(args.build_jobs or 1)
    if not hps_compare.BENCHMARK_BIN.exists():
        raise FileNotFoundError(f"{hps_compare.BENCHMARK_BIN} does not exist")

    rows: list[dict[str, object]] = []
    for repeat in range(args.repeat):
        if args.include_hps:
            for alias in ("hps_hash_map", "hps_rocksdb"):
                new_rows = run_hps_baseline(args, repeat, alias)
                rows.extend(new_rows)
                run_rows = [row for row in new_rows if row["phase"] == "run"]
                metric = run_rows[0]["throughput_keys_sec"] if run_rows else ""
                print(f"{alias} r{repeat}: exit={new_rows[0]['exit_code']} run_keys_sec={metric}")
                write_summary(args.output_dir / "tiered_hps_ratio_summary.csv", rows)

        if args.include_hps_native_tiered:
            for dram_fraction in args.dram_fractions:
                new_rows = run_hps_native_tiered(args, repeat, dram_fraction)
                rows.extend(new_rows)
                run_rows = [row for row in new_rows if row["phase"] == "run"]
                metric = run_rows[0]["throughput_keys_sec"] if run_rows else ""
                print(
                    f"hps_native_tiered dram_fraction={dram_fraction:.6f} "
                    f"r{repeat}: exit={new_rows[0]['exit_code']} "
                    f"run_keys_sec={metric}"
                )
                write_summary(args.output_dir / "tiered_hps_ratio_summary.csv", rows)

        for dram_fraction in args.dram_fractions:
            new_rows = run_recstore_tiered(args, repeat, dram_fraction)
            rows.extend(new_rows)
            run_rows = [row for row in new_rows if row["phase"] == "run"]
            metric = run_rows[0]["throughput_keys_sec"] if run_rows else ""
            print(
                f"recstore_tiered dram_fraction={dram_fraction:.6f} "
                f"r{repeat}: exit={new_rows[0]['exit_code']} run_keys_sec={metric}"
            )
            write_summary(args.output_dir / "tiered_hps_ratio_summary.csv", rows)

    write_summary(args.output_dir / "tiered_hps_ratio_summary.csv", rows)
    if not args.keep_data:
        shutil.rmtree(args.output_dir / "data", ignore_errors=True)
    return 0 if all(int(row["exit_code"]) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
