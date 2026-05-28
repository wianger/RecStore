#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import TextIO

import numpy as np


DEFAULT_NUM_EMBEDDINGS_PER_FEATURE = [
    40000000,
    39060,
    17295,
    7424,
    20265,
    3,
    7122,
    1543,
    63,
    40000000,
    3067956,
    405282,
    10,
    2209,
    11938,
    155,
    4,
    976,
    14,
    40000000,
    40000000,
    40000000,
    590152,
    12973,
    108,
    36,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slice Criteo day_0 TSV data and materialize single-day numpy "
            "files without requiring torchrec's preprocessing module."
        )
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--start-line", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100_000)
    return parser.parse_args()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def parse_line(line: str, line_no: int) -> tuple[float, list[float], list[int]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 40:
        raise ValueError(f"line {line_no} has {len(parts)} columns, expected 40")

    label = float(int(parts[0]) if parts[0] else 0)
    dense = [float(int(value)) if value else 0.0 for value in parts[1:14]]
    sparse = [int(value, 16) if value else 0 for value in parts[14:40]]
    return label, dense, sparse


def unique_counts_per_feature(sparse: np.ndarray) -> list[int]:
    return [int(np.unique(sparse[:, idx]).size) for idx in range(sparse.shape[1])]


def modulo_unique_counts(sparse: np.ndarray) -> list[int]:
    counts: list[int] = []
    for idx, vocab in enumerate(DEFAULT_NUM_EMBEDDINGS_PER_FEATURE):
        vocab = min(int(vocab), 800_000)
        if vocab <= 0:
            counts.append(0)
            continue
        counts.append(int(np.unique(sparse[:, idx] % vocab).size))
    return counts


def main() -> int:
    args = parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    if args.start_line < 0:
        raise ValueError("--start-line must be non-negative")
    if not args.input_file.exists():
        raise FileNotFoundError(args.input_file)

    args.output_raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_output = args.output_raw_dir / "day_0"

    labels = np.empty((args.rows,), dtype=np.float32)
    dense = np.empty((args.rows, 13), dtype=np.float32)
    sparse = np.empty((args.rows, 26), dtype=np.int64)

    accepted = 0
    skipped_bad = 0
    with open_text(args.input_file) as src, raw_output.open(
        "w", encoding="utf-8"
    ) as raw_dst:
        for line_no, line in enumerate(src):
            if line_no < args.start_line:
                continue
            if accepted >= args.rows:
                break
            try:
                label, dense_row, sparse_row = parse_line(line, line_no + 1)
            except ValueError as exc:
                skipped_bad += 1
                if skipped_bad <= 5:
                    print(f"warning: {exc}", flush=True)
                continue
            labels[accepted] = label
            dense[accepted, :] = dense_row
            sparse[accepted, :] = sparse_row
            raw_dst.write(line)
            accepted += 1
            if accepted % args.progress_interval == 0:
                print(f"accepted {accepted} rows", flush=True)

    if accepted == 0:
        raise ValueError("no valid rows were read")
    labels = labels[:accepted]
    dense = dense[:accepted, :]
    sparse = sparse[:accepted, :]

    if not args.no_shuffle:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(accepted)
        labels = labels[order]
        dense = dense[order]
        sparse = sparse[order]

    np.save(args.output_dir / "day_0_dense.npy", dense)
    np.save(args.output_dir / "day_0_sparse.npy", sparse)
    np.save(args.output_dir / "day_0_labels.npy", labels)

    raw_unique_per_feature = unique_counts_per_feature(sparse)
    modulo_unique_per_feature = modulo_unique_counts(sparse)
    metadata = {
        "input_file": str(args.input_file),
        "raw_output": str(raw_output),
        "output_dir": str(args.output_dir),
        "requested_rows": args.rows,
        "accepted_rows": int(accepted),
        "start_line": args.start_line,
        "shuffle": not args.no_shuffle,
        "seed": args.seed,
        "skipped_bad_rows": int(skipped_bad),
        "dense_shape": list(dense.shape),
        "sparse_shape": list(sparse.shape),
        "labels_shape": list(labels.shape),
        "dense_dtype": str(dense.dtype),
        "sparse_dtype": str(sparse.dtype),
        "labels_dtype": str(labels.dtype),
        "raw_sparse_unique_total": int(np.unique(sparse).size),
        "raw_sparse_unique_per_feature": raw_unique_per_feature,
        "modulo_800k_sparse_unique_total_estimate": int(
            sum(modulo_unique_per_feature)
        ),
        "modulo_800k_sparse_unique_per_feature": modulo_unique_per_feature,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
