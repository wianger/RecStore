from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = (
    "step_total_ms",
    "samples_per_sec",
    "batches_per_sec",
    "emb_stage_ms",
    "embed_lookup_local_ms",
    "lookup_total_ms",
    "lookup_wait_ms",
    "lookup_local_lookup_ms",
    "lookup_gpu_cache_query_ms",
    "lookup_gpu_cache_hit_rate",
    "planned_gpu_cache_prefill_ms",
    "planned_gpu_cache_prefill_successes",
    "planned_gpu_cache_prefill_fallbacks",
    "sparse_update_ms",
)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    measured = [row for row in rows if row.get("warmup_excluded", "") != "1"]
    return measured or rows


def _mean(rows: list[dict[str, str]], field: str) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            vals.append(float(row.get(field, "")))
        except ValueError:
            continue
    return sum(vals) / len(vals) if vals else 0.0


def summarize(path: Path) -> dict[str, float]:
    rows = _load_rows(path)
    return {field: _mean(rows, field) for field in METRICS}


def pct_delta(current: float, base: float) -> float:
    if base == 0.0:
        return 0.0
    return (current - base) / base * 100.0


def write_report(
    *,
    output: Path,
    iteration: int,
    title: str,
    hypothesis: str,
    optimization: str,
    recstore_csv: Path,
    torchrec_csv: Path,
    previous_csv: Path | None,
    notes: str,
) -> None:
    recstore = summarize(recstore_csv)
    torchrec = summarize(torchrec_csv)
    previous = summarize(previous_csv) if previous_csv else None
    output.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Iteration {iteration}: {title}")
    lines.append("")
    lines.append("## 目标")
    lines.append(hypothesis)
    lines.append("")
    lines.append("## 优化内容")
    lines.append(optimization)
    lines.append("")
    lines.append("## 原始数据")
    lines.append(f"- RecStore CSV: `{recstore_csv}`")
    lines.append(f"- TorchRec CSV: `{torchrec_csv}`")
    if previous_csv:
        lines.append(f"- 上一轮 RecStore CSV: `{previous_csv}`")
    lines.append("")
    lines.append("## 性能结果")
    lines.append("")
    lines.append("| 指标 | RecStore | TorchRec | RecStore vs TorchRec | RecStore vs 上一轮 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for metric in METRICS:
        rec_val = recstore[metric]
        tr_val = torchrec[metric]
        vs_torchrec = pct_delta(rec_val, tr_val)
        if previous is None:
            vs_prev = 0.0
        else:
            vs_prev = pct_delta(rec_val, previous[metric])
        lines.append(
            f"| `{metric}` | {rec_val:.6f} | {tr_val:.6f} | {vs_torchrec:+.2f}% | {vs_prev:+.2f}% |"
        )
    lines.append("")
    lines.append("## 结果分析")
    lines.append(notes)
    lines.append("")
    rec_lookup = recstore.get("lookup_total_ms", 0.0)
    rec_update = recstore.get("sparse_update_ms", 0.0)
    tr_lookup = torchrec.get("embed_lookup_local_ms", 0.0)
    tr_update = torchrec.get("sparse_update_ms", 0.0)
    lines.append("## 对比结论")
    lines.append(
        f"- 相比 TorchRec，RecStore embedding lookup 为 {rec_lookup:.6f} ms，"
        f"TorchRec lookup 为 {tr_lookup:.6f} ms；差值 {rec_lookup - tr_lookup:+.6f} ms。"
    )
    lines.append(
        f"- 相比 TorchRec，RecStore sparse update 为 {rec_update:.6f} ms，"
        f"TorchRec sparse update 为 {tr_update:.6f} ms；差值 {rec_update - tr_update:+.6f} ms。"
    )
    lines.append(
        f"- RecStore cache/prefill 摘要：cache hit rate="
        f"{recstore.get('lookup_gpu_cache_hit_rate', 0.0):.6f}，"
        f"cache query={recstore.get('lookup_gpu_cache_query_ms', 0.0):.6f} ms，"
        f"prefill={recstore.get('planned_gpu_cache_prefill_ms', 0.0):.6f} ms，"
        f"prefill fallback={recstore.get('planned_gpu_cache_prefill_fallbacks', 0.0):.6f}。"
    )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write one RecStore/TorchRec performance iteration report.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--hypothesis", type=str, required=True)
    parser.add_argument("--optimization", type=str, required=True)
    parser.add_argument("--recstore-csv", type=Path, required=True)
    parser.add_argument("--torchrec-csv", type=Path, required=True)
    parser.add_argument("--previous-csv", type=Path, default=None)
    parser.add_argument("--notes", type=str, required=True)
    args = parser.parse_args()
    write_report(
        output=args.output,
        iteration=args.iteration,
        title=args.title,
        hypothesis=args.hypothesis,
        optimization=args.optimization,
        recstore_csv=args.recstore_csv,
        torchrec_csv=args.torchrec_csv,
        previous_csv=args.previous_csv,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
