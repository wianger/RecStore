from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterable

from ..common import SPARSE_FEATURES_PER_SAMPLE, _read_csv
from .config import BenchmarkConfig, infer_client_deployment, infer_ps_deployment


def _warm_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    return [row for row in rows if str(row.get("warmup_excluded", "0")) not in {"1", "true", "True"}]


def _mean(rows: Iterable[dict[str, str]], column: str) -> float:
    vals = [float(row[column]) for row in rows if row.get(column, "") not in {"", "nan", "NaN"}]
    return statistics.fmean(vals) if vals else 0.0


def _p95(rows: Iterable[dict[str, str]], column: str) -> float:
    vals = sorted(float(row[column]) for row in rows if row.get(column, "") not in {"", "nan", "NaN"})
    if not vals:
        return 0.0
    return vals[int(round((len(vals) - 1) * 0.95))]


def collect_summary_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in manifest:
        path = Path(str(item.get("main_csv", "")))
        if not path.exists():
            continue
        warm = _warm_rows(path)
        batch_size = int(item["batch_size"])
        mean_step = _mean(warm, "step_total_ms")
        mean_lookup = _mean(warm, "embed_lookup_local_ms")
        mean_update = _mean(warm, "sparse_update_ms")
        sparse_rows = batch_size * SPARSE_FEATURES_PER_SAMPLE
        out.append(
            {
                **item,
                "mean_step_total_ms": mean_step,
                "p95_step_total_ms": _p95(warm, "step_total_ms"),
                "mean_embed_lookup_ms": mean_lookup,
                "mean_sparse_update_ms": mean_update,
                "samples_per_sec": batch_size * 1000.0 / mean_step if mean_step > 0.0 else 0.0,
                "lookup_mrows_per_sec": sparse_rows / (mean_lookup / 1000.0) / 1e6
                if mean_lookup > 0.0
                else 0.0,
                "update_mrows_per_sec": sparse_rows / (mean_update / 1000.0) / 1e6
                if mean_update > 0.0
                else 0.0,
            }
        )
    return out


def _unit(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.3f}K"
    return f"{value:.3f}"


def _repeat_stats(rows: list[dict[str, Any]], metric: str) -> tuple[float, float, int]:
    vals = [float(row.get(metric, 0.0) or 0.0) for row in rows if float(row.get(metric, 0.0) or 0.0) > 0.0]
    if not vals:
        return 0.0, 0.0, 0
    mean = statistics.fmean(vals)
    cv = statistics.pstdev(vals) / mean if len(vals) >= 2 and mean > 0.0 else 0.0
    return mean, cv, len(vals)


def render_summary_md(cfg: BenchmarkConfig, rows: list[dict[str, Any]]) -> str:
    clients = "; ".join(
        f"{client.ip}/gpu{client.gpu_id}/rank{client.node_rank}/nproc{client.nproc_per_node}"
        for client in cfg.clients
    )
    servers = "; ".join(f"{server.ip}:{server.port}/shard{server.shard_id}" for server in cfg.servers)
    lines = [
        "# Benchmark E2E Summary",
        "",
        "## Workload 说明",
        "",
        (
            f"本次测试模型为 {cfg.model}，client 部署为 {infer_client_deployment(cfg.clients)}，"
            f"PS 部署为 {infer_ps_deployment(cfg.servers)}，client=[{clients}]，PS=[{servers}]。"
            f"batch_size={cfg.batch_size}，embedding_dim={cfg.embedding_dim}，"
            f"num_embeddings={cfg.num_embeddings}，steps={cfg.steps}，warmup_steps={cfg.warmup_steps}，"
            f"init_rows={cfg.init_rows}，"
            f"repeat={cfg.repeat}，read_mode={cfg.read_mode}，prefetch_depth={cfg.prefetch_depth}，"
            f"index_type={cfg.index_type}，TorchRec baseline={','.join(cfg.torchrec_baselines) or 'disabled'}，"
            f"dataset={cfg.dataset_path}，runtime={cfg.resolved_runtime_dir}，"
            f"output={cfg.output_dir}。"
        ),
        "",
        "| lane | backend | batch | dim | repeat_n | mean samples/s | CV |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("lane", row.get("transport", ""))),
            str(row.get("backend", "")),
            str(row.get("batch_size", "")),
            str(row.get("embedding_dim", "")),
        )
        grouped.setdefault(key, []).append(row)
    for key, group in sorted(grouped.items()):
        mean, cv, count = _repeat_stats(group, "samples_per_sec")
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {count} | {_unit(mean)} | {cv:.3f} |")
    if not rows:
        lines.append("| - | - | - | - | 0 | 0.000 | 0.000 |")

    lines.extend(
        [
            "",
            "## E2E 吞吐（samples/s，...）",
            "",
            "| run_id | lane | backend | samples/s | lookup M rows/s | update M rows/s |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {run_id} | {lane} | {backend} | {samples} | {lookup:.3f} | {update:.3f} |".format(
                run_id=row.get("run_id", ""),
                lane=row.get("lane", row.get("transport", "")),
                backend=row.get("backend", ""),
                samples=_unit(float(row.get("samples_per_sec", 0.0) or 0.0)),
                lookup=float(row.get("lookup_mrows_per_sec", 0.0) or 0.0),
                update=float(row.get("update_mrows_per_sec", 0.0) or 0.0),
            )
        )
    if not rows:
        lines.append("| - | - | - | 0.000 | 0.000 | 0.000 |")

    lines.extend(
        [
            "",
            "## E2E 延迟分解（ms，...）",
            "",
            "| run_id | lane | backend | mean step | p95 step | lookup | sparse update |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {run_id} | {lane} | {backend} | {mean:.3f} | {p95:.3f} | {lookup:.3f} | {update:.3f} |".format(
                run_id=row.get("run_id", ""),
                lane=row.get("lane", row.get("transport", "")),
                backend=row.get("backend", ""),
                mean=float(row.get("mean_step_total_ms", 0.0) or 0.0),
                p95=float(row.get("p95_step_total_ms", 0.0) or 0.0),
                lookup=float(row.get("mean_embed_lookup_ms", 0.0) or 0.0),
                update=float(row.get("mean_sparse_update_ms", 0.0) or 0.0),
            )
        )
    if not rows:
        lines.append("| - | - | - | 0.000 | 0.000 | 0.000 | 0.000 |")
    lines.append("")
    return "\n".join(lines)
