from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

from .common import FigureSection, FigureSpec, _to_float


def _median_ok_by_label(
    rows: list[dict[str, Any]],
    *,
    predicate: Callable[[dict[str, Any]], bool],
    x_key: str,
    y_key: str = "samples_per_sec",
) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        if row.get("status") != "ok" or not predicate(row):
            continue
        try:
            x_value = float(row.get(x_key, 0))
            y_value = float(row.get(y_key, 0))
        except (TypeError, ValueError):
            continue
        if x_value <= 0.0 or y_value <= 0.0:
            continue
        grouped.setdefault((str(row.get("label", "")), x_value), []).append(y_value)
    out: dict[str, list[tuple[float, float]]] = {}
    for (label, x_value), values in grouped.items():
        out.setdefault(label, []).append((x_value, statistics.median(values)))
    return {label: sorted(points) for label, points in out.items()}


def _row_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(str(row.get(key, str(default)) or str(default)))
    except ValueError:
        return default


def _best_slice_for_x(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    fixed_keys: tuple[str, ...],
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, int]:
    grouped: dict[tuple[int, ...], dict[str, set[float]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if predicate is not None and not predicate(row):
            continue
        try:
            x_value = float(row.get(x_key, 0))
        except (TypeError, ValueError):
            continue
        if x_value <= 0:
            continue
        key = tuple(_row_int(row, fixed_key) for fixed_key in fixed_keys)
        label = str(row.get("label", ""))
        grouped.setdefault(key, {}).setdefault(label, set()).add(x_value)
    if not grouped:
        return {}

    def score(item: tuple[tuple[int, ...], dict[str, set[float]]]) -> tuple[int, int, int]:
        key, by_label = item
        multi_point_labels = sum(1 for values in by_label.values() if len(values) >= 2)
        total_points = sum(len(values) for values in by_label.values())
        series_count = len(by_label)
        return (multi_point_labels, total_points, series_count)

    best_key, best_by_label = max(grouped.items(), key=score)
    if not any(len(values) >= 2 for values in best_by_label.values()):
        return {}
    return dict(zip(fixed_keys, best_key))


def _median_points(rows: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    grouped: dict[float, list[float]] = {}
    for x_value, y_value in rows:
        if x_value > 0 and y_value > 0:
            grouped.setdefault(float(x_value), []).append(float(y_value))
    return [(x_value, statistics.median(values)) for x_value, values in sorted(grouped.items())]


def _filter_multi_point_series(series: dict[str, list[tuple[float, float]]]) -> dict[str, list[tuple[float, float]]]:
    return {label: points for label, points in series.items() if len(points) >= 2}


def _primary_lane(row: dict[str, Any]) -> bool:
    label = str(row.get("label", ""))
    if label in {"TorchRec-HBM-1proc", "TorchRec-UVMCache-1proc"}:
        return True
    return label in {
        "RecStore-BRPC-PET-1proc",
        "RecStore-GRPC-PET-1proc",
        "RecStore-RDMA-PET-1proc",
    }


def _rdma_lane(row: dict[str, Any]) -> bool:
    return str(row.get("ps_type", "")).upper() == "RDMA"


def build_figure_specs(
    summary_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    ps_rows: list[dict[str, Any]],
) -> list[FigureSpec]:
    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    figures: list[FigureSpec] = []
    if ok_rows:
        single_proc = [
            row for row in ok_rows if _row_int(row, "nproc_per_node", 1) == 1
        ]
        batch_slice = _best_slice_for_x(
            single_proc,
            x_key="batch_size",
            fixed_keys=("rows", "num_embeddings", "embedding_dim"),
        )
        if batch_slice:
            fixed_rows = batch_slice["rows"]
            fixed_embeddings = batch_slice["num_embeddings"]
            fixed_dim = batch_slice["embedding_dim"]
            figures.append(
                FigureSpec(
                    filename="e2e_batch.svg",
                    title=f"Batch size 曲线: rows={fixed_rows}, emb={fixed_embeddings}, dim={fixed_dim}",
                    xlabel="Batch size",
                    ylabel="Samples/s",
                    series=_filter_multi_point_series(_median_ok_by_label(
                        single_proc,
                        predicate=lambda row: _primary_lane(row)
                        and _row_int(row, "rows") == fixed_rows
                        and _row_int(row, "num_embeddings") == fixed_embeddings
                        and _row_int(row, "embedding_dim") == fixed_dim,
                        x_key="batch_size",
                    )),
                    description="端到端主线：固定数据行数、embedding 容量和维度，只改变 batch size；用于观察 TorchRec 与 RecStore 主路径的训练吞吐分界。",
                )
            )
        capacity_slice = _best_slice_for_x(
            single_proc,
            x_key="num_embeddings",
            fixed_keys=("rows", "batch_size", "embedding_dim"),
        )
        if capacity_slice:
            fixed_rows = capacity_slice["rows"]
            fixed_batch = capacity_slice["batch_size"]
            fixed_dim = capacity_slice["embedding_dim"]
            figures.append(
                FigureSpec(
                    filename="e2e_capacity.svg",
                    title=f"Embedding capacity 曲线: rows={fixed_rows}, batch={fixed_batch}, dim={fixed_dim}",
                    xlabel="Embedding rows per table cap",
                    ylabel="Samples/s",
                    series=_filter_multi_point_series(_median_ok_by_label(
                        single_proc,
                        predicate=lambda row: _primary_lane(row)
                        and _row_int(row, "rows") == fixed_rows
                        and _row_int(row, "batch_size") == fixed_batch
                        and _row_int(row, "embedding_dim") == fixed_dim,
                        x_key="num_embeddings",
                    )),
                    xmode_log=True,
                    description="容量敏感性：固定 batch 和 dim，横轴为每张 embedding table 的容量上限；用于展示 HBM/UVM 与 RecStore 参数存储在大容量下的差异。",
                )
            )
        dim_slice = _best_slice_for_x(
            single_proc,
            x_key="embedding_dim",
            fixed_keys=("rows", "batch_size", "num_embeddings"),
        )
        if dim_slice:
            fixed_rows = dim_slice["rows"]
            fixed_batch = dim_slice["batch_size"]
            fixed_embeddings = dim_slice["num_embeddings"]
            figures.append(
                FigureSpec(
                    filename="e2e_dim.svg",
                    title=f"Embedding dimension 曲线: rows={fixed_rows}, batch={fixed_batch}, emb={fixed_embeddings}",
                    xlabel="Embedding dim",
                    ylabel="Samples/s",
                    series=_filter_multi_point_series(_median_ok_by_label(
                        single_proc,
                        predicate=lambda row: _primary_lane(row)
                        and _row_int(row, "rows") == fixed_rows
                        and _row_int(row, "batch_size") == fixed_batch
                        and _row_int(row, "num_embeddings") == fixed_embeddings,
                        x_key="embedding_dim",
                    )),
                    description="向量维度敏感性：只有同一配置下存在至少两个 embedding dim 点时才生成，避免单点图误导。",
                )
            )
        rdma_rows = [row for row in ok_rows if str(row.get("ps_type", "")).upper() == "RDMA"]
        if rdma_rows:
            rdma_slice = _best_slice_for_x(
                rdma_rows,
                x_key="batch_size",
                fixed_keys=("rows", "num_embeddings", "embedding_dim"),
            )
            if rdma_slice:
                fixed_rows = rdma_slice["rows"]
                fixed_embeddings = rdma_slice["num_embeddings"]
                fixed_dim = rdma_slice["embedding_dim"]
                figures.append(
                    FigureSpec(
                        filename="e2e_rdma_batch.svg",
                        title="RecStore-RDMA PyTorch/model batch 曲线",
                        xlabel="Batch size",
                        ylabel="Samples/s",
                        series=_filter_multi_point_series(_median_ok_by_label(
                            rdma_rows,
                            predicate=lambda row: _row_int(row, "rows") == fixed_rows
                            and _row_int(row, "num_embeddings") == fixed_embeddings
                            and _row_int(row, "embedding_dim") == fixed_dim,
                            x_key="batch_size",
                        )),
                        description="RDMA 端到端创新点：固定模型规模后比较 PET/EH/MAP 等 RDMA 后端随 batch size 的变化。",
                    )
                )
            rdma_capacity_slice = _best_slice_for_x(
                rdma_rows,
                x_key="num_embeddings",
                fixed_keys=("rows", "batch_size", "embedding_dim"),
            )
            if rdma_capacity_slice:
                fixed_rows = rdma_capacity_slice["rows"]
                fixed_batch = rdma_capacity_slice["batch_size"]
                fixed_dim = rdma_capacity_slice["embedding_dim"]
                figures.append(
                    FigureSpec(
                        filename="e2e_rdma_capacity.svg",
                        title="RecStore-RDMA PyTorch/model capacity 曲线",
                        xlabel="Embedding rows per table cap",
                        ylabel="Samples/s",
                        series=_filter_multi_point_series(_median_ok_by_label(
                            rdma_rows,
                            predicate=lambda row: _row_int(row, "rows") == fixed_rows
                            and _row_int(row, "batch_size") == fixed_batch
                            and _row_int(row, "embedding_dim") == fixed_dim,
                            x_key="num_embeddings",
                        )),
                        xmode_log=True,
                        description="RDMA 容量敏感性：展示 RDMA 参数服务器接入后，在不同 embedding 容量下端到端吞吐是否稳定。",
                    )
                )
    rdma_status_rows = [
        row for row in summary_rows
        if row.get("status") != "ok" and str(row.get("ps_type", "")).upper() == "RDMA"
    ]
    rdma_failed_by_capacity: dict[float, float] = {}
    for row in rdma_status_rows:
        capacity = float(_row_int(row, "num_embeddings", 0))
        if capacity <= 0:
            continue
        rdma_failed_by_capacity[capacity] = rdma_failed_by_capacity.get(capacity, 0.0) + 1.0
    if rdma_failed_by_capacity:
        figures.append(
            FigureSpec(
                filename="rdma_failure_capacity.svg",
                title="RecStore-RDMA failed/skipped coverage",
                xlabel="Embedding rows per table cap",
                ylabel="Failed or skipped runs",
                series={
                    "RDMA failed/skipped": [
                        (capacity, count)
                        for capacity, count in sorted(rdma_failed_by_capacity.items())
                    ]
                },
                xmode_log=True,
                description="RDMA 大容量覆盖图：失败点单独画出，不插值为吞吐；用于展示哪些容量仍受启动、OOM 或硬件资源限制。",
            )
        )
    if gap_rows:
        figures.append(
            FigureSpec(
                filename="speedup_batch.svg",
                title="RecStore/TorchRec speedup vs batch",
                xlabel="Batch size",
                ylabel="Speedup",
                series={
                    "RecStore/HBM": _median_points(
                        (
                            float(row.get("batch_size", 0)),
                            _to_float(row.get("recstore_vs_hbm_speedup")),
                        )
                        for row in gap_rows
                    ),
                    "RecStore/UVM": _median_points(
                        (
                            float(row.get("batch_size", 0)),
                            _to_float(row.get("recstore_vs_uvm_speedup")),
                        )
                        for row in gap_rows
                    ),
                },
                description="端到端相对加速：每个 batch size 上取可配对配置的中位 speedup，用于避免大表中逐行数字掩盖趋势。",
            )
        )
        figures.append(
            FigureSpec(
                filename="speedup_capacity.svg",
                title="RecStore/TorchRec speedup vs capacity",
                xlabel="Embedding rows per table cap",
                ylabel="Speedup",
                series={
                    "RecStore/HBM": _median_points(
                        (
                            float(row.get("num_embeddings", 0)),
                            _to_float(row.get("recstore_vs_hbm_speedup")),
                        )
                        for row in gap_rows
                    ),
                    "RecStore/UVM": _median_points(
                        (
                            float(row.get("num_embeddings", 0)),
                            _to_float(row.get("recstore_vs_uvm_speedup")),
                        )
                        for row in gap_rows
                    ),
                },
                xmode_log=True,
                description="容量维度 speedup：按 embedding 容量聚合 RecStore 相对 TorchRec-HBM/UVM 的端到端速度比。",
            )
        )
    ps_series: dict[str, list[tuple[float, float]]] = {}
    ps_grouped: dict[int, list[float]] = {}
    for row in ps_rows:
        if row.get("status") not in {"ok", "success"} or str(row.get("phase", "")) != "run":
            continue
        try:
            cp = int(str(row.get("client_processes", "0") or "0"))
        except ValueError:
            continue
        throughput = 0.0
        if row.get("throughput_mkeys_sec", "") not in {"", None}:
            throughput = float(row["throughput_mkeys_sec"])
        elif row.get("key_ops_per_sec", "") not in {"", None}:
            throughput = float(row["key_ops_per_sec"]) / 1e6
        if cp > 0 and throughput > 0:
            ps_grouped.setdefault(cp, []).append(throughput)
    if ps_grouped:
        ps_series["RDMA per-client"] = [
            (float(cp), statistics.median(values)) for cp, values in sorted(ps_grouped.items())
        ]
        figures.append(
            FigureSpec(
                filename="rdma_ps_clients.svg",
                title="RDMA PS/network client process 扩展",
                xlabel="Client processes",
                ylabel="Median M keys/s per client",
                series=ps_series,
                description="PS/network 校准：只衡量 RDMA 参数服务器 GET 路径，不与 PyTorch/model samples/s 直接混算。",
            )
        )
    return [figure for figure in figures if any(figure.series.values())]


def _svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _nice_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _svg_line_plot(spec: FigureSpec, *, width: int = 960, height: int = 540) -> str:
    series = {label: points for label, points in spec.series.items() if points}
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        return ""
    plot_left, plot_right = 92, width - 34
    plot_top, plot_bottom = 62, height - 92
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    if spec.xmode_log:
        xs = [max(value, 1e-9) for value in xs]
        xmin_raw, xmax_raw = min(xs), max(xs)
        xmin, xmax = math.log10(xmin_raw), math.log10(xmax_raw)
    else:
        xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0.0, max(ys)
    if xmax <= xmin:
        delta = max(abs(xmin) * 0.1, 1.0)
        xmin -= delta
        xmax += delta
    if ymax <= ymin:
        ymax = 1.0
    ymax *= 1.08

    def sx(value: float) -> float:
        x_value = math.log10(max(value, 1e-9)) if spec.xmode_log else value
        return plot_left + (x_value - xmin) / (xmax - xmin) * (plot_right - plot_left)

    def sy(value: float) -> float:
        return plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top)

    palette = ["#0B7285", "#E8590C", "#2F9E44", "#5F3DC4", "#C92A2A", "#1864AB"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Georgia,'Noto Serif CJK SC',serif;fill:#1f2933}.axis{stroke:#2f3a45;stroke-width:1.4}.grid{stroke:#d9e2ec;stroke-width:1}.line{fill:none;stroke-width:3}.dot{stroke:white;stroke-width:1.4}.legend{font-size:14px}.tick{font-size:12px}.title{font-size:22px;font-weight:700}.label{font-size:15px}</style>",
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{_svg_escape(spec.title)}</text>',
    ]
    for i in range(5):
        y_value = ymin + (ymax - ymin) * i / 4
        y = sy(y_value)
        lines.append(f'<line class="grid" x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}"/>')
        lines.append(f'<text class="tick" x="{plot_left - 10}" y="{y + 4:.1f}" text-anchor="end">{_nice_number(y_value)}</text>')
    x_tick_values = sorted({point[0] for point in all_points})
    if len(x_tick_values) > 8:
        step = max(1, len(x_tick_values) // 6)
        x_tick_values = x_tick_values[::step]
    for x_value in x_tick_values:
        x = sx(x_value)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}"/>')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{plot_bottom + 22}" text-anchor="middle">{_nice_number(x_value)}</text>')
    lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}"/>')
    lines.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')
    lines.append(f'<text class="label" x="{(plot_left + plot_right) / 2:.1f}" y="{height - 20}" text-anchor="middle">{_svg_escape(spec.xlabel)}</text>')
    lines.append(f'<text class="label" transform="translate(24,{(plot_top + plot_bottom) / 2:.1f}) rotate(-90)" text-anchor="middle">{_svg_escape(spec.ylabel)}</text>')
    for idx, (label, points) in enumerate(sorted(series.items())):
        color = palette[idx % len(palette)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        lines.append(f'<polyline class="line" stroke="{color}" points="{coords}"/>')
        for x, y in points:
            lines.append(f'<circle class="dot" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{color}"/>')
        legend_x = plot_left + (idx % 2) * 360
        legend_y = plot_bottom + 48 + (idx // 2) * 22
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text class="legend" x="{legend_x + 32}" y="{legend_y + 5}">{_svg_escape(label)}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def write_svg_figures(
    output_root: Path,
    *,
    summary_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    ps_rows: list[dict[str, Any]],
) -> list[Path]:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for stale in figure_dir.glob("*.svg"):
        stale.unlink()
    written: list[Path] = []
    for spec in build_figure_specs(summary_rows, gap_rows, ps_rows):
        svg = _svg_line_plot(spec)
        if not svg:
            continue
        path = figure_dir / spec.filename
        path.write_text(svg + "\n", encoding="utf-8")
        written.append(path)
    return written
