from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .common import FigureSection, _status_reason, _to_float
from .figures import _primary_lane, _rdma_lane, build_figure_specs
from .summary import _geomean, build_result_insights


def render_latex_report(
    *,
    summary_rows: list[dict[str, Any]],
    ps_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    gpu_count = int(metadata.get("gpu_count", 0))
    multi_gpu_note = (
        "本机检测到多张 GPU，可运行单机多卡 TorchRec/RecStore 多进程扩展。"
        if gpu_count > 1
        else "本机仅检测到 1 张 GPU，因此单机多卡实验在本文档中标记为单机多进程/单 GPU 资源竞争观测，不能作为多 GPU 扩展性结论。"
    )
    rdma_note = (
        "RDMA verbs 设备存在，PS/network 层使用 RDMA GET 路径校准。"
        if metadata.get("rdma_available")
        else "未检测到 RDMA verbs 设备，RDMA 行应标记为 skipped。"
    )
    e2e_table = _latex_e2e_table(summary_rows)
    best_table = _latex_best_table(summary_rows)
    gap_table = _latex_gap_table(gap_rows)
    gap_group_table = _latex_gap_group_table(gap_rows)
    skipped_table = _latex_status_table(summary_rows)
    ps_table = _latex_ps_table(ps_rows)
    ps_client_scaling_table = _latex_ps_client_scaling_table(ps_rows)
    repeat_table = _latex_repeat_table(summary_rows)
    metadata_table = _latex_metadata_table(metadata, summary_rows, ps_rows, gap_rows)
    environment_table = _latex_environment_table(metadata)
    artifact_table = _latex_artifact_table(metadata, summary_rows, ps_rows, gap_rows)
    figure_section = _latex_figure_section(summary_rows, gap_rows, ps_rows)
    figure_reading_guide = _latex_figure_reading_guide(summary_rows, gap_rows, ps_rows)
    insights = _latex_insights(
        build_result_insights(
            summary_rows=summary_rows,
            gap_rows=gap_rows,
            ps_rows=ps_rows,
            metadata=metadata,
        )
    )
    executive_summary = _latex_executive_summary(gap_rows, metadata)
    return rf"""\documentclass[UTF8]{{ctexart}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{hyperref}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\title{{RecStore 与 TorchRec 端到端性能对比报告}}
\author{{RecStore Benchmark Automation}}
\date{{{_latex_escape(metadata.get('created_at', ''))}}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage

\section{{TorchRec 端到端对比实验}}
\subsection{{实验边界}}
本文将结果分为 PyTorch/model 与 PS/network 两层。PyTorch/model 层通过 \texttt{{model\_zoo/rs\_demo}} 比较 TorchRec 和 RecStore 的训练迭代耗时，包含 ID 准备、embedding lookup、pooling、backward、sparse update 和 Python runner 开销。PS/network 层通过 \texttt{{run\_benchmark\_ps.py}} 单独报告 RDMA 参数服务器路径，不把该层结果直接外推为端到端模型吞吐。

{multi_gpu_note}
{rdma_note}

\subsection{{结论摘要}}
{executive_summary}

\subsection{{实验元数据}}
{metadata_table}

\subsection{{硬件与软件环境}}
{environment_table}

\subsection{{Artifact 与 source 清单}}
{artifact_table}

\subsection{{论文实验章节对齐}}
近期推荐系统训练/存储类系统论文在使用 TorchRec 或 DLRM/TorchRec 生态作为 baseline 时，实验章节通常不只报告单一吞吐，而是同时覆盖训练 step latency、samples/s、embedding lookup/update 分解、HBM 与 UVM/cache 的容量敏感性、embedding table 规模、batch size、embedding dimension、通信/参数服务器路径、多进程或多 GPU 扩展、以及失败/OOM/跳过原因。本文档据此将 TorchRec-HBM 与 TorchRec-UVMCache 作为主 baseline，将 RecStore 的 BRPC/GRPC/LOCAL\_SHM、PET hash/extendible hash、prefetch 深度和 RDMA PS/network 校准拆成独立实验行。这样可以避免把存储层或网络层优势直接外推成端到端模型结论。

\subsection{{分场景图形对比}}
{figure_section}

\subsection{{图形阅读结论}}
{figure_reading_guide}

\subsection{{RecStore/TorchRec 分组几何均值}}
{gap_group_table}

\subsection{{失败与跳过行}}
{skipped_table}

\subsection{{RDMA PS/network 校准}}
{ps_table}

\subsection{{RDMA client process 扩展性}}
{ps_client_scaling_table}

\subsection{{重复实验稳定性}}
{repeat_table}

\subsection{{结果洞察}}
{insights}

\subsection{{当前解释}}
RecStore 模型层当前可运行传输包括 BRPC、GRPC、LOCAL\_SHM 和 RDMA。RecStore-RDMA PyTorch/model 行使用 RDMAPSClientAdapter 的 prefetch/get 与同步 update 闭环；PS/network RDMA 表仍作为单独校准层，不能直接与 PyTorch/model samples/s 混算。

\appendix
\section{{完整数字表}}
\subsection{{端到端 lane 分布摘要}}
{e2e_table}

\subsection{{按配置最优结果}}
{best_table}

\subsection{{RecStore 与 TorchRec 差距}}
{gap_table}

\subsection{{端到端明细截断表}}
完整明细见 \texttt{{summary\_e2e.csv}}，主报告只保留截断表用于查错。
{_latex_e2e_detail_table(summary_rows)}

\end{{document}}
"""


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _most_common_int(rows: list[dict[str, Any]], key: str, default: int) -> int:
    counts: dict[int, int] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            value = int(str(row.get(key, "0") or "0"))
        except ValueError:
            continue
        if value > 0:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return default
    return max(sorted(counts), key=lambda value: counts[value])


def _latex_line_plot(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: dict[str, list[tuple[float, float]]],
    xmode_log: bool = False,
) -> str:
    series = {label: points for label, points in series.items() if points}
    if not series:
        return ""
    lines = [
        "\\begin{figure}[htbp]",
        "\\centering",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        f"title={{{_latex_escape(title)}}},",
        f"xlabel={{{_latex_escape(xlabel)}}},",
        f"ylabel={{{_latex_escape(ylabel)}}},",
        "width=0.92\\linewidth,",
        "height=0.42\\linewidth,",
        "grid=both,",
        "legend style={font=\\scriptsize,at={(0.5,-0.22)},anchor=north,legend columns=2},",
        "tick label style={font=\\scriptsize},",
        "label style={font=\\small},",
    ]
    if xmode_log:
        lines.append("xmode=log,")
    lines.extend(["ymin=0,", "]"])
    for label, points in sorted(series.items()):
        coords = " ".join(f"({x:.6g},{y:.6g})" for x, y in points)
        lines.append(f"\\addplot+[mark=*] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{_latex_escape(label)}}}")
    lines.extend(
        [
            "\\end{axis}",
            "\\end{tikzpicture}",
            f"\\caption{{{_latex_escape(title)}}}",
            "\\end{figure}",
        ]
    )
    return "\n".join(lines)


def _latex_figure_section(
    summary_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    ps_rows: list[dict[str, Any]],
) -> str:
    figure_specs = build_figure_specs(summary_rows, gap_rows, ps_rows)
    if not figure_specs:
        return "当前可用结果不足以绘制分场景图；请先运行至少一个成功的 PyTorch/model 或 RDMA PS/network 实验。"
    by_name = {spec.filename: spec for spec in figure_specs}
    grouped_sections = [
        FigureSection(
            title="端到端主线",
            purpose="先看 TorchRec-HBM、TorchRec-UVMCache 与 RecStore 各端到端 lane 的 samples/s，判断模型层整体收益，而不是只看存储层吞吐。",
            figures=tuple(
                by_name[name]
                for name in ("e2e_batch.svg", "e2e_capacity.svg", "e2e_dim.svg")
                if name in by_name
            ),
        ),
        FigureSection(
            title="RecStore-RDMA 创新点",
            purpose="单独放大 RDMA PyTorch/model 闭环，比较 PET/EH/MAP 后端在 batch 和容量变化下的端到端表现。",
            figures=tuple(
                by_name[name]
                for name in ("e2e_rdma_batch.svg", "e2e_rdma_capacity.svg")
                if name in by_name
            ),
        ),
        FigureSection(
            title="相对 TorchRec 的速度比",
            purpose="把 RecStore 最优端到端点分别除以 TorchRec-HBM 与 TorchRec-UVMCache，避免从绝对吞吐表中人工找差距。",
            figures=tuple(
                by_name[name]
                for name in ("speedup_batch.svg", "speedup_capacity.svg")
                if name in by_name
            ),
        ),
        FigureSection(
            title="RDMA 覆盖与网络层校准",
            purpose="失败覆盖图只展示未形成稳态吞吐的容量点；PS/network 图只用于 RDMA 参数服务器 GET 路径校准，不能与模型层 samples/s 混算。",
            figures=tuple(
                by_name[name]
                for name in ("rdma_failure_capacity.svg", "rdma_ps_clients.svg")
                if name in by_name
            ),
        ),
    ]
    lines = [
        "本节按问题拆图，而不是把所有 lane 放进一个宽表。正文直接嵌入 SVG 图；若目标会议模板不支持 SVG，可用 Inkscape 或 rsvg-convert 将 \\texttt{figures/} 下文件转换为 PDF 后替换路径。",
        "",
    ]
    emitted: set[str] = set()
    for section in grouped_sections:
        if not section.figures:
            continue
        lines.append(f"\\paragraph{{{_latex_escape(section.title)}}}")
        lines.append(_latex_escape(section.purpose))
        for spec in section.figures:
            emitted.add(spec.filename)
            lines.extend(_latex_figure_block(spec))
    leftovers = [spec for spec in figure_specs if spec.filename not in emitted]
    if leftovers:
        lines.append("\\paragraph{其他诊断图}")
        lines.append("以下图由当前结果自动生成，用于补充诊断。")
        for spec in leftovers:
            lines.extend(_latex_figure_block(spec))
    return "\n".join(lines)


def _latex_figure_block(spec: FigureSpec) -> list[str]:
    return [
        "\\begin{figure}[htbp]",
        "\\centering",
        f"\\includegraphics[width=0.94\\linewidth]{{figures/{_latex_escape(spec.filename)}}}",
        f"\\caption{{{_latex_escape(spec.title)}。{_latex_escape(spec.description or spec.ylabel)}}}",
        f"\\label{{fig:{_latex_escape(Path(spec.filename).stem)}}}",
        "\\end{figure}",
    ]


def _format_best_point(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> str:
    candidates = [row for row in rows if row.get("status") == "ok" and predicate(row)]
    if not candidates:
        return "暂无成功点。"
    best = max(candidates, key=lambda row: _to_float(row.get("samples_per_sec")))
    return (
        f"{best.get('label', '')}: rows={best.get('rows', '')}, "
        f"batch={best.get('batch_size', '')}, emb={best.get('num_embeddings', '')}, "
        f"dim={best.get('embedding_dim', '')}, "
        f"{_to_float(best.get('samples_per_sec')):.1f} samples/s。"
    )


def _latex_figure_reading_guide(
    summary_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    ps_rows: list[dict[str, Any]],
) -> str:
    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    lines = ["\\begin{itemize}"]
    if ok_rows:
        lines.append(
            "\\item 端到端主线最佳点："
            + _latex_escape(_format_best_point(ok_rows, lambda row: True))
        )
        lines.append(
            "\\item RDMA 端到端最佳点："
            + _latex_escape(_format_best_point(ok_rows, _rdma_lane))
        )
    if gap_rows:
        hbm = [_to_float(row.get("recstore_vs_hbm_speedup")) for row in gap_rows]
        uvm = [_to_float(row.get("recstore_vs_uvm_speedup")) for row in gap_rows]
        lines.append(
            "\\item Speedup 图读法：每个横轴点聚合同类配置中位数，RecStore/HBM 几何均值为 "
            f"{_geomean(hbm):.2f}x，RecStore/UVM 几何均值为 {_geomean(uvm):.2f}x。"
        )
    rdma_failures = [
        row for row in summary_rows
        if row.get("status") != "ok" and str(row.get("ps_type", "")).upper() == "RDMA"
    ]
    if rdma_failures:
        reasons: dict[str, int] = {}
        for row in rdma_failures:
            reason = _status_reason(row)
            reasons[reason] = reasons.get(reason, 0) + 1
        reason_text = "; ".join(
            f"{reason} ({count})" for reason, count in sorted(reasons.items())
        )
        lines.append(
            "\\item RDMA 失败点不插值、不外推："
            + _latex_escape(reason_text)
            + "。"
        )
        if any("server did not publish ready" in reason for reason in reasons):
            lines.append(
                "\\item RDMA 大容量 ready timeout 需要结合系统日志解释；当前 artifact 保存了 "
                "\\texttt{diagnostics/rdma\\_petps\\_server\\_oom\\_dmesg.txt}，其中可见 "
                "\\texttt{petps\\_server} 被 OOM killer 杀掉。"
            )
    run_phase = [
        row for row in ps_rows
        if row.get("status") in {"ok", "success"} and str(row.get("phase", "")) == "run"
    ]
    if run_phase:
        values = []
        for row in run_phase:
            if row.get("throughput_mkeys_sec", "") not in {"", None}:
                values.append(_to_float(row.get("throughput_mkeys_sec")))
            elif row.get("key_ops_per_sec", "") not in {"", None}:
                values.append(_to_float(row.get("key_ops_per_sec")) / 1e6)
        if values:
            lines.append(
                "\\item RDMA PS/network 图只用于传输层校准：run phase 中位吞吐为 "
                f"{statistics.median(values):.2f} M keys/s。"
            )
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def _latex_executive_summary(gap_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    if not gap_rows:
        return "当前尚无可配对的 RecStore/TorchRec 结果，无法生成结论摘要。"
    hbm = [_to_float(row.get("recstore_vs_hbm_speedup")) for row in gap_rows]
    uvm = [_to_float(row.get("recstore_vs_uvm_speedup")) for row in gap_rows]

    def group_geomean(predicate: Callable[[dict[str, Any]], bool], key: str) -> float:
        values = [_to_float(row.get(key)) for row in gap_rows if predicate(row)]
        return _geomean(values)

    small_batch_uvm = group_geomean(
        lambda row: int(str(row.get("batch_size", "0") or "0")) <= 1024,
        "recstore_vs_uvm_speedup",
    )
    large_batch_uvm = group_geomean(
        lambda row: int(str(row.get("batch_size", "0") or "0")) >= 4096,
        "recstore_vs_uvm_speedup",
    )
    large_capacity_uvm = group_geomean(
        lambda row: int(str(row.get("num_embeddings", "0") or "0")) >= 4000000,
        "recstore_vs_uvm_speedup",
    )
    multi_gpu_text = (
        "当前机器 GPU 数不足 2，单机多卡结果只能保留 skipped/限制说明，不能作为扩展性结论。"
        if int(metadata.get("gpu_count", 0) or 0) < 2
        else "当前机器检测到多张 GPU，可补充真实单机多卡扩展性结果。"
    )
    lines = [
        "\\begin{itemize}",
        (
            f"\\item 共 {len(gap_rows)} 个可配对配置；最佳 RecStore 相对 TorchRec-HBM 胜 "
            f"{sum(value >= 1.0 for value in hbm)}/{len(hbm)}，RecStore/HBM 几何均值为 {_geomean(hbm):.2f}x。"
        ),
        (
            f"\\item 最佳 RecStore 相对 TorchRec-UVMCache 胜 {sum(value >= 1.0 for value in uvm)}/{len(uvm)}，"
            f"RecStore/UVM 几何均值为 {_geomean(uvm):.2f}x。"
        ),
        (
            f"\\item batch size 是主要分界：batch<=1024 时 RecStore/UVM 为 {small_batch_uvm:.2f}x，"
            f"batch>=4096 时为 {large_batch_uvm:.2f}x。"
        ),
        f"\\item 大容量组 emb>=4M 的 RecStore/UVM 几何均值为 {large_capacity_uvm:.2f}x。",
        "\\item RDMA 结果仅属于 PS/network 层校准，不能直接外推为 PyTorch/model 端到端 RDMA 加速。",
        f"\\item {_latex_escape(multi_gpu_text)}",
        "\\end{itemize}",
    ]
    return "\n".join(lines)


def _latex_e2e_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "尚无成功的 PyTorch/model 层结果。"
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return "尚无成功的 PyTorch/model 层结果。"
    by_lane: dict[str, list[float]] = {}
    for row in ok_rows:
        by_lane.setdefault(str(row.get("label", "")), []).append(_to_float(row.get("samples_per_sec")))
    lines = [
        f"共 {len(ok_rows)} 条成功 PyTorch/model 行。正文只展示每条 lane 的吞吐分布摘要，完整逐配置数据见 \\texttt{{summary\\_e2e.csv}}。",
        "",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Lane & Rows & Median samples/s & Max samples/s & Min samples/s \\\\",
        "\\midrule",
    ]
    for label, values in sorted(by_lane.items()):
        positive = [value for value in values if value > 0.0]
        if not positive:
            continue
        lines.append(
            f"{_latex_escape(label)} & {len(positive)} & {statistics.median(positive):.1f} & "
            f"{max(positive):.1f} & {min(positive):.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_e2e_detail_table(rows: list[dict[str, Any]]) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    selected = ok_rows[:40]
    if not selected:
        return "尚无成功的 PyTorch/model 层结果。"
    lines = [
        f"下表展示前 {len(selected)} 条，完整数据见 summary\\_e2e.csv。",
        "",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Lane & Rows & Batch & Emb rows & Dim & Step ms & Samples/s \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            f"{_latex_escape(row.get('label', ''))} & "
            f"{row.get('rows', '')} & {row.get('batch_size', '')} & "
            f"{row.get('num_embeddings', '')} & {row.get('embedding_dim', '')} & "
            f"{float(row.get('mean_step_total_ms', 0.0)):.2f} & "
            f"{float(row.get('samples_per_sec', 0.0)):.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_best_table(rows: list[dict[str, Any]]) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return "尚无可汇总的成功结果。"
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in ok_rows:
        key = (
            str(row.get("rows", "")),
            str(row.get("batch_size", "")),
            str(row.get("num_embeddings", "")),
            str(row.get("embedding_dim", "")),
        )
        current = grouped.get(key)
        if current is None or float(row.get("samples_per_sec", 0.0)) > float(
            current.get("samples_per_sec", 0.0)
        ):
            grouped[key] = row
    lines = [
        "\\begin{tabular}{rrrrrl}",
        "\\toprule",
        "Rows & Batch & Emb rows & Dim & Samples/s & Best lane \\\\",
        "\\midrule",
    ]
    for key, row in sorted(grouped.items(), key=lambda item: tuple(int(v or 0) for v in item[0])):
        lines.append(
            f"{key[0]} & {key[1]} & {key[2]} & {key[3]} & "
            f"{float(row.get('samples_per_sec', 0.0)):.1f} & "
            f"{_latex_escape(row.get('label', ''))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_gap_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "尚无可计算的 RecStore/TorchRec 配对结果。"
    lines = [
        f"共 {len(rows)} 个有 TorchRec-HBM/UVM 配对 baseline 的配置。",
        "",
        "\\begin{tabular}{rrrrrrr}",
        "\\toprule",
        "Rows & Batch & Emb rows & Dim & RecStore/HBM & RecStore/UVM & Best RecStore \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.get('rows', '')} & {row.get('batch_size', '')} & "
            f"{row.get('num_embeddings', '')} & "
            f"{row.get('embedding_dim', '')} & "
            f"{float(row.get('recstore_vs_hbm_speedup', 0.0)):.2f}x & "
            f"{float(row.get('recstore_vs_uvm_speedup', 0.0)):.2f}x & "
            f"{_latex_escape(row.get('best_recstore_label', ''))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_gap_group_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "尚无可分组的 RecStore/TorchRec 配对结果。"

    def batch_group(row: dict[str, Any]) -> str:
        batch_size = int(str(row.get("batch_size", "0") or "0"))
        if batch_size <= 1024:
            return "batch<=1024"
        if batch_size == 2048:
            return "batch=2048"
        return "batch>=4096"

    def capacity_group(row: dict[str, Any]) -> str:
        num_embeddings = int(str(row.get("num_embeddings", "0") or "0"))
        if num_embeddings <= 800000:
            return "emb<=800K"
        if num_embeddings == 2000000:
            return "emb=2M"
        return "emb>=4M"

    groups: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
        ("Batch", "batch<=1024", lambda row: batch_group(row) == "batch<=1024"),
        ("Batch", "batch=2048", lambda row: batch_group(row) == "batch=2048"),
        ("Batch", "batch>=4096", lambda row: batch_group(row) == "batch>=4096"),
        ("Capacity", "emb<=800K", lambda row: capacity_group(row) == "emb<=800K"),
        ("Capacity", "emb=2M", lambda row: capacity_group(row) == "emb=2M"),
        ("Capacity", "emb>=4M", lambda row: capacity_group(row) == "emb>=4M"),
    ]
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Group type & Group & Count & HBM wins & UVM wins & Geo RecStore/HBM & Geo RecStore/UVM \\\\",
        "\\midrule",
    ]
    for group_type, group_name, predicate in groups:
        group_rows = [row for row in rows if predicate(row)]
        if not group_rows:
            continue
        hbm = [_to_float(row.get("recstore_vs_hbm_speedup")) for row in group_rows]
        uvm = [_to_float(row.get("recstore_vs_uvm_speedup")) for row in group_rows]
        lines.append(
            f"{_latex_escape(group_type)} & {_latex_escape(group_name)} & {len(group_rows)} & "
            f"{sum(value >= 1.0 for value in hbm)} & {sum(value >= 1.0 for value in uvm)} & "
            f"{_geomean(hbm):.2f}x & {_geomean(uvm):.2f}x \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_insights(insights: list[str]) -> str:
    if not insights:
        return "当前结果不足以形成自动化洞察。"
    lines = ["\\begin{itemize}"]
    for insight in insights:
        lines.append(f"\\item {_latex_escape(insight)}")
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def _latex_status_table(rows: list[dict[str, Any]]) -> str:
    status_rows = [row for row in rows if row.get("status") != "ok"]
    if not status_rows:
        return "所有端到端行均成功完成。"
    unique: dict[tuple[str, str, str], int] = {}
    for row in status_rows:
        reason = _status_reason(row)
        key = (str(row.get("label", "")), str(row.get("status", "")), reason)
        unique[key] = unique.get(key, 0) + 1
    lines = [
        "\\begin{tabular}{llrl}",
        "\\toprule",
        "Lane & Status & Count & Reason \\\\",
        "\\midrule",
    ]
    for (label, status, reason), count in sorted(unique.items()):
        lines.append(
            f"{_latex_escape(label)} & "
            f"{_latex_escape(status)} & "
            f"{count} & "
            f"{_latex_escape(reason)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_ps_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "尚无 PS/network 层结果。"
    grouped: dict[tuple[str, str, str], list[float]] = {}
    failures: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        phase = str(row.get("phase", ""))
        status = str(row.get("status", ""))
        if status not in {"", "ok", "success"}:
            source = str(row.get("source_profile") or row.get("summary_csv") or row.get("layer", ""))
            reason = str(row.get("message") or row.get("status") or "failed")
            failures[
                (
                    str(row.get("transport", "RDMA")),
                    source,
                    str(row.get("value_size", "")),
                    f"batch_keys={row.get('batch_keys', '')}: {reason}",
                )
            ] = failures.get(
                (
                    str(row.get("transport", "RDMA")),
                    source,
                    str(row.get("value_size", "")),
                    f"batch_keys={row.get('batch_keys', '')}: {reason}",
                ),
                0,
            ) + 1
        throughput = 0.0
        for key in ("throughput_mkeys_sec", "key_ops_per_sec"):
            if row.get(key, "") not in {"", None}:
                throughput = float(row[key])
                if key == "key_ops_per_sec":
                    throughput /= 1e6
                break
        if throughput <= 0.0:
            continue
        source = str(row.get("source_profile") or row.get("summary_csv") or row.get("layer", ""))
        grouped.setdefault((str(row.get("transport", "RDMA")), source, phase), []).append(throughput)
    if grouped:
        lines = [
            "\\begin{tabular}{lllrr}",
            "\\toprule",
            "Transport & Source & Phase & Median M keys/s & Rows \\\\",
            "\\midrule",
        ]
        for (transport, source, phase), values in sorted(grouped.items()):
            lines.append(
                f"{_latex_escape(transport)} & "
                f"{_latex_escape(source)} & "
                f"{_latex_escape(phase)} & "
                f"{statistics.median(values):.2f} & {len(values)} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        if failures:
            lines.extend(
                [
                    "",
                    "失败或容量限制行：",
                    "",
                    "\\begin{tabular}{lllrl}",
                    "\\toprule",
                    "Transport & Source & Value bytes & Count & Reason \\\\",
                    "\\midrule",
                ]
            )
            for (transport, source, value_size, reason), count in sorted(failures.items()):
                lines.append(
                    f"{_latex_escape(transport)} & "
                    f"{_latex_escape(source)} & "
                    f"{_latex_escape(value_size)} & {count} & "
                    f"{_latex_escape(reason)} \\\\"
                )
            lines.extend(["\\bottomrule", "\\end{tabular}"])
        return "\n".join(lines)
    if failures:
        lines = [
            "无成功吞吐行；失败或容量限制如下：",
            "",
            "\\begin{tabular}{lllrl}",
            "\\toprule",
            "Transport & Source & Value bytes & Count & Reason \\\\",
            "\\midrule",
        ]
        for (transport, source, value_size, reason), count in sorted(failures.items()):
            lines.append(
                f"{_latex_escape(transport)} & "
                f"{_latex_escape(source)} & "
                f"{_latex_escape(value_size)} & {count} & "
                f"{_latex_escape(reason)} \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        return "\n".join(lines)
    lines = [
        "\\begin{tabular}{llr}",
        "\\toprule",
        "Transport & Status & M keys/s \\\\",
        "\\midrule",
    ]
    for row in rows[:12]:
        throughput = 0.0
        for key in ("throughput_mkeys_sec", "key_ops_per_sec"):
            if row.get(key, "") not in {"", None}:
                throughput = float(row[key])
                if key == "key_ops_per_sec":
                    throughput /= 1e6
                break
        lines.append(
            f"{_latex_escape(row.get('transport', row.get('layer', 'RDMA')))} & "
            f"{_latex_escape(row.get('status', ''))} & {throughput:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_ps_client_scaling_table(rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[int, int, int], list[float]] = {}
    repeat_totals: dict[tuple[int, int, int, str], float] = {}
    for row in rows:
        if row.get("status") not in {"ok", "success"}:
            continue
        if str(row.get("phase", "")) != "run":
            continue
        throughput = 0.0
        for key in ("throughput_mkeys_sec", "key_ops_per_sec"):
            if row.get(key, "") not in {"", None}:
                throughput = float(row[key])
                if key == "key_ops_per_sec":
                    throughput /= 1e6
                break
        if throughput <= 0.0:
            continue
        try:
            client_processes = int(str(row.get("client_processes", "")))
            value_size = int(str(row.get("value_size", "")))
            batch_keys = int(str(row.get("batch_keys", "")))
        except ValueError:
            continue
        key = (client_processes, value_size, batch_keys)
        grouped.setdefault(key, []).append(throughput)
        repeat_source = str(row.get("source_profile") or row.get("summary_csv") or "unknown")
        repeat_index = str(row.get("repeat_index", "single"))
        repeat_key = f"{repeat_source}:{repeat_index}"
        repeat_totals[(client_processes, value_size, batch_keys, repeat_key)] = (
            repeat_totals.get((client_processes, value_size, batch_keys, repeat_key), 0.0)
            + throughput
        )
    if not grouped:
        return "当前 PS/network 行缺少可按 client process 聚合的 RDMA run phase 吞吐。"
    lines = [
        "该表按 client process 数、value size、batch keys 聚合 run/fetch phase 的 per-client 中位吞吐和按 repeat 求和后的 total 中位吞吐；它是 PS/network 层扩展性校准，不代表 PyTorch/model 端到端 RDMA 加速。",
        "",
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "Client procs & Value bytes & Batch keys & Median per-client M keys/s & Median total M keys/s & Rows \\\\",
        "\\midrule",
    ]
    for (client_processes, value_size, batch_keys), values in sorted(grouped.items()):
        totals = [
            total
            for (cp, vs, bk, _repeat), total in repeat_totals.items()
            if cp == client_processes and vs == value_size and bk == batch_keys
        ]
        lines.append(
            f"{client_processes} & {value_size} & {batch_keys} & "
            f"{statistics.median(values):.2f} & "
            f"{statistics.median(totals) if totals else 0.0:.2f} & {len(values)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_metadata_table(
    metadata: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    ps_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> str:
    status_counts: dict[str, int] = {}
    for row in summary_rows:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
    status_text = ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items()))
    rows = [
        ("profile", metadata.get("profile", "")),
        ("output_root", metadata.get("output_root", "")),
        ("input_file", metadata.get("input_file", "")),
        ("gpu_count", metadata.get("gpu_count", "")),
        ("rdma_available", metadata.get("rdma_available", "")),
        ("data_rows", metadata.get("data_rows", "")),
        ("batch_sizes", metadata.get("batch_sizes", "")),
        ("num_embeddings", metadata.get("num_embeddings", "")),
        ("embedding_dims", metadata.get("embedding_dims", "")),
        ("steps", metadata.get("steps", "")),
        ("warmup_steps", metadata.get("warmup_steps", "")),
        ("repeat", metadata.get("repeat", "")),
        ("summary_e2e_rows", len(summary_rows)),
        ("summary_e2e_status", status_text),
        ("summary_gap_rows", len(gap_rows)),
        ("summary_ps_network_rows", len(ps_rows)),
        ("ps_network_sources", metadata.get("ps_network_sources", "")),
    ]
    lines = [
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Key & Value \\\\",
        "\\midrule",
    ]
    for key, value in rows:
        lines.append(f"{_latex_escape(key)} & {_latex_escape(value)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_environment_table(metadata: dict[str, Any]) -> str:
    rows = [
        ("hostname", metadata.get("hostname", "")),
        ("kernel", metadata.get("kernel", "")),
        ("git_branch", metadata.get("git_branch", "")),
        ("git_commit", metadata.get("git_commit", "")),
        ("gpu_count", metadata.get("gpu_count", "")),
        ("nvidia_smi_gpu", metadata.get("nvidia_smi_gpu", "")),
        ("torch_version", metadata.get("torch_version", "")),
        ("torch_cuda", metadata.get("torch_cuda", "")),
        ("cudnn_version", metadata.get("cudnn_version", "")),
        ("torchrec_version", metadata.get("torchrec_version", "")),
        ("rdma_available", metadata.get("rdma_available", "")),
    ]
    lines = [
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Key & Value \\\\",
        "\\midrule",
    ]
    for key, value in rows:
        lines.append(f"{_latex_escape(key)} & {_latex_escape(value)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _latex_artifact_table(
    metadata: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    ps_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> str:
    output_root = str(metadata.get("output_root", ""))
    final_artifacts = [
        ("final", "manifest.csv", len(summary_rows), output_root),
        ("final", "summary_e2e.csv", len(summary_rows), output_root),
        ("final", "summary_gap.csv", len(gap_rows), output_root),
        ("final", "summary_ps_network.csv", len(ps_rows), output_root),
        ("final", "metadata.json", 1, output_root),
        ("final", "bench_e2e_report.tex", 1, output_root),
    ]
    for figure in metadata.get("svg_figures", []) or []:
        final_artifacts.append(("final", str(figure), 1, output_root))
    source_counts: dict[tuple[str, str, str], int] = {}
    for layer, rows in (("PyTorch/model", summary_rows), ("PS/network", ps_rows)):
        for row in rows:
            source_root = str(row.get("source_root", "") or row.get("summary_csv", ""))
            if not source_root:
                continue
            source_profile = str(row.get("source_profile", ""))
            key = (layer, source_profile, source_root)
            source_counts[key] = source_counts.get(key, 0) + 1

    lines = [
        "完整原始输出仍保留在各 source root 下；本节只列最终聚合 artifact 和参与合并的 source 摘要。",
        "",
        "\\begin{longtable}{llrl}",
        "\\toprule",
        "Kind & Name/Profile & Rows & Path \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Kind & Name/Profile & Rows & Path \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for kind, name, count, path in final_artifacts:
        lines.append(
            f"{_latex_escape(kind)} & {_latex_escape(name)} & {count} & {_latex_escape(path)} \\\\"
        )
    for (layer, source_profile, source_root), count in sorted(source_counts.items()):
        name = source_profile or layer
        lines.append(
            f"{_latex_escape(layer)} & {_latex_escape(name)} & {count} & "
            f"{_latex_escape(source_root)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{longtable}"])
    return "\n".join(lines)


def _latex_repeat_table(rows: list[dict[str, Any]]) -> str:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    grouped: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in ok_rows:
        key = (
            str(row.get("rows", "")),
            str(row.get("batch_size", "")),
            str(row.get("num_embeddings", "")),
            str(row.get("embedding_dim", "")),
            str(row.get("label", "")),
        )
        grouped.setdefault(key, []).append(_to_float(row.get("samples_per_sec")))
    repeat_rows = [
        (key, values)
        for key, values in grouped.items()
        if len([value for value in values if value > 0.0]) >= 3
    ]
    if not repeat_rows:
        return "当前没有同配置同 lane 的 repeat>=3 结果。"
    lines = [
        "\\begin{tabular}{rrrrlrr}",
        "\\toprule",
        "Rows & Batch & Emb rows & Dim & Lane & Mean samples/s & CV \\\\",
        "\\midrule",
    ]
    for key, values in sorted(repeat_rows, key=lambda item: tuple(int(v or 0) for v in item[0][:4]) + (item[0][4],)):
        positive = [value for value in values if value > 0.0]
        mean = statistics.fmean(positive)
        cv = statistics.pstdev(positive) / mean if len(positive) > 1 and mean > 0.0 else 0.0
        lines.append(
            f"{key[0]} & {key[1]} & {key[2]} & {key[3]} & "
            f"{_latex_escape(key[4])} & {mean:.1f} & {cv:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)
