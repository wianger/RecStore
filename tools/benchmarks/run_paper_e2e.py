from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path("/nas/home/shq/docker/rs_demo/paper_e2e")
DEFAULT_NAS_DAY0 = Path("/nas/home/shq/RecStore_/model_zoo/torchrec_dlrm/partial_data/day_0.bak")
SPARSE_FEATURES_PER_SAMPLE = 26


@dataclass(frozen=True)
class E2ELane:
    slug: str
    label: str
    backend: str
    ps_type: str = ""
    recstore_index_type: str = ""
    ps_kv_backend: str = ""
    allocator: str = "R2ShmMalloc"
    nproc_per_node: int = 1
    torchrec_memory_mode: str = "hbm"
    prefetch_depth: int = 0
    enable_single_node_fast_path: bool = False
    single_node_ps_backend: str = "local_shm"
    extra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExperimentPlan:
    profile: str
    output_root: Path
    data_rows: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    embedding_dims: tuple[int, ...]
    steps: int
    warmup_steps: int
    repeat: int
    lanes: tuple[E2ELane, ...]


@dataclass(frozen=True)
class PlanOverrides:
    data_rows: tuple[int, ...] = ()
    batch_sizes: tuple[int, ...] = ()
    num_embeddings: tuple[int, ...] = ()
    embedding_dims: tuple[int, ...] = ()
    steps: int | None = None
    warmup_steps: int | None = None
    repeat: int | None = None
    only_lanes: tuple[str, ...] = ()
    include_ablation_lanes: bool = False


@dataclass(frozen=True)
class ExecutionContext:
    remote_train_host: str = ""
    remote_repo_root: Path = ROOT
    python_bin: str = sys.executable
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    external_recstore_runtime_dir: Path | None = None
    no_start_recstore_server: bool = False
    server_host: str = ""
    server_port0: int | None = None
    server_port1: int | None = None


@dataclass(frozen=True)
class FigureSpec:
    filename: str
    title: str
    xlabel: str
    ylabel: str
    series: dict[str, list[tuple[float, float]]]
    xmode_log: bool = False
    description: str = ""


@dataclass(frozen=True)
class FigureSection:
    title: str
    purpose: str
    figures: tuple[FigureSpec, ...]


def _has_rdma() -> bool:
    infiniband = Path("/dev/infiniband")
    return infiniband.exists() and any(infiniband.glob("uverbs*"))


def _gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _run_text(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=10)
    except Exception:
        return ""
    return (result.stdout or result.stderr or "").strip()


def _collect_environment_metadata() -> dict[str, Any]:
    env: dict[str, Any] = {
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "git_branch": _run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "hostname": _run_text(["hostname"]),
        "kernel": _run_text(["uname", "-srmo"]),
    }
    nvidia = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    if nvidia:
        env["nvidia_smi_gpu"] = nvidia
    try:
        import torch

        env["torch_version"] = getattr(torch, "__version__", "")
        env["torch_cuda"] = getattr(torch.version, "cuda", "")
        env["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        env["torch_version"] = ""
        env["torch_cuda"] = ""
        env["cudnn_version"] = ""
    try:
        import torchrec

        env["torchrec_version"] = getattr(torchrec, "__version__", "installed")
    except Exception:
        env["torchrec_version"] = ""
    return env


def _dense_arch_for_embedding_dim(embedding_dim: int) -> str:
    if int(embedding_dim) >= 128:
        return "512,256,128"
    return f"512,256,{int(embedding_dim)}"


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if any(value <= 0 for value in values):
        raise ValueError(f"all values must be positive: {raw}")
    return values


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def build_plan(profile: str, output_root: Path, overrides: PlanOverrides | None = None) -> ExperimentPlan:
    overrides = overrides or PlanOverrides()
    normalized = profile.strip().lower()
    if normalized == "smoke":
        data_rows = (4096,)
        batch_sizes = (256,)
        num_embeddings = (10000,)
        embedding_dims = (128,)
        steps = 3
        warmup_steps = 1
        repeat = 1
    elif normalized == "pilot":
        data_rows = (4096, 32768)
        batch_sizes = (512, 1024)
        num_embeddings = (50000, 200000)
        embedding_dims = (64, 128)
        steps = 20
        warmup_steps = 3
        repeat = 1
    elif normalized == "stress":
        data_rows = (131072, 524288)
        batch_sizes = (1024, 4096)
        num_embeddings = (200000, 800000)
        embedding_dims = (64, 128)
        steps = 30
        warmup_steps = 5
        repeat = 1
    elif normalized == "full":
        data_rows = (4096, 32768, 131072, 524288)
        batch_sizes = (512, 1024, 2048, 4096)
        num_embeddings = (50000, 200000, 800000)
        embedding_dims = (64, 128)
        steps = 60
        warmup_steps = 5
        repeat = 3
    else:
        raise ValueError(f"unknown profile: {profile}")

    lanes = [
        E2ELane(
            slug="torchrec-hbm-1p",
            label="TorchRec-HBM-1proc",
            backend="torchrec",
            nproc_per_node=1,
            torchrec_memory_mode="hbm",
        ),
        E2ELane(
            slug="torchrec-uvm-1p",
            label="TorchRec-UVMCache-1proc",
            backend="torchrec",
            nproc_per_node=1,
            torchrec_memory_mode="uvm_caching",
        ),
        E2ELane(
            slug="recstore-brpc-pet-1p",
            label="RecStore-BRPC-PET-1proc",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        ),
        E2ELane(
            slug="recstore-rdma-pet-1p",
            label="RecStore-RDMA-PET-1proc",
            backend="recstore",
            ps_type="RDMA",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        ),
        E2ELane(
            slug="recstore-brpc-eh-1p",
            label="RecStore-BRPC-EH-1proc",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_EXTENDIBLE_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        ),
        E2ELane(
            slug="recstore-grpc-pet-1p",
            label="RecStore-GRPC-PET-1proc",
            backend="recstore",
            ps_type="GRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
        ),
        E2ELane(
            slug="recstore-brpc-pet-prefetch4-1p",
            label="RecStore-BRPC-PET-prefetch4-1proc",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=1,
            prefetch_depth=4,
        ),
        E2ELane(
            slug="torchrec-hbm-2p",
            label="TorchRec-HBM-2proc",
            backend="torchrec",
            nproc_per_node=2,
            torchrec_memory_mode="hbm",
        ),
        E2ELane(
            slug="recstore-local-shm-pet-2p",
            label="RecStore-LOCAL_SHM-PET-2proc",
            backend="recstore",
            ps_type="LOCAL_SHM",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=2,
            enable_single_node_fast_path=True,
            single_node_ps_backend="local_shm",
        ),
        E2ELane(
            slug="recstore-brpc-pet-2p",
            label="RecStore-BRPC-PET-2proc",
            backend="recstore",
            ps_type="BRPC",
            recstore_index_type="DRAM_PET_HASH",
            ps_kv_backend="recstore_dram",
            nproc_per_node=2,
        ),
    ]
    if overrides.include_ablation_lanes:
        lanes.extend(
            [
                E2ELane(
                    slug="recstore-brpc-map-1p",
                    label="RecStore-BRPC-MAP-1proc",
                    backend="recstore",
                    ps_type="BRPC",
                    recstore_index_type="DRAM_UNORDERED_MAP",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-grpc-eh-1p",
                    label="RecStore-GRPC-EH-1proc",
                    backend="recstore",
                    ps_type="GRPC",
                    recstore_index_type="DRAM_EXTENDIBLE_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-local-shm-pet-1p",
                    label="RecStore-LOCAL_SHM-PET-1proc",
                    backend="recstore",
                    ps_type="LOCAL_SHM",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-rdma-eh-1p",
                    label="RecStore-RDMA-EH-1proc",
                    backend="recstore",
                    ps_type="RDMA",
                    recstore_index_type="DRAM_EXTENDIBLE_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-rdma-map-1p",
                    label="RecStore-RDMA-MAP-1proc",
                    backend="recstore",
                    ps_type="RDMA",
                    recstore_index_type="DRAM_UNORDERED_MAP",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                ),
                E2ELane(
                    slug="recstore-brpc-pet-prefetch1-1p",
                    label="RecStore-BRPC-PET-prefetch1-1proc",
                    backend="recstore",
                    ps_type="BRPC",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                    prefetch_depth=1,
                ),
                E2ELane(
                    slug="recstore-rdma-pet-prefetch1-1p",
                    label="RecStore-RDMA-PET-prefetch1-1proc",
                    backend="recstore",
                    ps_type="RDMA",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                    prefetch_depth=1,
                ),
                E2ELane(
                    slug="recstore-brpc-pet-prefetch8-1p",
                    label="RecStore-BRPC-PET-prefetch8-1proc",
                    backend="recstore",
                    ps_type="BRPC",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                    prefetch_depth=8,
                ),
                E2ELane(
                    slug="recstore-rdma-pet-prefetch4-1p",
                    label="RecStore-RDMA-PET-prefetch4-1proc",
                    backend="recstore",
                    ps_type="RDMA",
                    recstore_index_type="DRAM_PET_HASH",
                    ps_kv_backend="recstore_dram",
                    nproc_per_node=1,
                    prefetch_depth=4,
                ),
            ]
        )
    if overrides.only_lanes:
        allowed = set(overrides.only_lanes)
        known = {lane.slug for lane in lanes}
        unknown = sorted(allowed - known)
        if unknown:
            raise ValueError(f"unknown lane slug(s): {', '.join(unknown)}")
        lanes = [lane for lane in lanes if lane.slug in allowed]
    return ExperimentPlan(
        profile=normalized,
        output_root=output_root,
        data_rows=overrides.data_rows or data_rows,
        batch_sizes=overrides.batch_sizes or batch_sizes,
        num_embeddings=overrides.num_embeddings or num_embeddings,
        embedding_dims=overrides.embedding_dims or embedding_dims,
        steps=overrides.steps if overrides.steps is not None else steps,
        warmup_steps=overrides.warmup_steps if overrides.warmup_steps is not None else warmup_steps,
        repeat=overrides.repeat if overrides.repeat is not None else repeat,
        lanes=tuple(lanes),
    )


def _run(cmd: list[str], *, cwd: Path = ROOT, log_path: Path | None = None, dry_run: bool = False) -> int:
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            f.write("$ " + format_command(cmd) + "\n")
    if dry_run:
        return 0
    start = time.time()
    with (log_path.open("a", encoding="utf-8") if log_path else subprocess.DEVNULL) as sink:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=sink,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            check=False,
        )
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[exit_code] {proc.returncode}\n[elapsed_s] {time.time() - start:.3f}\n")
    return int(proc.returncode)


def prepare_dataset_slice(
    *,
    input_file: Path,
    rows: int,
    output_root: Path,
    dry_run: bool = False,
) -> Path:
    out_dir = output_root / "datasets" / f"criteo_day0_rows{rows}"
    marker = out_dir / "metadata.json"
    if marker.exists():
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            if int(metadata.get("accepted_rows", 0)) >= int(rows):
                return out_dir
        except json.JSONDecodeError:
            pass
    raw_dir = output_root / "raw_slices" / f"rows{rows}"
    cmd = [
        sys.executable,
        str(ROOT / "model_zoo/torchrec_dlrm/scripts/slice_preprocess_single_day.py"),
        "--input-file",
        str(input_file),
        "--output-raw-dir",
        str(raw_dir),
        "--output-dir",
        str(out_dir),
        "--rows",
        str(rows),
        "--seed",
        "20260330",
        "--progress-interval",
        "100000",
    ]
    log_path = output_root / "logs" / "dataset" / f"slice_rows{rows}.log"
    code = _run(cmd, log_path=log_path, dry_run=dry_run)
    if code != 0:
        raise RuntimeError(f"dataset slicing failed for rows={rows}; log={log_path}")
    return out_dir


def build_rs_demo_command(
    *,
    lane: E2ELane,
    context: ExecutionContext,
    run_id: str,
    data_dir: Path,
    output_root: Path,
    rows: int,
    batch_size: int,
    steps: int,
    warmup_steps: int,
    num_embeddings: int,
    embedding_dim: int,
    master_port: int,
) -> list[str]:
    del rows
    cmd = [
        context.python_bin,
        str(context.remote_repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
        "--backend",
        lane.backend,
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--data-dir",
        str(data_dir),
        "--steps",
        str(steps),
        "--warmup-steps",
        str(warmup_steps),
        "--batch-size",
        str(batch_size),
        "--num-embeddings",
        str(num_embeddings),
        "--embedding-dim",
        str(embedding_dim),
        "--dense-arch-layer-sizes",
        _dense_arch_for_embedding_dim(embedding_dim),
        "--nnodes",
        str(context.nnodes),
        "--node-rank",
        str(context.node_rank),
        "--nproc-per-node",
        str(lane.nproc_per_node),
        "--master-addr",
        context.master_addr,
        "--master-port",
        str(master_port),
        "--rdzv-id",
        run_id,
    ]
    if lane.backend == "torchrec":
        cmd.extend(["--no-start-server", "--torchrec-memory-mode", lane.torchrec_memory_mode])
    else:
        cmd.extend(
            [
                "--ps-type",
                lane.ps_type,
                "--recstore-index-type",
                lane.recstore_index_type,
                "--ps-kv-backend",
                lane.ps_kv_backend,
                "--allocator",
                lane.allocator,
                "--prefetch-depth",
                str(lane.prefetch_depth),
            ]
        )
        if lane.enable_single_node_fast_path:
            cmd.extend(
                [
                    "--enable-single-node-distributed-fast-path",
                    "--single-node-ps-backend",
                    lane.single_node_ps_backend,
                ]
            )
        if context.external_recstore_runtime_dir is not None:
            cmd.extend(["--recstore-runtime-dir", str(context.external_recstore_runtime_dir)])
        if context.no_start_recstore_server:
            cmd.append("--no-start-server")
        if context.server_host:
            cmd.extend(["--server-host", context.server_host])
        if context.server_port0 is not None:
            cmd.extend(["--server-port0", str(context.server_port0)])
        if context.server_port1 is not None:
            cmd.extend(["--server-port1", str(context.server_port1)])
    cmd.extend(lane.extra_args)
    return cmd


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def wrap_remote_command(cmd: list[str], host: str, *, cwd: Path) -> list[str]:
    remote = "cd {cwd} && {cmd}".format(
        cwd=shlex.quote(str(cwd)),
        cmd=" ".join(shlex.quote(part) for part in cmd),
    )
    return ["ssh", host, remote]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _warm_rows(path: Path) -> list[dict[str, str]]:
    rows = _read_csv(path)
    return [row for row in rows if str(row.get("warmup_excluded", "0")) not in {"1", "true", "True"}]


def _mean(rows: Iterable[dict[str, str]], column: str) -> float:
    vals = [float(row[column]) for row in rows if row.get(column, "") not in {"", "nan", "NaN"}]
    if not vals:
        return 0.0
    return statistics.fmean(vals)


def _p95(rows: Iterable[dict[str, str]], column: str) -> float:
    vals = sorted(float(row[column]) for row in rows if row.get(column, "") not in {"", "nan", "NaN"})
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    idx = int(round((len(vals) - 1) * 0.95))
    return vals[idx]


def collect_e2e_summary(
    *,
    manifest: list[dict[str, Any]],
    output_root: Path,
) -> list[dict[str, Any]]:
    del output_root
    rows_out: list[dict[str, Any]] = []
    for item in manifest:
        if item.get("status") != "ok":
            rows_out.append(
                {
                    **item,
                    "mean_step_total_ms": 0.0,
                    "p95_step_total_ms": 0.0,
                    "samples_per_sec": 0.0,
                    "lookup_mrows_per_sec": 0.0,
                    "update_mrows_per_sec": 0.0,
                }
            )
            continue
        main_csv = Path(str(item["main_csv"]))
        if not main_csv.exists():
            rows_out.append(
                {
                    **item,
                    "status": "missing_output",
                    "mean_step_total_ms": 0.0,
                    "p95_step_total_ms": 0.0,
                    "samples_per_sec": 0.0,
                    "lookup_mrows_per_sec": 0.0,
                    "update_mrows_per_sec": 0.0,
                }
            )
            continue
        warm = _warm_rows(main_csv)
        batch_size = int(item["batch_size"])
        mean_step = _mean(warm, "step_total_ms")
        mean_lookup = _mean(warm, "embed_lookup_local_ms")
        mean_update = _mean(warm, "sparse_update_ms")
        samples_per_sec = (
            batch_size * 1000.0 / mean_step if mean_step > 0.0 else 0.0
        )
        sparse_rows_per_step = batch_size * SPARSE_FEATURES_PER_SAMPLE
        lookup_mrows = (
            sparse_rows_per_step / (mean_lookup / 1000.0) / 1e6
            if mean_lookup > 0.0
            else 0.0
        )
        update_mrows = (
            sparse_rows_per_step / (mean_update / 1000.0) / 1e6
            if mean_update > 0.0
            else 0.0
        )
        rows_out.append(
            {
                **item,
                "mean_step_total_ms": mean_step,
                "p95_step_total_ms": _p95(warm, "step_total_ms"),
                "mean_embed_lookup_ms": mean_lookup,
                "mean_sparse_update_ms": mean_update,
                "samples_per_sec": samples_per_sec,
                "lookup_mrows_per_sec": lookup_mrows,
                "update_mrows_per_sec": update_mrows,
            }
        )
    return rows_out


@lru_cache(maxsize=512)
def _cached_text_file(path_text: str, max_bytes: int = 262_144) -> str:
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except OSError:
        return ""


def _status_reason(row: dict[str, Any]) -> str:
    explicit = str(row.get("skip_reason") or row.get("message") or "").strip()
    if explicit:
        return explicit
    if row.get("status") == "failed":
        log_text = _cached_text_file(str(row.get("log_path") or ""))
        lower_log = log_text.lower()
        if "out of memory" in lower_log or "oom" in lower_log:
            return "OOM during server/client startup; see per-run log"
        if "wait_server timeout" in log_text or "RdmaControlPlaneClient::WaitServer" in log_text:
            return "RDMA server did not publish ready before client init; see per-run log"
        if "SIGSEGV" in log_text or "SIGABRT" in log_text:
            return "native process crashed; see per-run log"
        return "benchmark failed; see per-run log"
    return str(row.get("log_path") or row.get("status") or "")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in _read_csv(path)]


def _config_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("rows", "")),
        str(row.get("batch_size", "")),
        str(row.get("num_embeddings", "")),
        str(row.get("embedding_dim", "")),
    )


def _lane_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (*_config_key(row), str(row.get("label", "")))


def _speedup(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_gap_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        if row.get("status") != "ok":
            continue
        grouped.setdefault(_config_key(row), []).append(row)

    out: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(int(v or 0) for v in item[0])):
        lane_medians: list[dict[str, Any]] = []
        lane_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            lane_groups.setdefault(_lane_key(row), []).append(row)
        for lane_rows in lane_groups.values():
            values = [_to_float(row.get("samples_per_sec")) for row in lane_rows]
            positive = [value for value in values if value > 0.0]
            if not positive:
                continue
            representative = dict(lane_rows[0])
            representative["samples_per_sec"] = statistics.median(positive)
            representative["repeat_count"] = len(positive)
            lane_medians.append(representative)

        recstore_lanes = [row for row in lane_medians if row.get("backend") == "recstore"]
        if not recstore_lanes:
            continue
        best_recstore = max(
            recstore_lanes,
            key=lambda row: _to_float(row.get("samples_per_sec")),
        )
        hbm = next(
            (
                row
                for row in lane_medians
                if row.get("backend") == "torchrec"
                and row.get("torchrec_memory_mode") == "hbm"
            ),
            None,
        )
        uvm = next(
            (
                row
                for row in lane_medians
                if row.get("backend") == "torchrec"
                and row.get("torchrec_memory_mode") == "uvm_caching"
            ),
            None,
        )
        recstore_sps = float(best_recstore.get("samples_per_sec", 0.0))
        hbm_sps = float(hbm.get("samples_per_sec", 0.0)) if hbm else 0.0
        uvm_sps = float(uvm.get("samples_per_sec", 0.0)) if uvm else 0.0
        if hbm_sps <= 0.0 and uvm_sps <= 0.0:
            continue
        out.append(
            {
                "rows": key[0],
                "batch_size": key[1],
                "num_embeddings": key[2],
                "embedding_dim": key[3],
                "best_recstore_label": best_recstore.get("label", ""),
                "best_recstore_samples_per_sec": recstore_sps,
                "torchrec_hbm_samples_per_sec": hbm_sps,
                "torchrec_uvm_samples_per_sec": uvm_sps,
                "recstore_vs_hbm_speedup": _speedup(recstore_sps, hbm_sps),
                "recstore_vs_uvm_speedup": _speedup(recstore_sps, uvm_sps),
                "best_recstore_repeat_count": best_recstore.get("repeat_count", 0),
                "torchrec_hbm_repeat_count": hbm.get("repeat_count", 0) if hbm else 0,
                "torchrec_uvm_repeat_count": uvm.get("repeat_count", 0) if uvm else 0,
            }
        )
    return out


def build_result_insights(
    *,
    summary_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    ps_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> list[str]:
    insights: list[str] = []
    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    if gap_rows:
        vs_hbm = [_to_float(row.get("recstore_vs_hbm_speedup")) for row in gap_rows]
        vs_uvm = [_to_float(row.get("recstore_vs_uvm_speedup")) for row in gap_rows]
        hbm_wins = sum(1 for value in vs_hbm if value >= 1.0)
        uvm_wins = sum(1 for value in vs_uvm if value >= 1.0)
        insights.append(
            "在 {total} 个可配对配置中，最佳 RecStore 路径有 {hbm_wins} 个配置快于 TorchRec-HBM，"
            "{uvm_wins} 个配置快于 TorchRec-UVMCache；RecStore/HBM 几何均值为 {hbm_geo:.2f}x，"
            "RecStore/UVM 几何均值为 {uvm_geo:.2f}x。".format(
                total=len(gap_rows),
                hbm_wins=hbm_wins,
                uvm_wins=uvm_wins,
                hbm_geo=_geomean(vs_hbm),
                uvm_geo=_geomean(vs_uvm),
            )
        )
        small_batch = [
            _to_float(row.get("recstore_vs_uvm_speedup"))
            for row in gap_rows
            if int(row.get("batch_size", 0) or 0) <= 1024
        ]
        large_batch = [
            _to_float(row.get("recstore_vs_uvm_speedup"))
            for row in gap_rows
            if int(row.get("batch_size", 0) or 0) >= 4096
        ]
        if small_batch and large_batch:
            insights.append(
                "batch size 对结论有明显影响：batch<=1024 时 RecStore/UVM 几何均值为 "
                f"{_geomean(small_batch):.2f}x，而 batch>=4096 时为 {_geomean(large_batch):.2f}x；"
                "因此大 batch 下 TorchRec-UVM 的 GPU 侧批处理优势需要单独报告，不能被小 batch 结果掩盖。"
            )
        best = max(gap_rows, key=lambda row: _to_float(row.get("recstore_vs_uvm_speedup")))
        worst = min(gap_rows, key=lambda row: _to_float(row.get("recstore_vs_uvm_speedup")))
        insights.append(
            "相对 TorchRec-UVM 的最佳配置是 rows={rows}, batch={batch}, emb_rows={emb}, dim={dim}，"
            "速度比为 {best:.2f}x；最弱配置是 rows={wrows}, batch={wbatch}, emb_rows={wemb}, dim={wdim}，"
            "速度比为 {worst:.2f}x。".format(
                rows=best.get("rows", ""),
                batch=best.get("batch_size", ""),
                emb=best.get("num_embeddings", ""),
                dim=best.get("embedding_dim", ""),
                best=_to_float(best.get("recstore_vs_uvm_speedup")),
                wrows=worst.get("rows", ""),
                wbatch=worst.get("batch_size", ""),
                wemb=worst.get("num_embeddings", ""),
                wdim=worst.get("embedding_dim", ""),
                worst=_to_float(worst.get("recstore_vs_uvm_speedup")),
            )
        )
    prefetch_rows = [
        row for row in ok_rows
        if row.get("backend") == "recstore" and int(row.get("prefetch_depth", 0) or 0) > 0
    ]
    if prefetch_rows:
        grouped_no_prefetch: dict[tuple[str, str, str, str], float] = {}
        for row in ok_rows:
            if row.get("backend") != "recstore" or int(row.get("prefetch_depth", 0) or 0) != 0:
                continue
            key = _config_key(row)
            grouped_no_prefetch[key] = max(
                grouped_no_prefetch.get(key, 0.0),
                _to_float(row.get("samples_per_sec")),
            )
        ratios = []
        for row in prefetch_rows:
            base = grouped_no_prefetch.get(_config_key(row), 0.0)
            if base > 0.0:
                ratios.append(_to_float(row.get("samples_per_sec")) / base)
        if ratios:
            insights.append(
                "prefetch_depth>0 当前是消融项而非主结果：相对同配置最佳非 prefetch RecStore，"
                f"吞吐几何均值保留率为 {_geomean(ratios):.2f}x，应在论文中作为负例或待优化路径解释。"
            )
    run_phase = [
        row for row in ps_rows
        if row.get("status") == "ok" and str(row.get("phase", "")).lower() in {"run", "fetch", "steady"}
    ]
    if run_phase:
        mkeys = []
        for row in run_phase:
            if row.get("throughput_mkeys_sec", "") not in {"", None}:
                mkeys.append(_to_float(row.get("throughput_mkeys_sec")))
            elif row.get("key_ops_per_sec", "") not in {"", None}:
                mkeys.append(_to_float(row.get("key_ops_per_sec")) / 1e6)
        if mkeys:
            insights.append(
                f"RDMA 仅作为 PS/network 层校准：run/fetch phase 中位吞吐为 {statistics.median(mkeys):.2f} M keys/s，"
                "不能直接写入 PyTorch/model 主表或等价为端到端训练加速。"
            )
    if int(metadata.get("gpu_count", 0) or 0) < 2:
        insights.append("当前机器 GPU 数不足 2，单机多卡行只能标记 skipped；投稿前需要在 2/4/8 GPU 节点补跑扩展性。")
    return insights


def _geomean(values: Iterable[float]) -> float:
    positive = [value for value in values if value > 0.0]
    if not positive:
        return 0.0
    return float(statistics.geometric_mean(positive))


def combine_existing_roots(roots: list[Path], output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    ps_rows: list[dict[str, Any]] = []
    for root in roots:
        source_profile = root.name.replace("paper_e2e_", "")
        for row in _load_manifest(root / "manifest.csv"):
            row.setdefault("source_profile", source_profile)
            row.setdefault("source_root", str(root))
            manifest.append(row)
        ps_path = root / "summary_ps_network.csv"
        if ps_path.exists():
            for row in _read_csv(ps_path):
                row.setdefault("source_profile", source_profile)
                row.setdefault("source_root", str(root))
                ps_rows.append(row)
    if not manifest:
        raise RuntimeError("no manifest rows found in --combine-roots inputs")
    manifest = _dedupe_manifest_for_report(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "manifest.csv", manifest)
    if ps_rows:
        _write_csv(output_root / "summary_ps_network.csv", ps_rows)
    return manifest, ps_rows


def _manifest_config_key(row: dict[str, Any]) -> tuple[str, ...]:
    config_fields = (
        "slug",
        "rows",
        "batch_size",
        "num_embeddings",
        "embedding_dim",
        "nproc_per_node",
        "repeat",
    )
    if not all(str(row.get(field, "")) for field in config_fields):
        return (str(row.get("run_id", "")),)
    return (
        str(row.get("slug", "")),
        str(row.get("backend", "")),
        str(row.get("ps_type", "")),
        str(row.get("recstore_index_type", "")),
        str(row.get("ps_kv_backend", "")),
        str(row.get("torchrec_memory_mode", "")),
        str(row.get("prefetch_depth", "")),
        str(row.get("rows", "")),
        str(row.get("batch_size", "")),
        str(row.get("num_embeddings", "")),
        str(row.get("embedding_dim", "")),
        str(row.get("nproc_per_node", "")),
        str(row.get("repeat", "")),
    )


def _dedupe_manifest_for_report(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a repaired successful rerun instead of an older failed duplicate."""
    selected: dict[tuple[str, ...], dict[str, Any]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = _manifest_config_key(row)
        if key not in selected:
            selected[key] = row
            order.append(key)
            continue
        current = selected[key]
        row_ok = row.get("status") == "ok"
        current_ok = current.get("status") == "ok"
        if row_ok and not current_ok:
            selected[key] = row
        elif row_ok == current_ok:
            selected[key] = row
    return [selected[key] for key in order]


def run_e2e_plan(
    plan: ExperimentPlan,
    *,
    input_file: Path,
    context: ExecutionContext | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    context = context or ExecutionContext()
    manifest: list[dict[str, Any]] = []
    run_index = 0
    detected_gpus = _gpu_count()
    for rows in plan.data_rows:
        dataset_dir = prepare_dataset_slice(
            input_file=input_file,
            rows=rows,
            output_root=plan.output_root,
            dry_run=dry_run,
        )
        for batch_size in plan.batch_sizes:
            train_rows = int(rows * 0.8)
            if batch_size > train_rows:
                continue
            for num_embeddings in plan.num_embeddings:
                for embedding_dim in plan.embedding_dims:
                    for repeat in range(1, plan.repeat + 1):
                        for lane in plan.lanes:
                            if lane.ps_type.upper() == "RDMA" and not _has_rdma():
                                manifest.append(
                                    {
                                        "run_id": "",
                                        "label": lane.label,
                                        "slug": lane.slug,
                                        "backend": lane.backend,
                                        "ps_type": lane.ps_type,
                                        "recstore_index_type": lane.recstore_index_type,
                                        "ps_kv_backend": lane.ps_kv_backend,
                                        "torchrec_memory_mode": lane.torchrec_memory_mode,
                                        "prefetch_depth": lane.prefetch_depth,
                                        "rows": rows,
                                        "batch_size": batch_size,
                                        "num_embeddings": num_embeddings,
                                        "embedding_dim": embedding_dim,
                                        "nproc_per_node": lane.nproc_per_node,
                                        "repeat": repeat,
                                        "status": "skipped",
                                        "exit_code": "",
                                        "skip_reason": "RDMA verbs devices are not available",
                                        "main_csv": "",
                                        "log_path": "",
                                        "command": "",
                                    }
                                )
                                continue
                            if lane.nproc_per_node > max(detected_gpus, 1):
                                manifest.append(
                                    {
                                        "run_id": "",
                                        "label": lane.label,
                                        "slug": lane.slug,
                                        "backend": lane.backend,
                                        "ps_type": lane.ps_type,
                                        "recstore_index_type": lane.recstore_index_type,
                                        "ps_kv_backend": lane.ps_kv_backend,
                                        "torchrec_memory_mode": lane.torchrec_memory_mode,
                                        "prefetch_depth": lane.prefetch_depth,
                                        "rows": rows,
                                        "batch_size": batch_size,
                                        "num_embeddings": num_embeddings,
                                        "embedding_dim": embedding_dim,
                                        "nproc_per_node": lane.nproc_per_node,
                                        "repeat": repeat,
                                        "status": "skipped",
                                        "exit_code": "",
                                        "skip_reason": (
                                            f"requires {lane.nproc_per_node} local CUDA devices, "
                                            f"detected {detected_gpus}"
                                        ),
                                        "main_csv": "",
                                        "log_path": "",
                                        "command": "",
                                    }
                                )
                                continue
                            run_index += 1
                            run_id = (
                                f"paper-{plan.profile}-{lane.slug}-"
                                f"r{rows}-b{batch_size}-n{num_embeddings}-"
                                f"d{embedding_dim}-rep{repeat}"
                            )
                            main_csv = (
                                plan.output_root
                                / "outputs"
                                / run_id
                                / ("torchrec_main.csv" if lane.backend == "torchrec" else "recstore_main.csv")
                            )
                            cmd = build_rs_demo_command(
                                lane=lane,
                                context=context,
                                run_id=run_id,
                                data_dir=dataset_dir,
                                output_root=plan.output_root,
                                rows=rows,
                                batch_size=batch_size,
                                steps=plan.steps,
                                warmup_steps=plan.warmup_steps,
                                num_embeddings=num_embeddings,
                                embedding_dim=embedding_dim,
                                master_port=29600 + (run_index % 1000),
                            )
                            run_cmd = (
                                wrap_remote_command(cmd, context.remote_train_host, cwd=context.remote_repo_root)
                                if context.remote_train_host
                                else cmd
                            )
                            log_path = plan.output_root / "logs" / "e2e" / f"{run_id}.log"
                            status = "dry_run" if dry_run else "ok"
                            exit_code = _run(run_cmd, log_path=log_path, dry_run=dry_run)
                            if not dry_run and (exit_code != 0 or not main_csv.exists()):
                                status = "failed"
                            manifest.append(
                                {
                                    "run_id": run_id,
                                    "label": lane.label,
                                    "slug": lane.slug,
                                    "backend": lane.backend,
                                    "ps_type": lane.ps_type,
                                    "recstore_index_type": lane.recstore_index_type,
                                    "ps_kv_backend": lane.ps_kv_backend,
                                    "torchrec_memory_mode": lane.torchrec_memory_mode,
                                    "prefetch_depth": lane.prefetch_depth,
                                    "rows": rows,
                                    "batch_size": batch_size,
                                    "num_embeddings": num_embeddings,
                                    "embedding_dim": embedding_dim,
                                    "nproc_per_node": lane.nproc_per_node,
                                    "repeat": repeat,
                                    "status": status,
                                    "exit_code": exit_code,
                                    "main_csv": str(main_csv),
                                    "log_path": str(log_path),
                                    "command": format_command(run_cmd),
                                    "remote_train_host": context.remote_train_host,
                                    "server_host": context.server_host,
                                    "recstore_runtime_dir": (
                                        str(context.external_recstore_runtime_dir)
                                        if context.external_recstore_runtime_dir is not None
                                        else ""
                                    ),
                                }
                            )
    return manifest


def run_rdma_ps_calibration(
    *,
    output_root: Path,
    profile: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if not _has_rdma():
        return [{"layer": "PS/network", "status": "skipped", "reason": "no RDMA verbs device"}]
    runtime_seconds = "1" if profile == "smoke" else "5"
    record_count = "10000" if profile == "smoke" else "1000000"
    batch_keys = "64" if profile == "smoke" else "500"
    client_processes = "1" if profile == "smoke" else "6"
    server_threads = "1" if profile == "smoke" else "16"
    out_dir = output_root / "ps_network" / f"rdma_{profile}"
    cmd = [
        sys.executable,
        str(ROOT / "tools/benchmarks/run_benchmark_ps.py"),
        "--transports",
        "rdma",
        "--client-ips",
        "127.0.0.1",
        "--server-shard-ips",
        "127.0.0.1",
        "--client-processes-per-ip",
        client_processes,
        "--record-count",
        record_count,
        "--value-size",
        "512",
        "--batch-keys",
        batch_keys,
        "--index-type",
        "DRAM_PET_HASH",
        "--client-threads-per-process",
        "1",
        "--client-load-threads-per-process",
        "1",
        "--runtime-seconds",
        runtime_seconds,
        "--repeat",
        "1",
        "--execution-backend",
        "local",
        "--prefetch-depth",
        "16",
        "--rdma-rc-qps-per-client-per-shard",
        "16",
        "--rdma-rc-slots-per-qp",
        "1",
        "--server-rdma-threads",
        server_threads,
        "--rdma-rc-server-get-workers",
        "0",
        "--rdma-rc-server-coroutines-per-thread",
        "1",
        "--rdma-get-response-mode",
        "auto",
        "--output-dir",
        str(out_dir),
    ]
    log_path = output_root / "logs" / "ps_network" / f"rdma_{profile}.log"
    exit_code = _run(cmd, log_path=log_path, dry_run=dry_run)
    summary_csv = out_dir / "summary.csv"
    if exit_code != 0 or (not dry_run and not summary_csv.exists()):
        return [
            {
                "layer": "PS/network",
                "status": "failed",
                "exit_code": exit_code,
                "summary_csv": str(summary_csv),
                "log_path": str(log_path),
                "command": " ".join(cmd),
            }
        ]
    rows = _read_csv(summary_csv) if not dry_run else []
    for row in rows:
        row["layer"] = "PS/network"
        row["status"] = "ok"
        row["summary_csv"] = str(summary_csv)
        row["log_path"] = str(log_path)
        row["command"] = " ".join(cmd)
    return rows


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
        ("final", "paper_e2e_report.tex", 1, output_root),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run paper-oriented RecStore vs TorchRec E2E benchmark matrix.")
    parser.add_argument("--profile", choices=["smoke", "pilot", "stress", "full"], default="pilot")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_NAS_DAY0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-rdma-ps", action="store_true")
    parser.add_argument("--data-rows", default="", help="Comma-separated override for dataset row counts.")
    parser.add_argument("--batch-sizes", default="", help="Comma-separated override for batch sizes.")
    parser.add_argument("--num-embeddings", default="", help="Comma-separated override for embedding table rows.")
    parser.add_argument("--embedding-dims", default="", help="Comma-separated override for embedding dimensions.")
    parser.add_argument("--steps", type=int, default=None, help="Override measured steps per run.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override warmup steps per run.")
    parser.add_argument("--repeat", type=int, default=None, help="Override repeats per configuration.")
    parser.add_argument(
        "--only-lanes",
        default="",
        help="Comma-separated lane slugs to run after lane construction.",
    )
    parser.add_argument(
        "--include-ablation-lanes",
        action="store_true",
        help="Include extra 1P RecStore backend/transport/prefetch ablation lanes.",
    )
    parser.add_argument(
        "--remote-train-host",
        default="",
        help="Run E2E benchmark commands through ssh on this host/container.",
    )
    parser.add_argument(
        "--remote-repo-root",
        type=Path,
        default=ROOT,
        help="Repository path visible from --remote-train-host.",
    )
    parser.add_argument(
        "--remote-python-bin",
        default=sys.executable,
        help="Python executable used on --remote-train-host.",
    )
    parser.add_argument("--nnodes", type=int, default=1, help="torchrun/rs_demo node count.")
    parser.add_argument("--node-rank", type=int, default=0, help="torchrun/rs_demo node rank for this runner.")
    parser.add_argument("--master-addr", default="127.0.0.1", help="torchrun master address.")
    parser.add_argument(
        "--external-recstore-runtime-dir",
        type=Path,
        default=None,
        help="Existing RecStore runtime dir/config visible to the train host.",
    )
    parser.add_argument(
        "--no-start-recstore-server",
        action="store_true",
        help="Pass --no-start-server for RecStore lanes so training uses an external PS.",
    )
    parser.add_argument("--server-host", default="", help="RecStore PS host passed to RecStore lanes.")
    parser.add_argument("--server-port0", type=int, default=None, help="RecStore shard-0 port.")
    parser.add_argument("--server-port1", type=int, default=None, help="RecStore shard-1 port.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Regenerate summary CSV and LaTeX from existing manifest/results.",
    )
    parser.add_argument(
        "--combine-roots",
        type=Path,
        nargs="+",
        default=[],
        help="Combine existing paper_e2e output roots before regenerating summaries.",
    )
    args = parser.parse_args(argv)

    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.repeat is not None and args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.nnodes <= 0:
        raise ValueError("--nnodes must be positive")
    if args.node_rank < 0:
        raise ValueError("--node-rank must be non-negative")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    overrides = PlanOverrides(
        data_rows=_parse_int_tuple(args.data_rows),
        batch_sizes=_parse_int_tuple(args.batch_sizes),
        num_embeddings=_parse_int_tuple(args.num_embeddings),
        embedding_dims=_parse_int_tuple(args.embedding_dims),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        repeat=args.repeat,
        only_lanes=_parse_str_tuple(args.only_lanes),
        include_ablation_lanes=args.include_ablation_lanes,
    )
    plan = build_plan(args.profile, output_root, overrides)
    context = ExecutionContext(
        remote_train_host=args.remote_train_host,
        remote_repo_root=args.remote_repo_root,
        python_bin=args.remote_python_bin,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        external_recstore_runtime_dir=args.external_recstore_runtime_dir,
        no_start_recstore_server=args.no_start_recstore_server,
        server_host=args.server_host,
        server_port0=args.server_port0,
        server_port1=args.server_port1,
    )
    metadata = {
        "profile": plan.profile,
        "output_root": str(output_root),
        "input_file": str(args.input_file),
        "gpu_count": _gpu_count(),
        "rdma_available": _has_rdma(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_rows": list(plan.data_rows),
        "batch_sizes": list(plan.batch_sizes),
        "num_embeddings": list(plan.num_embeddings),
        "embedding_dims": list(plan.embedding_dims),
        "steps": plan.steps,
        "warmup_steps": plan.warmup_steps,
        "repeat": plan.repeat,
        "lanes": [lane.slug for lane in plan.lanes],
        "remote_train_host": context.remote_train_host,
        "remote_repo_root": str(context.remote_repo_root),
        "nnodes": context.nnodes,
        "node_rank": context.node_rank,
        "master_addr": context.master_addr,
        "external_recstore_runtime_dir": (
            str(context.external_recstore_runtime_dir)
            if context.external_recstore_runtime_dir is not None
            else ""
        ),
        "no_start_recstore_server": context.no_start_recstore_server,
        "server_host": context.server_host,
        "server_port0": context.server_port0 if context.server_port0 is not None else "",
        "server_port1": context.server_port1 if context.server_port1 is not None else "",
    }
    metadata.update(_collect_environment_metadata())
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.combine_roots:
        manifest, combined_ps_rows = combine_existing_roots(args.combine_roots, output_root)
    elif args.aggregate_only:
        manifest = _load_manifest(output_root / "manifest.csv")
        combined_ps_rows = []
    else:
        manifest = (
            []
            if args.skip_e2e
            else run_e2e_plan(plan, input_file=args.input_file, context=context, dry_run=args.dry_run)
        )
        combined_ps_rows = []
    _write_csv(output_root / "manifest.csv", manifest)
    summary_rows = collect_e2e_summary(manifest=manifest, output_root=output_root)
    _write_csv(output_root / "summary_e2e.csv", summary_rows)
    gap_rows = build_gap_summary(summary_rows)
    _write_csv(output_root / "summary_gap.csv", gap_rows)

    if args.combine_roots:
        ps_rows = combined_ps_rows
    elif args.skip_rdma_ps:
        ps_rows = _read_csv(output_root / "summary_ps_network.csv") if (output_root / "summary_ps_network.csv").exists() else []
    else:
        ps_rows = run_rdma_ps_calibration(
            output_root=output_root,
            profile=plan.profile,
            dry_run=args.dry_run,
        )
        _write_csv(output_root / "summary_ps_network.csv", ps_rows)

    svg_paths = write_svg_figures(
        output_root,
        summary_rows=summary_rows,
        gap_rows=gap_rows,
        ps_rows=ps_rows,
    )
    metadata["svg_figures"] = [str(path.relative_to(output_root)) for path in svg_paths]
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = render_latex_report(
        summary_rows=summary_rows,
        ps_rows=ps_rows,
        gap_rows=gap_rows,
        metadata=metadata,
    )
    report_path = output_root / "paper_e2e_report.tex"
    report_path.write_text(report, encoding="utf-8")
    print(f"[paper-e2e] output_root={output_root}")
    print(f"[paper-e2e] manifest={output_root / 'manifest.csv'}")
    print(f"[paper-e2e] e2e_summary={output_root / 'summary_e2e.csv'}")
    print(f"[paper-e2e] gap_summary={output_root / 'summary_gap.csv'}")
    print(f"[paper-e2e] ps_summary={output_root / 'summary_ps_network.csv'}")
    print(f"[paper-e2e] latex={report_path}")
    print(f"[paper-e2e] svg_figures={output_root / 'figures'} ({len(svg_paths)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
