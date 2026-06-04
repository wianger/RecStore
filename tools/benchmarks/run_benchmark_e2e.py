from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SPARSE_FEATURES_PER_SAMPLE = 26
DEFAULT_OUTPUT_DIR = Path("results") / time.strftime("brpc_e2e_%m%d%H%M")


@dataclass(frozen=True)
class ClientSpec:
    ssh_host: str = "local"
    repo_root: Path = ROOT
    ip: str = "127.0.0.1"
    gpu_id: int = 0
    node_rank: int = 0
    nproc_per_node: int = 1


@dataclass(frozen=True)
class ServerSpec:
    ssh_host: str = "local"
    repo_root: Path = ROOT
    ip: str = "127.0.0.1"
    port: int = 15000
    shard_id: int = 0


@dataclass(frozen=True)
class BenchmarkConfig:
    clients: tuple[ClientSpec, ...]
    servers: tuple[ServerSpec, ...]
    output_dir: Path = DEFAULT_OUTPUT_DIR
    runtime_dir: Path | None = None
    dataset_path: Path = Path("model_zoo/torchrec_dlrm/processed_day_0_data")
    model: str = "dlrm"
    batch_size: int = 1024
    embedding_dim: int = 128
    num_embeddings: int = 200000
    init_rows: int = 50000
    steps: int = 80
    warmup_steps: int = 5
    repeat: int = 3
    read_mode: str = "prefetch"
    prefetch_depth: int = 0
    index_type: str = "DRAM_PET_HASH"
    torchrec_baselines: tuple[str, ...] = ("hbm",)
    master_port: int = 29500
    python_bin: str = sys.executable
    skip_build: bool = False
    skip_tests: bool = False

    @property
    def resolved_runtime_dir(self) -> Path:
        return self.runtime_dir or (self.output_dir / "runtime")


def _parse_key_values(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected key=value item in spec: {part!r}")
        key, value = part.split("=", 1)
        out[key.strip().lower()] = value.strip()
    return out


def parse_client_spec(raw: str) -> ClientSpec:
    values = _parse_key_values(raw)
    return ClientSpec(
        ssh_host=values.get("ssh", values.get("ssh_host", "local")),
        repo_root=Path(values.get("repo", values.get("repo_root", str(ROOT)))),
        ip=values.get("ip", "127.0.0.1"),
        gpu_id=int(values.get("gpu", values.get("gpu_id", "0"))),
        node_rank=int(values.get("node_rank", values.get("rank", "0"))),
        nproc_per_node=int(values.get("nproc", values.get("nproc_per_node", "1"))),
    )


def parse_server_spec(raw: str) -> ServerSpec:
    values = _parse_key_values(raw)
    return ServerSpec(
        ssh_host=values.get("ssh", values.get("ssh_host", "local")),
        repo_root=Path(values.get("repo", values.get("repo_root", str(ROOT)))),
        ip=values.get("ip", "127.0.0.1"),
        port=int(values.get("port", "15000")),
        shard_id=int(values.get("shard", values.get("shard_id", "0"))),
    )


def infer_client_deployment(clients: tuple[ClientSpec, ...]) -> str:
    ips = {client.ip for client in clients}
    ranks = {client.node_rank for client in clients}
    return "distributed" if len(ips) > 1 or len(ranks) > 1 else "single-node"


def infer_ps_deployment(servers: tuple[ServerSpec, ...]) -> str:
    return "sharded-ps" if len(servers) > 1 else "single-ps"


def _dense_arch_for_embedding_dim(embedding_dim: int) -> str:
    if int(embedding_dim) >= 128:
        return "512,256,128"
    return f"512,256,{int(embedding_dim)}"


def _server_entries(servers: tuple[ServerSpec, ...]) -> list[dict[str, Any]]:
    return [
        {"host": server.ip, "port": int(server.port), "shard": int(server.shard_id)}
        for server in sorted(servers, key=lambda item: item.shard_id)
    ]


def estimate_runtime_capacity(num_embeddings: int, init_rows: int) -> int:
    # DLRM uses one sparse id per table, so the PS key space spans all tables.
    per_table_rows = max(int(num_embeddings), int(init_rows))
    return max(per_table_rows * SPARSE_FEATURES_PER_SAMPLE * 2, 100_000)


def build_runtime_config(cfg: BenchmarkConfig, *, transport: str, value_path: Path) -> dict[str, Any]:
    transport_upper = transport.upper()
    if transport_upper not in {"BRPC", "GRPC"}:
        raise ValueError(f"unsupported E2E transport: {transport}")
    value_size = int(cfg.embedding_dim) * 4
    servers = _server_entries(cfg.servers)
    capacity = estimate_runtime_capacity(cfg.num_embeddings, cfg.init_rows)
    capacity_bytes = capacity * value_size * 2
    base_kv = {
        "capacity": capacity,
        "index": {"type": cfg.index_type},
        "value": {
            "type": "DRAM_VALUE_STORE",
            "path": str(value_path),
            "default_value_size_hint": value_size,
            "dram_allocator": {
                "type": "PERSIST_LOOP_SLAB",
                "capacity_bytes": capacity_bytes,
            },
        },
    }
    return {
        "cache_ps": {
            "ps_type": transport_upper,
            "max_batch_keys_size": 65536,
            "num_threads": 32,
            "num_shards": len(servers),
            "servers": servers,
            "base_kv_config": base_kv,
        },
        "distributed_client": {
            "num_shards": len(servers),
            "hash_method": "city_hash",
            "max_keys_per_request": 65536,
            "servers": servers,
        },
        "client": servers[0],
    }


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _wrap_remote(cmd: list[str], *, ssh_host: str, cwd: Path) -> list[str]:
    if ssh_host in {"", "local", "localhost"}:
        return cmd
    remote = "cd {cwd} && {cmd}".format(
        cwd=shlex.quote(str(cwd)),
        cmd=_format_command(cmd),
    )
    return ["ssh", ssh_host, remote]


def build_server_command(*, server: ServerSpec, runtime_config: Path, transport: str) -> list[str]:
    cmd = [str(server.repo_root / "build/bin/ps_server"), "--config_path", str(runtime_config)]
    if transport.upper() == "BRPC":
        cmd.extend(["--brpc_server_port", str(server.port)])
    return _wrap_remote(cmd, ssh_host=server.ssh_host, cwd=server.repo_root)


def _client_node_count(clients: tuple[ClientSpec, ...]) -> int:
    return max((client.node_rank for client in clients), default=0) + 1


def build_client_command(
    *,
    cfg: BenchmarkConfig,
    transport: str,
    client: ClientSpec,
    run_id: str,
) -> list[str]:
    first_server = sorted(cfg.servers, key=lambda item: item.shard_id)[0]
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={int(client.gpu_id)}",
        cfg.python_bin,
        str(client.repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
        "--backend",
        "recstore",
        "--ps-type",
        transport.upper(),
        "--recstore-index-type",
        cfg.index_type,
        "--ps-kv-backend",
        "recstore_dram",
        "--batch-size",
        str(cfg.batch_size),
        "--embedding-dim",
        str(cfg.embedding_dim),
        "--num-embeddings",
        str(cfg.num_embeddings),
        "--init-rows",
        str(cfg.init_rows),
        "--steps",
        str(cfg.steps),
        "--warmup-steps",
        str(cfg.warmup_steps),
        "--read-mode",
        cfg.read_mode,
        "--prefetch-depth",
        str(cfg.prefetch_depth),
        "--dense-arch-layer-sizes",
        _dense_arch_for_embedding_dim(cfg.embedding_dim),
        "--data-dir",
        str(cfg.dataset_path),
        "--output-root",
        str(cfg.output_dir),
        "--run-id",
        run_id,
        "--recstore-runtime-dir",
        str(cfg.resolved_runtime_dir / transport.lower()),
        "--no-start-server",
        "--server-host",
        first_server.ip,
        "--server-port0",
        str(first_server.port),
        "--nnodes",
        str(_client_node_count(cfg.clients)),
        "--node-rank",
        str(client.node_rank),
        "--nproc-per-node",
        str(client.nproc_per_node),
        "--master-addr",
        cfg.clients[0].ip,
        "--master-port",
        str(cfg.master_port),
        "--rdzv-id",
        run_id,
    ]
    return _wrap_remote(cmd, ssh_host=client.ssh_host, cwd=client.repo_root)


def build_torchrec_command(
    *,
    cfg: BenchmarkConfig,
    memory_mode: str,
    client: ClientSpec,
    run_id: str,
) -> list[str]:
    output_dir = cfg.output_dir / "outputs" / run_id
    cmd = [
        "env",
        f"CUDA_VISIBLE_DEVICES={int(client.gpu_id)}",
        cfg.python_bin,
        str(client.repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
        "--backend",
        "torchrec",
        "--batch-size",
        str(cfg.batch_size),
        "--embedding-dim",
        str(cfg.embedding_dim),
        "--num-embeddings",
        str(cfg.num_embeddings),
        "--steps",
        str(cfg.steps),
        "--warmup-steps",
        str(cfg.warmup_steps),
        "--dense-arch-layer-sizes",
        _dense_arch_for_embedding_dim(cfg.embedding_dim),
        "--data-dir",
        str(cfg.dataset_path),
        "--output-root",
        str(cfg.output_dir),
        "--run-id",
        run_id,
        "--torchrec-memory-mode",
        memory_mode,
        "--torchrec-main-csv",
        str(output_dir / "torchrec_main.csv"),
        "--torchrec-main-agg-csv",
        str(output_dir / "torchrec_main_agg.csv"),
        "--no-start-server",
        "--nnodes",
        str(_client_node_count(cfg.clients)),
        "--node-rank",
        str(client.node_rank),
        "--nproc-per-node",
        str(client.nproc_per_node),
        "--master-addr",
        cfg.clients[0].ip,
        "--master-port",
        str(cfg.master_port),
        "--rdzv-id",
        run_id,
    ]
    return _wrap_remote(cmd, ssh_host=client.ssh_host, cwd=client.repo_root)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
    servers = "; ".join(
        f"{server.ip}:{server.port}/shard{server.shard_id}" for server in cfg.servers
    )
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _write_deployment(path: Path, cfg: BenchmarkConfig, transports: tuple[str, ...]) -> None:
    lines = [
        "# Benchmark E2E Deployment",
        "",
        f"模型: {cfg.model}",
        f"RecStore 传输: {','.join(transports)}",
        f"TorchRec baseline: {','.join(cfg.torchrec_baselines) or 'disabled'}",
        "client:",
    ]
    for client in cfg.clients:
        lines.append(
            f"  - ssh_host={client.ssh_host}, repo={client.repo_root}, ip={client.ip}, "
            f"gpu={client.gpu_id}, node_rank={client.node_rank}, nproc_per_node={client.nproc_per_node}"
        )
    lines.append("ps:")
    for server in cfg.servers:
        lines.append(
            f"  - ssh_host={server.ssh_host}, repo={server.repo_root}, ip={server.ip}, "
            f"port={server.port}, shard={server.shard_id}"
        )
    lines.extend(
        [
            f"client 部署: {infer_client_deployment(cfg.clients)}",
            f"PS 部署: {infer_ps_deployment(cfg.servers)}",
            f"分片: {len(cfg.servers)}, hash_method=city_hash, max_keys_per_request=65536",
            f"dataset: {cfg.dataset_path}",
            f"runtime: {cfg.resolved_runtime_dir}",
            f"output: {cfg.output_dir}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _start_process(cmd: list[str], *, log_path: Path, cwd: Path) -> subprocess.Popen[Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)


def _stop_processes(processes: list[subprocess.Popen[Any]]) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    for proc in processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _run_client_group(
    entries: list[dict[str, Any]],
    *,
    dry_run: bool,
    commands: list[str],
    manifest: list[dict[str, Any]],
) -> None:
    for entry in entries:
        commands.append(_format_command(entry["cmd"]))
    if dry_run:
        for entry in entries:
            manifest.append({**entry["row"], "status": "dry_run", "exit_code": ""})
        return

    running: list[tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = []
    try:
        for entry in entries:
            log_path = Path(entry["row"]["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                entry["cmd"],
                cwd=str(entry["cwd"]),
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running.append((proc, log, entry["row"]))
        for proc, log, row in running:
            returncode = proc.wait()
            log.close()
            manifest.append(
                {
                    **row,
                    "status": "ok" if returncode == 0 else "failed",
                    "exit_code": returncode,
                }
            )
    finally:
        for proc, log, _row in running:
            if proc.poll() is None:
                proc.terminate()
            if not log.closed:
                log.close()


def _parse_transports(raw: str) -> tuple[str, ...]:
    transports = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    if not transports:
        return ("BRPC",)
    unsupported = [item for item in transports if item not in {"BRPC", "GRPC"}]
    if unsupported:
        raise ValueError(f"unsupported transports: {unsupported}")
    return transports


def _parse_torchrec_baselines(raw: str) -> tuple[str, ...]:
    baselines = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unsupported = [item for item in baselines if item not in {"hbm", "uvm_caching"}]
    if unsupported:
        raise ValueError(f"unsupported TorchRec baselines: {unsupported}")
    return baselines


def _torchrec_label(memory_mode: str) -> str:
    if memory_mode == "uvm_caching":
        return "TorchRec-UVMCache"
    return "TorchRec-HBM"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _build_config_from_args(args: argparse.Namespace) -> tuple[BenchmarkConfig, tuple[str, ...]]:
    clients = tuple(parse_client_spec(raw) for raw in args.client) if args.client else (ClientSpec(),)
    servers = tuple(parse_server_spec(raw) for raw in args.ps) if args.ps else (ServerSpec(),)
    output_dir = Path(args.output_dir)
    runtime_dir = Path(args.runtime_dir) if args.runtime_dir else output_dir / "runtime"
    cfg = BenchmarkConfig(
        clients=clients,
        servers=servers,
        output_dir=output_dir,
        runtime_dir=runtime_dir,
        dataset_path=Path(args.data_dir),
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        init_rows=args.init_rows,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        repeat=args.repeat,
        read_mode=args.read_mode,
        prefetch_depth=args.prefetch_depth,
        index_type=args.index_type,
        torchrec_baselines=() if args.no_torchrec else _parse_torchrec_baselines(args.torchrec_baselines),
        master_port=args.master_port,
        python_bin=args.python_bin,
        skip_build=args.skip_build,
        skip_tests=args.skip_tests,
    )
    return cfg, _parse_transports(args.transports)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RecStore BRPC/GRPC E2E benchmarks.")
    parser.add_argument("--client", action="append", default=[])
    parser.add_argument("--ps", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--runtime-dir", default="")
    parser.add_argument("--transports", default="brpc")
    parser.add_argument("--data-dir", default="model_zoo/torchrec_dlrm/processed_day_0_data")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-embeddings", type=int, default=200000)
    parser.add_argument("--init-rows", type=int, default=50000)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--torchrec-baselines", default="hbm")
    parser.add_argument("--no-torchrec", action="store_true")
    parser.add_argument("--read-mode", choices=["prefetch", "direct"], default="prefetch")
    parser.add_argument("--prefetch-depth", type=int, default=0)
    parser.add_argument(
        "--index-type",
        choices=["DRAM_PET_HASH", "DRAM_EXTENDIBLE_HASH", "DRAM_UNORDERED_MAP"],
        default="DRAM_PET_HASH",
    )
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)

    cfg, transports = _build_config_from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = cfg.output_dir / "logs"
    _write_deployment(cfg.output_dir / "deployment.md", cfg, transports)

    commands: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    manifest: list[dict[str, Any]] = []

    if args.aggregate_only:
        manifest = _load_manifest(cfg.output_dir / "manifest.csv")
        summary_rows = collect_summary_rows(manifest)
        _write_csv(cfg.output_dir / "summary_e2e.csv", summary_rows)
        (cfg.output_dir / "summary.md").write_text(render_summary_md(cfg, summary_rows), encoding="utf-8")
        return 0

    if not cfg.skip_build and not args.dry_run:
        _run(["cmake", "-S", ".", "-B", "build"], cwd=ROOT)
        _run(["cmake", "--build", "build", "--target", "ps_server", "-j"], cwd=ROOT)
    if not cfg.skip_tests and not args.dry_run:
        _run(
            [
                "ctest",
                "-R",
                "brpc_ps_client_test|dist_brpc_ps_client_test|test_ps_server_launcher|test_ps_client_factory|test_allshards_ps_client",
                "--output-on-failure",
            ],
            cwd=ROOT,
        )

    for transport in transports:
        transport_lower = transport.lower()
        runtime_dir = cfg.resolved_runtime_dir / transport_lower
        config_path = runtime_dir / "recstore_config.json"
        runtime = build_runtime_config(
            cfg,
            transport=transport,
            value_path=runtime_dir / "value",
        )
        _write_json(config_path, runtime)
        processes: list[subprocess.Popen[Any]] = []
        try:
            for server in cfg.servers:
                server_cmd = build_server_command(server=server, runtime_config=config_path, transport=transport)
                commands.append(_format_command(server_cmd))
                if not args.dry_run:
                    processes.append(
                        _start_process(
                            server_cmd,
                            log_path=logs_dir / f"{transport_lower}_server_shard{server.shard_id}.log",
                            cwd=server.repo_root,
                        )
                    )
            if processes:
                time.sleep(3.0)
            for repeat_index in range(cfg.repeat):
                client_entries: list[dict[str, Any]] = []
                for client in cfg.clients:
                    run_id = f"{transport_lower}_b{cfg.batch_size}_d{cfg.embedding_dim}_r{repeat_index}_n{client.node_rank}"
                    client_cmd = build_client_command(
                        cfg=cfg,
                        transport=transport,
                        client=client,
                        run_id=run_id,
                    )
                    log_path = logs_dir / f"{run_id}.log"
                    main_csv = cfg.output_dir / "outputs" / run_id / "recstore_main.csv"
                    row = {
                        "run_id": run_id,
                        "lane": transport,
                        "backend": "recstore",
                        "transport": transport,
                        "torchrec_memory_mode": "",
                        "client_ip": client.ip,
                        "server_count": len(cfg.servers),
                        "batch_size": cfg.batch_size,
                        "embedding_dim": cfg.embedding_dim,
                        "num_embeddings": cfg.num_embeddings,
                        "repeat_index": repeat_index,
                        "main_csv": str(main_csv),
                        "log_path": str(log_path),
                    }
                    client_entries.append({"cmd": client_cmd, "cwd": client.repo_root, "row": row})
                _run_client_group(
                    client_entries,
                    dry_run=args.dry_run,
                    commands=commands,
                    manifest=manifest,
                )
        finally:
            _stop_processes(processes)

    for memory_mode in cfg.torchrec_baselines:
        lane = _torchrec_label(memory_mode)
        mode_slug = memory_mode.replace("_", "")
        for repeat_index in range(cfg.repeat):
            client_entries = []
            for client in cfg.clients:
                run_id = f"torchrec_{mode_slug}_b{cfg.batch_size}_d{cfg.embedding_dim}_r{repeat_index}_n{client.node_rank}"
                client_cmd = build_torchrec_command(
                    cfg=cfg,
                    memory_mode=memory_mode,
                    client=client,
                    run_id=run_id,
                )
                log_path = logs_dir / f"{run_id}.log"
                main_csv = cfg.output_dir / "outputs" / run_id / "torchrec_main.csv"
                row = {
                    "run_id": run_id,
                    "lane": lane,
                    "backend": "torchrec",
                    "transport": "",
                    "torchrec_memory_mode": memory_mode,
                    "client_ip": client.ip,
                    "server_count": 0,
                    "batch_size": cfg.batch_size,
                    "embedding_dim": cfg.embedding_dim,
                    "num_embeddings": cfg.num_embeddings,
                    "repeat_index": repeat_index,
                    "main_csv": str(main_csv),
                    "log_path": str(log_path),
                }
                client_entries.append({"cmd": client_cmd, "cwd": client.repo_root, "row": row})
            _run_client_group(
                client_entries,
                dry_run=args.dry_run,
                commands=commands,
                manifest=manifest,
            )

    (cfg.output_dir / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    os.chmod(cfg.output_dir / "commands.sh", 0o755)
    _write_csv(cfg.output_dir / "manifest.csv", manifest)
    summary_rows = collect_summary_rows(manifest)
    _write_csv(cfg.output_dir / "summary_e2e.csv", summary_rows)
    (cfg.output_dir / "summary.md").write_text(render_summary_md(cfg, summary_rows), encoding="utf-8")
    print(f"[benchmark-e2e] output={cfg.output_dir}")
    print(f"[benchmark-e2e] deployment={cfg.output_dir / 'deployment.md'}")
    print(f"[benchmark-e2e] summary={cfg.output_dir / 'summary.md'}")
    failed = [row for row in manifest if str(row.get("status", "")) == "failed"]
    if failed:
        print(f"[benchmark-e2e] failed_runs={len(failed)} manifest={cfg.output_dir / 'manifest.csv'}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
