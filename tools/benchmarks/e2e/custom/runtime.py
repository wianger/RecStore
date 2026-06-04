from __future__ import annotations

from pathlib import Path
from typing import Any

from ..commands import wrap_remote_command
from ..common import SPARSE_FEATURES_PER_SAMPLE, _dense_arch_for_embedding_dim
from .config import BenchmarkConfig, ClientSpec, ServerSpec


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


def _wrap_remote(cmd: list[str], *, ssh_host: str, cwd: Path) -> list[str]:
    if ssh_host in {"", "local", "localhost"}:
        return cmd
    return wrap_remote_command(cmd, ssh_host, cwd=cwd)


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
