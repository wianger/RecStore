#!/usr/bin/env python3

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.config.recstore_config_path import resolve_recstore_config_path

DEFAULT_GRPC_MAIN_CONFIG = "./recstore_config.json"
DEFAULT_BRPC_BENCHMARK_CONFIG = "./src/test/configs/recstore_config.brpc.json"
DEFAULT_RDMA_SINGLE_SHARD_CONFIG = "./src/test/configs/recstore_config.rdma_test.json"
DEFAULT_RDMA_MULTI_SHARD_CONFIG = "./src/test/configs/recstore_config.rdma_multishard_test.json"


def resolve_repo_path(config_path):
    resolved = Path(config_path)
    if str(config_path) == DEFAULT_GRPC_MAIN_CONFIG:
        return resolve_recstore_config_path()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def resolve_rdma_integration_config(server_count, config_path):
    if config_path:
        return config_path
    if server_count > 1:
        return DEFAULT_RDMA_MULTI_SHARD_CONFIG
    return DEFAULT_RDMA_SINGLE_SHARD_CONFIG


def load_client_endpoint(config_path):
    with resolve_repo_path(config_path).open() as fh:
        config = json.load(fh)
    client = config["client"]
    return client["host"], client["port"]
