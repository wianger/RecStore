import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools.config.recstore_config_path as recstore_config_path
from ps_test_config import (
    DEFAULT_BRPC_BENCHMARK_CONFIG,
    DEFAULT_GRPC_MAIN_CONFIG,
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    load_client_endpoint,
    resolve_rdma_integration_config,
    resolve_repo_path,
)


class TestPSTestConfig(unittest.TestCase):
    def test_rdma_config_roles_are_explicit(self):
        self.assertEqual(
            DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
            "./src/test/configs/recstore_config.rdma_test.json",
        )
        self.assertEqual(
            DEFAULT_RDMA_MULTI_SHARD_CONFIG,
            "./src/test/configs/recstore_config.rdma_multishard_test.json",
        )

    def test_default_non_rdma_config_roles_are_explicit(self):
        self.assertEqual(DEFAULT_GRPC_MAIN_CONFIG, "./recstore_config.json")
        self.assertEqual(
            DEFAULT_BRPC_BENCHMARK_CONFIG,
            "./src/test/configs/recstore_config.brpc.json",
        )

    def test_default_grpc_config_resolves_from_search_path_before_default(self):
        with tempfile.TemporaryDirectory() as default_tmp, tempfile.TemporaryDirectory() as cwd_tmp:
            default_config = Path(default_tmp) / "recstore_config.json"
            search_config = Path(cwd_tmp) / "repo" / "recstore_config.json"
            nested = search_config.parent / "a" / "b"
            nested.mkdir(parents=True)
            default_config.write_text("{}", encoding="utf-8")
            search_config.write_text("{}", encoding="utf-8")
            with mock.patch.object(
                recstore_config_path,
                "DEFAULT_RECSTORE_CONFIG_PATH",
                default_config,
            ), mock.patch("pathlib.Path.cwd", return_value=nested):
                self.assertEqual(
                    resolve_repo_path(DEFAULT_GRPC_MAIN_CONFIG),
                    search_config.resolve(),
                )

    def test_resolve_rdma_integration_config_prefers_explicit_path(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=2, config_path="./custom.json"),
            "./custom.json",
        )

    def test_resolve_rdma_integration_config_uses_single_shard_default(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=1, config_path=None),
            DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
        )

    def test_resolve_rdma_integration_config_uses_multi_shard_default(self):
        self.assertEqual(
            resolve_rdma_integration_config(server_count=2, config_path=None),
            DEFAULT_RDMA_MULTI_SHARD_CONFIG,
        )

    def test_load_client_endpoint_for_default_grpc_config(self):
        host, port = load_client_endpoint(DEFAULT_GRPC_MAIN_CONFIG)
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 15000)

    def test_load_client_endpoint_for_brpc_benchmark_config(self):
        host, port = load_client_endpoint(DEFAULT_BRPC_BENCHMARK_CONFIG)
        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 25000)


if __name__ == "__main__":
    unittest.main()
