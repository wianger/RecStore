import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PYTORCH_ROOT = Path(__file__).resolve().parents[2]
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))


class TestEBCBaselineHarnessConfig(unittest.TestCase):
    def test_resolve_ps_endpoint_prefers_client_config(self):
        from recstore.unittest.ebc_baseline.config import resolve_ps_endpoint

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "client": {"host": "10.1.2.3", "port": 17000},
                    "cache_ps": {
                        "servers": [
                            {"host": "127.0.0.1", "port": 15000, "shard": 0}
                        ]
                    },
                },
                f,
            )
            config_path = f.name

        try:
            self.assertEqual(resolve_ps_endpoint(config_path), ("10.1.2.3", 17000))
        finally:
            os.unlink(config_path)

    def test_resolve_ps_endpoint_falls_back_to_first_cache_server(self):
        from recstore.unittest.ebc_baseline.config import resolve_ps_endpoint

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "cache_ps": {
                        "servers": [
                            {"host": "127.0.0.2", "port": 15123, "shard": 0}
                        ]
                    }
                },
                f,
            )
            config_path = f.name

        try:
            self.assertEqual(resolve_ps_endpoint(config_path), ("127.0.0.2", 15123))
        finally:
            os.unlink(config_path)


class TestEBCBaselineHarnessCases(unittest.TestCase):
    def test_default_cases_are_cpu_safe_and_named(self):
        from recstore.unittest.ebc_baseline.cases import default_precision_cases

        cases = default_precision_cases()

        self.assertGreaterEqual(len(cases), 2)
        self.assertTrue(all(case.cpu for case in cases))
        self.assertIn("basic_cpu", [case.name for case in cases])
        self.assertIn("small_batch_cpu", [case.name for case in cases])

    def test_case_converts_to_argparse_namespace(self):
        from recstore.unittest.ebc_baseline.cases import PrecisionCase

        namespace = PrecisionCase(
            name="probe",
            num_embeddings=17,
            embedding_dim=8,
            batch_size=3,
            seed=9,
            cpu=True,
        ).to_namespace()

        self.assertIsInstance(namespace, argparse.Namespace)
        self.assertEqual(namespace.num_embeddings, 17)
        self.assertEqual(namespace.embedding_dim, 8)
        self.assertEqual(namespace.batch_size, 3)
        self.assertEqual(namespace.seed, 9)
        self.assertTrue(namespace.cpu)


class TestEBCBaselineHarnessWrapper(unittest.TestCase):
    def test_multiprocess_command_includes_endpoint_and_case_options(self):
        from recstore.unittest.ebc_baseline.cases import MultiProcessCase
        from recstore.unittest.ebc_baseline.wrapper import build_multiprocess_command

        case = MultiProcessCase(
            name="mp",
            num_embeddings=100,
            embedding_dim=16,
            batch_size=4,
            seed=11,
            cpu=True,
            world_size=3,
        )

        cmd = build_multiprocess_command(
            case=case,
            script_path=Path("/tmp/test_ebc_precision_multiprocess.py"),
            ps_endpoint=("127.0.0.1", 15123),
            python_executable="/usr/bin/python3",
        )

        self.assertEqual(cmd[0], "/usr/bin/python3")
        self.assertIn("--ps-host", cmd)
        self.assertIn("127.0.0.1", cmd)
        self.assertIn("--ps-port", cmd)
        self.assertIn("15123", cmd)
        self.assertIn("--world-size", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--cpu", cmd)

    def test_module_setup_reuses_existing_server_when_helper_says_skip(self):
        from recstore.unittest.ebc_baseline.wrapper import EBCPrecisionModuleHarness

        harness = EBCPrecisionModuleHarness(
            should_skip_server_start=lambda: (True, "already_running"),
            check_ps_server_running=lambda: (True, [15123]),
        )

        harness.setup()
        try:
            self.assertIsNone(harness.server_runner)
        finally:
            harness.teardown()

    def test_module_setup_starts_server_with_helper_config(self):
        from recstore.unittest.ebc_baseline.wrapper import EBCPrecisionModuleHarness

        runner = mock.Mock()
        runner.start.return_value = True
        runner.is_running.return_value = True
        runner.stop.return_value = True
        runner.config_path = "/tmp/recstore_config.json"
        runner_factory = mock.Mock(return_value=runner)

        harness = EBCPrecisionModuleHarness(
            should_skip_server_start=lambda: (False, None),
            get_server_config=lambda: {
                "server_path": "/tmp/ps_server",
                "config_path": "/tmp/recstore_config.json",
                "log_dir": "/tmp/logs",
                "timeout": 7,
                "num_shards": 2,
            },
            server_runner_factory=runner_factory,
            ready_delay_seconds=0,
        )

        harness.setup()
        try:
            runner_factory.assert_called_once_with(
                server_path="/tmp/ps_server",
                config_path="/tmp/recstore_config.json",
                log_dir="/tmp/logs",
                timeout=7,
                num_shards=2,
                verbose=True,
            )
            runner.start.assert_called_once_with()
        finally:
            harness.teardown()
        runner.stop.assert_called_once_with()

