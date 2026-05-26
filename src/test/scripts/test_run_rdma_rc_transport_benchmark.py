import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rdma_rc_transport_benchmark import build_benchmark_cmd  # noqa: E402


class TestRunRDMARCTransportBenchmark(unittest.TestCase):
    def test_build_benchmark_cmd_uses_normalized_argument_names(self):
        args = SimpleNamespace(
            benchmark_binary="./build/bin/rdma_rc_transport_benchmark",
            server_count=1,
            iterations=20,
            rounds=5,
            warmup_rounds=1,
            batch_keys=16,
            op="all",
            get_ratio=95,
            async_depth=2,
            report_mode="summary",
            qps_per_client_per_shard=32,
            rdma_wait_timeout_ms=5000,
            profile_interval_ms=250,
            verify_values=False,
            verify_value_row_stride=1,
        )

        cmd = build_benchmark_cmd(args)

        self.assertIn("--rdma_rc_qps_per_client_per_shard=32", cmd)
        self.assertIn("--rdma_rc_profile_interval_ms=250", cmd)

    def test_help_exposes_normalized_rdma_rc_arguments(self):
        script = Path(__file__).resolve().parent / "run_rdma_rc_transport_benchmark.py"
        completed = subprocess.run(
            ["python3", str(script), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("--qps-per-client-per-shard", completed.stdout)
        self.assertIn("--profile-interval-ms", completed.stdout)
        self.assertIn("--server-coroutines-per-thread", completed.stdout)
        self.assertIn("--fake-get-mode", completed.stdout)
        self.assertIn("--skip-client-copy", completed.stdout)


if __name__ == "__main__":
    unittest.main()
