import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_rdma_rc_transport_benchmark import (  # noqa: E402
    _stream_process_output,
    build_benchmark_cmd,
    collect_summary_rows,
)


class TestRunRDMARCTransportBenchmark(unittest.TestCase):
    def test_quiet_mode_filters_progress_lines(self):
        from run_rdma_rc_transport_benchmark import print_filtered_output  # noqa: E402

        text = (
            "transport=RDMA_RC op=async_stream progress stage=complete phase=measure round=1/1 completed=1/1 submitted=1 elapsed_ms=1.0\n"
            "transport=RDMA_RC op=async_stream128 phase=measure summary rounds=30 iterations=2560 batch_keys=500 elapsed_us_mean=1 elapsed_us_p50=1 elapsed_us_p95=1 elapsed_us_p99=1 ops_per_sec=1 key_ops_per_sec=1\n"
        )
        from io import StringIO
        from contextlib import redirect_stdout

        out = StringIO()
        with redirect_stdout(out):
            print_filtered_output(text, show_runner_logs=False, quiet=True)
        rendered = out.getvalue()
        self.assertNotIn("progress", rendered)
        self.assertIn("summary", rendered)
        self.assertNotIn("wrote", rendered)

    def test_stream_process_output_preserves_summary_line_boundaries(self):
        from io import StringIO

        text = (
            "transport=RDMA_RC op=put phase=measure summary rounds=1 iterations=2 "
            "batch_keys=4 elapsed_us_mean=10 elapsed_us_p50=10 elapsed_us_p95=10 "
            "elapsed_us_p99=10 ops_per_sec=200 key_ops_per_sec=800\n"
            "transport=RDMA_RC op=get phase=measure summary rounds=1 iterations=2 "
            "batch_keys=4 elapsed_us_mean=20 elapsed_us_p50=20 elapsed_us_p95=20 "
            "elapsed_us_p99=20 ops_per_sec=100 key_ops_per_sec=400\n"
        )
        sink = []

        _stream_process_output(
            client_index=0,
            stream=StringIO(text),
            sink=sink,
            show_runner_logs=False,
            is_stderr=False,
        )

        rows = collect_summary_rows("".join(sink))
        self.assertEqual([row["op"] for row in rows], ["put", "get"])

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
