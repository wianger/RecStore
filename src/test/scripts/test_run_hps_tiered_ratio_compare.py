import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.benchmarks import run_hps_tiered_ratio_compare as ratio_compare  # noqa: E402
from tools.benchmarks import run_hps_backend_compare as hps_compare  # noqa: E402


class TestRunHpsTieredRatioCompare(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            mode="fetch",
            read_ratio=100,
            record_count=10_000_000,
            runtime_seconds=5,
            threads=16,
            load_threads=0,
            hps_rocksdb_load_threads=1,
            hps_rocksdb_db_threads=1,
            hps_native_cache_missed_embeddings=False,
            hps_native_overflow_policy="evict_random",
            hps_native_overflow_resolution_target=0.8,
            batch_size=1024,
            value_size=128,
            distribution="uniform",
            zipfian_alpha=0.9,
            dram_allocator="PERSIST_LOOP_SLAB",
            ssd_io_backend="IOURING",
            ssd_queue_depth=512,
            ssd_capacity_multiplier=2.0,
            tiered_high_watermark_ratio=0.85,
            include_recstore_tiered=True,
            output_dir=Path("/tmp/out"),
            keep_data=False,
            extra_arg=[],
        )

    def test_fasterkv_hlog_fraction_uses_default_full_memory_log(self):
        args = self._args()

        self.assertEqual(
            ratio_compare.configured_fasterkv_hlog_memory_bytes(args, 1.0),
            8 * 1024 * 1024 * 1024,
        )
        self.assertEqual(
            ratio_compare.configured_fasterkv_hlog_memory_bytes(args, 0.1),
            512 * 1024 * 1024,
        )
        self.assertEqual(
            ratio_compare.configured_fasterkv_hlog_memory_bytes(args, 0.0),
            256 * 1024 * 1024,
        )

    def test_run_fasterkv_baseline_annotates_memory_backend_without_ratio(self):
        args = self._args()
        with mock.patch.object(hps_compare, "run_one") as run_one:
            run_one.return_value = [{"phase": "run", "exit_code": 0}]

            rows = ratio_compare.run_fasterkv_baseline(args, repeat=2)

        run_one.assert_called_once()
        alias, repeat, run_args, spec = run_one.call_args.args
        self.assertEqual(alias, "fasterkv")
        self.assertEqual(repeat, 2)
        self.assertIs(spec, hps_compare.BACKEND_ALIASES["fasterkv"])
        self.assertNotIn("--fasterkv_storage=memory", run_args.extra_arg)
        self.assertEqual(rows[0]["comparison_group"], "fasterkv")
        self.assertEqual(rows[0]["target_dram_fraction"], "")
        self.assertEqual(rows[0]["configured_fasterkv_hlog_memory_bytes"], "")

    def test_run_fasterkv_ssd_sweep_maps_dram_fraction_to_hlog_window(self):
        args = self._args()
        with mock.patch.object(hps_compare, "run_one") as run_one:
            run_one.return_value = [{"phase": "run", "exit_code": 0}]

            rows = ratio_compare.run_fasterkv_ssd(args, repeat=1, dram_fraction=0.25)

        run_one.assert_called_once()
        alias, repeat, run_args, spec = run_one.call_args.args
        self.assertEqual(alias, "fasterkv_ssd_hlog0.250000")
        self.assertEqual(repeat, 1)
        self.assertIs(spec, hps_compare.BACKEND_ALIASES["fasterkv_ssd"])
        self.assertIn("--fasterkv_storage=ssd", run_args.extra_arg)
        self.assertIn("--fasterkv_hlog_memory_bytes=2147483648", run_args.extra_arg)
        self.assertEqual(rows[0]["comparison_group"], "fasterkv_ssd")
        self.assertEqual(rows[0]["target_dram_fraction"], "0.250000")
        self.assertEqual(rows[0]["target_ssd_fraction"], "0.750000")
        self.assertEqual(rows[0]["target_memory_window_fraction"], "0.250000")
        self.assertEqual(rows[0]["configured_fasterkv_hlog_memory_bytes"], "2147483648")

    def test_main_can_skip_recstore_tiered_for_backend_smoke(self):
        args = self._args()
        args.include_hps = False
        args.include_hps_native_tiered = False
        args.include_fasterkv = False
        args.include_recstore_tiered = False
        args.build = False
        args.build_jobs = 0
        args.repeat = 1
        args.dram_fractions = [1.0]

        with (
            mock.patch.object(ratio_compare, "parse_args", return_value=args),
            mock.patch.object(ratio_compare.hps_compare, "BENCHMARK_BIN", Path("/tmp")),
            mock.patch.object(ratio_compare, "run_recstore_tiered") as run_recstore,
            mock.patch.object(ratio_compare, "write_summary"),
            mock.patch.object(ratio_compare.shutil, "rmtree"),
        ):
            exit_code = ratio_compare.main()

        self.assertEqual(exit_code, 0)
        run_recstore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
