from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from model_zoo.rs_demo.config import parse_config
from model_zoo.rs_demo.config import populate_default_paths


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "model_zoo" / "rs_demo" / "run_mock_stress.py").is_file():
            return parent
    raise RuntimeError("Could not locate RecStore repository root")


class TestHybridConfig(unittest.TestCase):
    def test_parse_config_accepts_dlrm_arch_flags(self) -> None:
        cfg = parse_config(
            [
                "--embedding-dim",
                "128",
                "--dense-arch-layer-sizes",
                "512,256,128",
                "--over-arch-layer-sizes",
                "1024,1024,512,256,1",
            ]
        )

        self.assertEqual(cfg.embedding_dim, 128)
        self.assertEqual(cfg.dense_arch_layer_sizes, "512,256,128")
        self.assertEqual(cfg.over_arch_layer_sizes, "1024,1024,512,256,1")

    def test_parse_config_accepts_ps_kv_backend_flag(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "recstore",
                "--ps-kv-backend",
                "hps_rocksdb",
            ]
        )

        self.assertEqual(cfg.ps_kv_backend, "hps_rocksdb")

    def test_parse_config_accepts_tiered_dram_capacity_multiplier(self) -> None:
        cfg = parse_config(
            [
                "--backend",
                "recstore",
                "--ps-kv-backend",
                "recstore_tiered",
                "--tiered-dram-capacity-multiplier",
                "0.02",
            ]
        )

        self.assertEqual(cfg.tiered_dram_capacity_multiplier, 0.02)

    def test_run_mock_stress_script_help_works_from_repo_root(self) -> None:
        repo_root = _repo_root()
        res = subprocess.run(
            [
                sys.executable,
                str(repo_root / "model_zoo/rs_demo/run_mock_stress.py"),
                "--help",
            ],
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertIn("--ps-kv-backend", res.stdout)

    def test_populate_default_paths_makes_relative_output_root_absolute(self) -> None:
        cfg = parse_config(
            [
                "--output-root",
                "relative-output",
                "--run-id",
                "case-relative-output",
            ]
        )

        populate_default_paths(cfg)

        self.assertTrue(Path(cfg.output_root).is_absolute())
        self.assertTrue(Path(cfg.recstore_main_csv).is_absolute())
        self.assertIn("relative-output", cfg.recstore_main_csv)


if __name__ == "__main__":
    unittest.main()
