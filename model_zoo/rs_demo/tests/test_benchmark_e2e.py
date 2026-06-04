from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path


class TestBenchmarkE2E(unittest.TestCase):
    def test_parse_specs_and_infer_topology(self) -> None:
        from tools.benchmarks.run_benchmark_e2e import (
            BenchmarkConfig,
            infer_client_deployment,
            infer_ps_deployment,
            parse_client_spec,
            parse_server_spec,
        )

        client = parse_client_spec(
            "ssh=root@10.0.2.191 -p 50201,repo=/app/RecStore,ip=10.0.2.191,gpu=1,node_rank=0,nproc=2"
        )
        server = parse_server_spec(
            "ssh=root@10.0.2.190 -p 50201,repo=/app/RecStore,ip=10.0.2.190,port=25000,shard=0"
        )
        cfg = BenchmarkConfig(clients=(client,), servers=(server,))

        self.assertEqual(client.gpu_id, 1)
        self.assertEqual(client.nproc_per_node, 2)
        self.assertEqual(server.port, 25000)
        self.assertEqual(infer_client_deployment(cfg.clients), "single-node")
        self.assertEqual(infer_ps_deployment(cfg.servers), "single-ps")

    def test_runtime_config_uses_requested_transport_and_shards(self) -> None:
        from tools.benchmarks.run_benchmark_e2e import (
            BenchmarkConfig,
            ClientSpec,
            ServerSpec,
            build_runtime_config,
        )

        cfg = BenchmarkConfig(
            clients=(ClientSpec(),),
            servers=(
                ServerSpec(ip="10.0.2.190", port=25000, shard_id=0),
                ServerSpec(ip="10.0.2.191", port=25001, shard_id=1),
            ),
            num_embeddings=12345,
            init_rows=12345,
            embedding_dim=16,
        )

        runtime = build_runtime_config(cfg, transport="GRPC", value_path=Path("/tmp/value"))

        self.assertEqual(runtime["cache_ps"]["ps_type"], "GRPC")
        self.assertEqual(runtime["cache_ps"]["num_shards"], 2)
        self.assertEqual(runtime["distributed_client"]["num_shards"], 2)
        self.assertEqual(runtime["distributed_client"]["servers"][1]["shard"], 1)
        self.assertEqual(runtime["cache_ps"]["base_kv_config"]["capacity"], 641940)
        self.assertEqual(
            runtime["cache_ps"]["base_kv_config"]["value"]["default_value_size_hint"],
            64,
        )

    def test_build_client_command_has_transport_and_no_rdma(self) -> None:
        from tools.benchmarks.run_benchmark_e2e import (
            BenchmarkConfig,
            ClientSpec,
            ServerSpec,
            build_client_command,
        )

        cfg = BenchmarkConfig(
            clients=(ClientSpec(repo_root=Path("/app/RecStore"), gpu_id=0),),
            servers=(ServerSpec(ip="10.0.2.190", port=26000),),
            output_dir=Path("/tmp/out"),
            runtime_dir=Path("/tmp/out/runtime/brpc"),
            steps=3,
            warmup_steps=1,
        )

        cmd = build_client_command(
            cfg=cfg,
            transport="BRPC",
            client=cfg.clients[0],
            run_id="brpc_b1024_d128_r0",
        )

        self.assertIn("--ps-type", cmd)
        self.assertIn("BRPC", cmd)
        self.assertNotIn("RDMA", cmd)
        self.assertIn("--no-start-server", cmd)
        self.assertIn("--server-host", cmd)
        self.assertIn("10.0.2.190", cmd)

    def test_build_torchrec_command_uses_same_workload(self) -> None:
        from tools.benchmarks.run_benchmark_e2e import (
            BenchmarkConfig,
            ClientSpec,
            ServerSpec,
            build_torchrec_command,
        )

        cfg = BenchmarkConfig(
            clients=(ClientSpec(repo_root=Path("/app/RecStore"), gpu_id=1),),
            servers=(ServerSpec(),),
            output_dir=Path("/tmp/out"),
            batch_size=128,
            embedding_dim=64,
            num_embeddings=10000,
            steps=7,
            warmup_steps=2,
        )

        cmd = build_torchrec_command(
            cfg=cfg,
            memory_mode="hbm",
            client=cfg.clients[0],
            run_id="torchrec_hbm_b128_d64_r0_n0",
        )

        self.assertIn("CUDA_VISIBLE_DEVICES=1", cmd)
        self.assertIn("--backend", cmd)
        self.assertIn("torchrec", cmd)
        self.assertIn("--torchrec-memory-mode", cmd)
        self.assertIn("hbm", cmd)
        self.assertIn("--batch-size", cmd)
        self.assertIn("128", cmd)
        self.assertNotIn("--ps-type", cmd)

    def test_collect_summary_and_render_chinese_report(self) -> None:
        from tools.benchmarks.run_benchmark_e2e import (
            BenchmarkConfig,
            ClientSpec,
            ServerSpec,
            collect_summary_rows,
            render_summary_md,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "outputs" / "run1" / "recstore_main.csv"
            csv_path.parent.mkdir(parents=True)
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "warmup_excluded",
                        "step_total_ms",
                        "embed_lookup_local_ms",
                        "sparse_update_ms",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "step": 0,
                        "warmup_excluded": 1,
                        "step_total_ms": 100,
                        "embed_lookup_local_ms": 20,
                        "sparse_update_ms": 30,
                    }
                )
                writer.writerow(
                    {
                        "step": 1,
                        "warmup_excluded": 0,
                        "step_total_ms": 50,
                        "embed_lookup_local_ms": 5,
                        "sparse_update_ms": 10,
                    }
                )
            manifest = [
                {
                    "run_id": "run1",
                    "lane": "BRPC",
                    "backend": "recstore",
                    "transport": "BRPC",
                    "main_csv": str(csv_path),
                    "batch_size": 256,
                    "embedding_dim": 128,
                    "num_embeddings": 10000,
                    "repeat_index": 0,
                }
            ]

            rows = collect_summary_rows(manifest)
            report = render_summary_md(
                BenchmarkConfig(
                    clients=(ClientSpec(),),
                    servers=(ServerSpec(),),
                    output_dir=root,
                    runtime_dir=root / "runtime",
                ),
                rows,
            )

        self.assertEqual(rows[0]["mean_step_total_ms"], 50.0)
        self.assertAlmostEqual(rows[0]["samples_per_sec"], 5120.0)
        self.assertIn("## Workload 说明", report)
        self.assertIn("## E2E 吞吐", report)
        self.assertIn("## E2E 延迟分解", report)

    def test_dry_run_writes_configs_commands_and_report(self) -> None:
        from tools.benchmarks.run_benchmark_e2e import main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "e2e"
            rc = main(
                [
                    "--output-dir",
                    str(out),
                    "--transports",
                    "brpc,grpc",
                    "--steps",
                    "3",
                    "--warmup-steps",
                    "1",
                    "--repeat",
                    "1",
                    "--dry-run",
                    "--skip-build",
                    "--skip-tests",
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue((out / "deployment.md").exists())
            self.assertTrue((out / "commands.sh").exists())
            self.assertTrue((out / "summary.md").exists())
            brpc_cfg = json.loads((out / "runtime" / "brpc" / "recstore_config.json").read_text())
            grpc_cfg = json.loads((out / "runtime" / "grpc" / "recstore_config.json").read_text())
            commands = (out / "commands.sh").read_text(encoding="utf-8")

        self.assertEqual(brpc_cfg["cache_ps"]["ps_type"], "BRPC")
        self.assertEqual(grpc_cfg["cache_ps"]["ps_type"], "GRPC")
        self.assertIn("--backend torchrec", commands)


if __name__ == "__main__":
    unittest.main()
