#!/usr/bin/env python3

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SUMMARY_RE = re.compile(
    r"system=(?P<system>\S+) "
    r"transport=(?P<transport>\S+) "
    r"phase=(?P<phase>\S+) "
    r"summary "
    r"rounds=(?P<rounds>\d+) "
    r"iterations=(?P<iterations>\d+) "
    r"batch_keys=(?P<batch_keys>\d+) "
    r"num_embeddings=(?P<num_embeddings>\d+) "
    r"elapsed_us_mean=(?P<mean>[0-9.eE+-]+) "
    r"elapsed_us_p50=(?P<p50>[0-9.eE+-]+) "
    r"elapsed_us_p95=(?P<p95>[0-9.eE+-]+) "
    r"elapsed_us_p99=(?P<p99>[0-9.eE+-]+) "
    r"ops_per_sec=(?P<ops>[0-9.eE+-]+) "
    r"key_ops_per_sec=(?P<key_ops>[0-9.eE+-]+)"
)

def collect_summary_rows(text: str) -> list[dict[str, str | int | float]]:
    rows = []
    for line in text.splitlines():
        m = SUMMARY_RE.search(line)
        if m is None:
            continue
        rows.append(
            {
                "system": m.group("system"),
                "transport": m.group("transport"),
                "phase": m.group("phase"),
                "rounds": int(m.group("rounds")),
                "iterations": int(m.group("iterations")),
                "batch_keys": int(m.group("batch_keys")),
                "num_embeddings": int(m.group("num_embeddings")),
                "mean": float(m.group("mean")),
                "p50": float(m.group("p50")),
                "p95": float(m.group("p95")),
                "p99": float(m.group("p99")),
                "ops": float(m.group("ops")),
                "key_ops": float(m.group("key_ops")),
            }
        )
    return rows


def print_summary_table(rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        print("[summary] no parsed measure summary rows found")
        return
    header = [
        "system",
        "transport",
        "phase",
        "rounds",
        "iterations",
        "batch_keys",
        "num_embeddings",
        "mean_us",
        "p50_us",
        "p95_us",
        "p99_us",
        "ops/s",
        "key_ops/s",
    ]
    table = [header]
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row["system"]),
            str(row["phase"]),
            str(row["transport"]),
            int(row["num_embeddings"]),
        ),
    )
    for row in sorted_rows:
        table.append(
            [
                str(row["system"]),
                str(row["transport"]),
                str(row["phase"]),
                str(row["rounds"]),
                str(row["iterations"]),
                str(row["batch_keys"]),
                str(row["num_embeddings"]),
                f"{row['mean']:,.2f}",
                f"{row['p50']:,.2f}",
                f"{row['p95']:,.2f}",
                f"{row['p99']:,.2f}",
                f"{row['ops']:,.2f}",
                f"{row['key_ops']:,.2f}",
            ]
        )

    widths = [max(len(r[i]) for r in table) for i in range(len(header))]

    def render(r: list[str]) -> str:
        return "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(r))) + " |"

    sep = "|-" + "-|-".join("-" * widths[i] for i in range(len(widths))) + "-|"
    print("\n=== Mixed Benchmark Summary ===")
    print(render(table[0]))
    print(sep)
    for row in table[1:]:
        print(render(row))


def build_recstore_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        str(args.recstore_binary),
        f"--transport={args.transport}",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--iterations={args.iterations}",
        f"--rounds={args.rounds}",
        f"--warmup_rounds={args.warmup_rounds}",
        f"--batch_keys={args.batch_keys}",
        f"--embedding_dim={args.embedding_dim}",
        f"--num_embeddings={args.num_embeddings}",
        f"--init_chunk_size={args.init_chunk_size}",
        f"--report_mode={args.report_mode}",
        f"--update_scale={args.update_scale}",
    ]
    if args.transport.lower() == "brpc":
        cmd.append(f"--brpc_timeout_ms={args.brpc_timeout_ms}")
    return cmd


def build_hierkv_cmd(args: argparse.Namespace) -> list[str]:
    init_capacity = (
        args.init_capacity if args.init_capacity > 0 else args.num_embeddings
    )
    max_capacity = (
        args.max_capacity if args.max_capacity > 0 else args.num_embeddings
    )
    return [
        str(args.hierkv_binary),
        f"--iterations={args.iterations}",
        f"--rounds={args.rounds}",
        f"--warmup_rounds={args.warmup_rounds}",
        f"--batch_keys={args.batch_keys}",
        f"--embedding_dim={args.embedding_dim}",
        f"--num_embeddings={args.num_embeddings}",
        f"--init_chunk_size={args.init_chunk_size}",
        f"--report_mode={args.report_mode}",
        f"--update_scale={args.update_scale}",
        f"--init_capacity={init_capacity}",
        f"--max_capacity={max_capacity}",
        f"--max_hbm_for_vectors={args.max_hbm_for_vectors}",
    ]


def build_recstore_server_cmd(args: argparse.Namespace, repo_root: Path) -> list[str]:
    server_binary = Path(args.recstore_server_binary)
    config_path = (repo_root / args.recstore_config).resolve()
    if args.transport.lower() == "brpc":
        return [str(server_binary), "--brpc_config_path", str(config_path)]
    return [str(server_binary), "--config_path", str(config_path)]


def run_cmd(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def wait_for_recstore_server_ready(
    transport: str, process: subprocess.Popen[str], server_log_path: Path
) -> None:
    ready_pattern = (
        "bRPC Server shard 0 listening on"
        if transport.lower() == "brpc"
        else "listening on"
    )
    for _ in range(80):
        time.sleep(0.5)
        text = server_log_path.read_text(encoding="utf-8", errors="ignore")
        if ready_pattern in text:
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"recstore server exited early with code {process.returncode}; "
                f"log: {server_log_path}"
            )
    raise RuntimeError(f"timed out waiting for recstore server; log: {server_log_path}")


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "transport",
        "phase",
        "rounds",
        "iterations",
        "batch_keys",
        "num_embeddings",
        "mean",
        "p50",
        "p95",
        "p99",
        "ops",
        "key_ops",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/app/RecStore")
    parser.add_argument(
        "--recstore-binary",
        default="/app/RecStore/build/bin/recstore_mixed_benchmark",
    )
    parser.add_argument(
        "--hierkv-binary",
        default="/app/RecStore/third_party/HierarchicalKV/build/mixed_benchmark",
    )
    parser.add_argument(
        "--recstore-server-binary",
        default="/app/RecStore/build/bin/brpc_ps_server",
    )
    parser.add_argument("--transport", default="brpc")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15000)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--batch-keys", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-embeddings", type=int, default=1024)
    parser.add_argument(
        "--sweep-num-embeddings", type=int, nargs="*", default=[]
    )
    parser.add_argument("--init-chunk-size", type=int, default=4096)
    parser.add_argument("--report-mode", choices=["summary", "per_round", "both"], default="summary")
    parser.add_argument("--update-scale", type=float, default=0.001)
    parser.add_argument("--brpc-timeout-ms", type=int, default=5000)
    parser.add_argument("--init-capacity", type=int, default=0)
    parser.add_argument("--max-capacity", type=int, default=0)
    parser.add_argument("--max-hbm-for-vectors", type=int, default=1073741824)
    parser.add_argument("--recstore-config", default="recstore_config.json")
    parser.add_argument("--server-log-dir", default="/tmp/recstore_mixed_benchmark_logs")
    parser.add_argument("--skip-recstore-server", action="store_true", default=False)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    all_rows: list[dict[str, str | int | float]] = []
    sweep_values = args.sweep_num_embeddings or [args.num_embeddings]

    for num_embeddings in sweep_values:
        args.num_embeddings = num_embeddings
        if args.skip_recstore_server:
            recstore_output = run_cmd(build_recstore_cmd(args), repo_root)
        else:
            log_dir = Path(args.server_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            server_log_path = log_dir / f"recstore_server_{num_embeddings}.log"
            with server_log_path.open("w", encoding="utf-8") as server_log:
                server = subprocess.Popen(
                    build_recstore_server_cmd(args, repo_root),
                    cwd=str(repo_root),
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                wait_for_recstore_server_ready(
                    args.transport, server, server_log_path
                )
                recstore_output = run_cmd(build_recstore_cmd(args), repo_root)
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                if server.returncode not in (0, -15):
                    raise RuntimeError(
                        f"recstore server exited with code {server.returncode}; "
                        f"log: {server_log_path}"
                    )
        print(recstore_output, end="" if recstore_output.endswith("\n") else "\n")
        all_rows.extend(collect_summary_rows(recstore_output))

        hierkv_output = run_cmd(build_hierkv_cmd(args), repo_root)
        print(hierkv_output, end="" if hierkv_output.endswith("\n") else "\n")
        all_rows.extend(collect_summary_rows(hierkv_output))

    print_summary_table(all_rows)
    if args.output_csv:
        csv_path = Path(args.output_csv)
        write_csv(csv_path, all_rows)
        print(f"[mixed-benchmark] csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
