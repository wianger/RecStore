#!/usr/bin/env python3

import argparse
import re
import subprocess

from petps_cluster_runner import PetPSClusterRunner
from ps_test_config import (
    DEFAULT_BRPC_BENCHMARK_CONFIG,
    DEFAULT_GRPC_MAIN_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    load_client_endpoint,
)
from ps_server_runner import PSServerRunner


CONTROL_PLANE_NOISE_PATTERNS = (
    "[petps-control-plane]",
    "component=rdma_control_plane",
)

BENCHMARK_NOISE_PATTERNS = (
    "I open mlx5_0 :)",
)

SUMMARY_RE = re.compile(
    r"transport=(?P<transport>\S+) "
    r"op=(?P<op>\S+) "
    r"phase=(?P<phase>\S+) "
    r"summary "
    r"rounds=(?P<rounds>\d+) "
    r"iterations=(?P<iterations>\d+) "
    r"batch_keys=(?P<batch_keys>\d+) "
    r"elapsed_us_mean=(?P<mean>[0-9.eE+-]+) "
    r"elapsed_us_p50=(?P<p50>[0-9.eE+-]+) "
    r"elapsed_us_p95=(?P<p95>[0-9.eE+-]+) "
    r"elapsed_us_p99=(?P<p99>[0-9.eE+-]+) "
    r"ops_per_sec=(?P<ops>[0-9.eE+-]+) "
    r"key_ops_per_sec=(?P<key_ops>[0-9.eE+-]+)"
)


def build_rdma_runner(args):
    # Keep RDMA transport request-size limit aligned with benchmark batch size.
    # Otherwise batch_keys > max_kv_num_per_request can crash server side and
    # make the client appear stuck waiting for completion.
    max_kv_num_per_request = max(1, int(args.batch_keys))
    return PetPSClusterRunner(
        config_path=DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
        num_servers=1,
        num_clients=1,
        thread_num=getattr(args, "rdma_thread_num", 1),
        max_kv_num_per_request=max_kv_num_per_request,
        verbose=args.show_runner_logs,
        show_status_logs=args.show_runner_logs,
        show_control_plane_logs=args.show_runner_logs,
        rdma_namespace=getattr(args, "rdma_namespace", "auto"),
        rdma_control_plane_host=getattr(
            args, "rdma_control_plane_host", "127.0.0.1"
        ),
        rdma_control_plane_port=getattr(args, "rdma_control_plane_port", None),
        rdma_per_thread_response_limit_bytes=getattr(
            args, "rdma_per_thread_response_limit_bytes", None
        ),
        rdma_client_receive_arena_bytes=getattr(
            args, "rdma_client_receive_arena_bytes", None
        ),
        rdma_put_protocol_version=getattr(args, "rdma_put_protocol_version", 2),
        rdma_put_v2_transfer_mode=getattr(
            args, "rdma_put_v2_transfer_mode", "push"
        ),
        rdma_put_v2_push_slot_bytes=getattr(
            args, "rdma_put_v2_push_slot_bytes", None
        ),
        rdma_put_v2_push_slots_per_client=getattr(
            args, "rdma_put_v2_push_slots_per_client", None
        ),
        rdma_put_v2_push_region_offset=getattr(
            args, "rdma_put_v2_push_region_offset", None
        ),
        rdma_put_client_send_arena_bytes=getattr(
            args, "rdma_put_client_send_arena_bytes", None
        ),
        rdma_put_server_scratch_bytes=getattr(
            args, "rdma_put_server_scratch_bytes", None
        ),
        rdma_wait_timeout_ms=getattr(args, "rdma_wait_timeout_ms", None),
        validate_routing=getattr(args, "validate_routing", False),
    )

def is_runner_noise_line(line):
    return any(pattern in line for pattern in CONTROL_PLANE_NOISE_PATTERNS)


def is_benchmark_noise_line(line):
    return any(pattern in line for pattern in BENCHMARK_NOISE_PATTERNS)


def print_filtered_output(text, show_runner_logs):
    for line in text.splitlines():
        if not show_runner_logs:
            if is_runner_noise_line(line) or is_benchmark_noise_line(line):
                continue
        print(line)


def build_benchmark_cmd(
    benchmark_binary,
    transport,
    iterations,
    rounds,
    warmup_rounds,
    report_mode,
    batch_keys,
    host=None,
    port=None,
    num_shards=None,
):
    cmd = [
        benchmark_binary,
        f"--transport={transport}",
        f"--iterations={iterations}",
        f"--rounds={rounds}",
        f"--warmup_rounds={warmup_rounds}",
        f"--report_mode={report_mode}",
        f"--batch_keys={batch_keys}",
    ]
    if host is not None:
        cmd.append(f"--host={host}")
    if port is not None:
        cmd.append(f"--port={port}")
    if num_shards is not None:
        cmd.append(f"--num_shards={num_shards}")
    return cmd


def collect_summary_rows(text):
    rows = []
    for line in text.splitlines():
        m = SUMMARY_RE.search(line)
        if m is None:
            continue
        if m.group("phase") != "measure":
            continue
        rows.append(
            {
                "transport": m.group("transport"),
                "op": m.group("op"),
                "rounds": int(m.group("rounds")),
                "iterations": int(m.group("iterations")),
                "batch_keys": int(m.group("batch_keys")),
                "mean": float(m.group("mean")),
                "p50": float(m.group("p50")),
                "p95": float(m.group("p95")),
                "p99": float(m.group("p99")),
                "ops": float(m.group("ops")),
                "key_ops": float(m.group("key_ops")),
            }
        )
    return rows


def _fmt_num(value):
    return f"{value:,.2f}"


def _per_request_us(row, field):
    if row["iterations"] <= 0:
        return 0.0
    return row[field] / row["iterations"]


def print_summary_table(rows):
    if not rows:
        print("[summary] no parsed measure summary rows found")
        return

    rows = sorted(rows, key=lambda r: r["transport"])
    header = [
        "transport",
        "mode",
        "put_v2",
        "op",
        "rounds",
        "iterations",
        "batch_keys",
        "mean_req_us",
        "p50_req_us",
        "p95_req_us",
        "key_ops/s",
    ]
    table = [header]
    for row in rows:
        table.append(
            [
                row["transport"],
                row.get("transport_mode", ""),
                row.get("put_v2_transfer_mode", ""),
                row["op"],
                str(row["rounds"]),
                str(row["iterations"]),
                str(row["batch_keys"]),
                _fmt_num(_per_request_us(row, "mean")),
                _fmt_num(_per_request_us(row, "p50")),
                _fmt_num(_per_request_us(row, "p95")),
                _fmt_num(row["key_ops"]),
            ]
        )

    widths = [max(len(r[i]) for r in table) for i in range(len(header))]

    def render(r):
        return "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(r))) + " |"

    sep = "|-" + "-|-".join("-" * widths[i] for i in range(len(widths))) + "-|"
    print("\n=== Benchmark Summary (measure phase) ===")
    print(render(table[0]))
    print(sep)
    for r in table[1:]:
        print(render(r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-binary", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--rdma-thread-num",
        type=int,
        default=1,
        help="number of RDMA server polling threads",
    )
    parser.add_argument(
        "--batch-keys",
        type=int,
        default=4,
        help="number of keys carried by each put/get RPC",
    )
    parser.add_argument("--rdma-warmup-rounds", type=int, default=2)
    parser.add_argument(
        "--report-mode",
        choices=["summary", "per_round", "both"],
        default="summary",
        help="benchmark output style: summary only, per-round only, or both",
    )
    parser.add_argument("--grpc-config", default=DEFAULT_GRPC_MAIN_CONFIG)
    parser.add_argument(
        "--brpc-config",
        default=DEFAULT_BRPC_BENCHMARK_CONFIG,
    )
    parser.add_argument("--rdma-namespace", default="auto")
    parser.add_argument("--rdma-control-plane-host", default="127.0.0.1")
    parser.add_argument("--rdma-control-plane-port", type=int)
    parser.add_argument("--rdma-per-thread-response-limit-bytes", type=int)
    parser.add_argument("--rdma-client-receive-arena-bytes", type=int)
    parser.add_argument(
        "--rdma-put-protocol-version",
        type=int,
        choices=[1, 2],
        default=2,
        help="RDMA PUT protocol version: 2(default)=remote payload, 1=legacy",
    )
    parser.add_argument(
        "--rdma-put-v2-transfer-mode",
        choices=["read", "push"],
        default="push",
        help="RDMA PUT-v2 payload mode: read(server read payload) or push(client push payload)",
    )
    parser.add_argument("--rdma-put-v2-push-slot-bytes", type=int)
    parser.add_argument("--rdma-put-v2-push-slots-per-client", type=int)
    parser.add_argument("--rdma-put-v2-push-region-offset", type=int)
    parser.add_argument("--rdma-put-client-send-arena-bytes", type=int)
    parser.add_argument("--rdma-put-server-scratch-bytes", type=int)
    parser.add_argument("--rdma-wait-timeout-ms", type=int)
    parser.add_argument(
        "--rdma-client-timeout-sec",
        type=int,
        default=120,
        help="timeout for RDMA benchmark client process (seconds), <=0 means no timeout",
    )
    parser.add_argument("--validate-routing", action="store_true")
    parser.add_argument(
        "--show-runner-logs",
        action="store_true",
        help="show control-plane/status logs from runner and benchmark binaries",
    )
    parser.add_argument(
        "--rdma-only",
        action="store_true",
        help="run RDMA benchmark only and skip GRPC/BRPC stages",
    )
    args = parser.parse_args()
    summary_rows = []

    print("[阶段] 1/3 RDMA benchmark")
    rdma_runner = build_rdma_runner(args)
    with rdma_runner.run():
        completed = rdma_runner.run_client(
            build_benchmark_cmd(
                benchmark_binary=args.benchmark_binary,
                transport="rdma",
                num_shards=1,
                iterations=args.iterations,
                rounds=args.rounds,
                warmup_rounds=args.rdma_warmup_rounds,
                report_mode=args.report_mode,
                batch_keys=args.batch_keys,
            ),
            stream_output=args.show_runner_logs,
            timeout=(args.rdma_client_timeout_sec if args.rdma_client_timeout_sec > 0 else None),
        )
        if not args.show_runner_logs:
            print_filtered_output(completed.stdout, args.show_runner_logs)
            print_filtered_output(completed.stderr, args.show_runner_logs)
        rows = collect_summary_rows(completed.stdout)
        for row in rows:
            row["put_v2_transfer_mode"] = args.rdma_put_v2_transfer_mode
        summary_rows.extend(rows)
        rc = completed.returncode
        if rc != 0:
            return rc

    if args.rdma_only:
        print("[阶段] 跳过 GRPC/BRPC（--rdma-only）")
        print_summary_table(summary_rows)
        return 0

    print("[阶段] 2/3 GRPC benchmark")
    grpc_host, grpc_port = load_client_endpoint(args.grpc_config)
    grpc_runner = PSServerRunner(config_path=args.grpc_config, num_shards=2)
    with grpc_runner.run():
        completed = subprocess.run(
            build_benchmark_cmd(
                benchmark_binary=args.benchmark_binary,
                transport="grpc",
                host=grpc_host,
                port=grpc_port,
                iterations=args.iterations,
                rounds=args.rounds,
                warmup_rounds=0,
                report_mode=args.report_mode,
                batch_keys=args.batch_keys,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        print_filtered_output(completed.stdout, args.show_runner_logs)
        print_filtered_output(completed.stderr, args.show_runner_logs)
        summary_rows.extend(collect_summary_rows(completed.stdout))
        if completed.returncode != 0:
            return completed.returncode

    print("[阶段] 3/3 BRPC benchmark")
    brpc_host, brpc_port = load_client_endpoint(args.brpc_config)
    brpc_runner = PSServerRunner(config_path=args.brpc_config, num_shards=2)
    with brpc_runner.run():
        completed = subprocess.run(
            build_benchmark_cmd(
                benchmark_binary=args.benchmark_binary,
                transport="brpc",
                host=brpc_host,
                port=brpc_port,
                iterations=args.iterations,
                rounds=args.rounds,
                warmup_rounds=0,
                report_mode=args.report_mode,
                batch_keys=args.batch_keys,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        print_filtered_output(completed.stdout, args.show_runner_logs)
        print_filtered_output(completed.stderr, args.show_runner_logs)
        summary_rows.extend(collect_summary_rows(completed.stdout))
        if completed.returncode != 0:
            return completed.returncode

    print_summary_table(summary_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
