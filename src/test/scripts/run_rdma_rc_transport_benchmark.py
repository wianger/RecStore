#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import tempfile
import time

from petps_cluster_runner import PetPSClusterRunner, REPO_ROOT, _to_text
from ps_server_helpers import RDMA_SKIP_EXIT_CODE, get_rdma_skip_reason
from ps_test_config import (
    DEFAULT_RDMA_MULTI_SHARD_CONFIG,
    DEFAULT_RDMA_SINGLE_SHARD_CONFIG,
    resolve_rdma_integration_config,
)


MEMCACHED_NOISE_PATTERNS = (
    "[petps-memcached]",
    "[petps-status] phase=memcached",
    "[memcached-endpoint]",
    "use memcached in ",
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


def is_memcached_noise_line(line):
    return any(pattern in line for pattern in MEMCACHED_NOISE_PATTERNS)


def print_filtered_output(text, show_runner_logs):
    for line in text.splitlines():
        if not show_runner_logs and is_memcached_noise_line(line):
            continue
        print(line)


def collect_summary_rows(text):
    rows = []
    for line in text.splitlines():
        match = SUMMARY_RE.search(line)
        if match is None or match.group("phase") != "measure":
            continue
        rows.append(
            {
                "transport": match.group("transport"),
                "op": match.group("op"),
                "rounds": int(match.group("rounds")),
                "iterations": int(match.group("iterations")),
                "batch_keys": int(match.group("batch_keys")),
                "mean": float(match.group("mean")),
                "p50": float(match.group("p50")),
                "p95": float(match.group("p95")),
                "p99": float(match.group("p99")),
                "ops": float(match.group("ops")),
                "key_ops": float(match.group("key_ops")),
                "client_index": None,
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

    header = [
        "client",
        "transport",
        "op",
        "rounds",
        "iterations",
        "batch_keys",
        "mean_req_us",
        "p50_req_us",
        "p95_req_us",
        "p99_req_us",
        "key_ops/s",
    ]
    table = [header]
    for row in rows:
        table.append(
            [
                str(row.get("client_index", "")),
                row["transport"],
                row["op"],
                str(row["rounds"]),
                str(row["iterations"]),
                str(row["batch_keys"]),
                _fmt_num(_per_request_us(row, "mean")),
                _fmt_num(_per_request_us(row, "p50")),
                _fmt_num(_per_request_us(row, "p95")),
                _fmt_num(_per_request_us(row, "p99")),
                _fmt_num(row["key_ops"]),
            ]
        )

    widths = [max(len(row[idx]) for row in table) for idx in range(len(header))]

    def render(row):
        return "| " + " | ".join(
            row[idx].ljust(widths[idx]) for idx in range(len(row))
        ) + " |"

    sep = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    print("\n=== RDMA RC Benchmark Summary (measure phase) ===")
    print(render(table[0]))
    print(sep)
    for row in table[1:]:
        print(render(row))


def print_aggregate_table(rows):
    if not rows:
        return

    grouped = {}
    for row in rows:
        key = (row["transport"], row["op"], row["batch_keys"])
        grouped.setdefault(key, []).append(row)

    header = [
        "transport",
        "op",
        "clients",
        "batch_keys",
        "agg_ops/s",
        "agg_key_ops/s",
        "mean_req_us_avg",
    ]
    table = [header]
    for (transport, op, batch_keys), group in sorted(grouped.items()):
        agg_ops = sum(row["ops"] for row in group)
        agg_key_ops = sum(row["key_ops"] for row in group)
        avg_req_us = sum(_per_request_us(row, "mean") for row in group) / len(group)
        table.append(
            [
                transport,
                op,
                str(len(group)),
                str(batch_keys),
                _fmt_num(agg_ops),
                _fmt_num(agg_key_ops),
                _fmt_num(avg_req_us),
            ]
        )

    widths = [max(len(row[idx]) for row in table) for idx in range(len(header))]

    def render(row):
        return "| " + " | ".join(
            row[idx].ljust(widths[idx]) for idx in range(len(row))
        ) + " |"

    sep = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    print("\n=== RDMA RC Aggregate Summary (measure phase) ===")
    print(render(table[0]))
    print(sep)
    for row in table[1:]:
        print(render(row))


def build_benchmark_cmd(args):
    cmd = [
        args.benchmark_binary,
        f"--num_shards={args.server_count}",
        f"--iterations={args.iterations}",
        f"--rounds={args.rounds}",
        f"--warmup_rounds={args.warmup_rounds}",
        f"--batch_keys={args.batch_keys}",
        f"--op={args.op}",
        f"--get_ratio={args.get_ratio}",
        f"--async_depth={args.async_depth}",
        f"--report_mode={args.report_mode}",
    ]
    if args.rdma_rc_qps_per_client_per_shard is not None:
        cmd.append(
            "--rdma_rc_qps_per_client_per_shard="
            f"{args.rdma_rc_qps_per_client_per_shard}"
        )
    if args.rdma_wait_timeout_ms is not None:
        cmd.append(f"--rdma_wait_timeout_ms={args.rdma_wait_timeout_ms}")
    if args.rdma_rc_profile_interval_ms is not None:
        cmd.append(
            "--rdma_rc_profile_interval_ms="
            f"{args.rdma_rc_profile_interval_ms}"
        )
    if args.verify_values:
        cmd.append("--verify_values=true")
        cmd.append(
            f"--verify_value_row_stride={args.verify_value_row_stride}"
        )
    return cmd


def write_runtime_config(args, source_config_path, runtime_dir):
    with open(source_config_path) as fh:
        config = json.load(fh)

    cache_ps = config.setdefault("cache_ps", {})
    source_base_kv = cache_ps.get("base_kv_config", {})
    capacity = int(source_base_kv.get("capacity", 1000000))
    base_path = (
        f"/dev/shm/recstore_rdma_rc_benchmark_{os.getpid()}_{time.time_ns()}"
    )
    cache_ps["base_kv_config"] = {
        "capacity": capacity,
        "index": {"type": "DRAM_PET_HASH"},
        "value": {
            "type": "DRAM_VALUE_STORE",
            "path": f"{base_path}/value",
            "default_value_size_hint": args.value_size,
            "dram_allocator": {
                "type": "CONCURRENT_SLAB_MEMORY_POOL",
                "capacity_bytes": capacity * args.value_size,
            },
        },
    }

    runtime_config_path = (
        f"{runtime_dir}/recstore_config.rdma_runtime.json"
    )
    with open(runtime_config_path, "w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    return runtime_config_path


def terminate_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_benchmark_clients(runner, args):
    if args.client_count == 1:
        completed = runner.run_client(
            build_benchmark_cmd(args),
            timeout=args.client_timeout,
            stream_output=args.show_runner_logs,
        )
        if not args.show_runner_logs:
            print_filtered_output(completed.stdout, args.show_runner_logs)
            print_filtered_output(completed.stderr, args.show_runner_logs)
        rows = collect_summary_rows(completed.stdout)
        for row in rows:
            row["client_index"] = 0
        return completed.returncode, rows

    env = runner.build_env()
    processes = []
    deadline = time.monotonic() + args.client_timeout
    for client_index in range(args.client_count):
        cmd = runner.build_client_cmd(
            build_benchmark_cmd(args), client_index=client_index
        )
        processes.append(
            (
                client_index,
                subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                ),
            )
        )

    rows = []
    results = []
    timed_out = False
    for client_index, process in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _to_text(exc.stdout)
            stderr = _to_text(exc.stderr)
            results.append((client_index, 124, stdout, stderr))
            break
        results.append((client_index, process.returncode, stdout, stderr))

    if timed_out:
        for _client_index, process in processes:
            terminate_process(process)
        seen = {client_index for client_index, *_rest in results}
        for client_index, process in processes:
            if client_index not in seen:
                stdout, stderr = process.communicate()
                results.append((client_index, 124, stdout, stderr))

    return_code = 124 if timed_out else 0
    for client_index, rc, stdout, stderr in results:
        if not args.show_runner_logs:
            print_filtered_output(stdout, args.show_runner_logs)
            print_filtered_output(stderr, args.show_runner_logs)
        parsed = collect_summary_rows(stdout)
        for row in parsed:
            row["client_index"] = client_index
        rows.extend(parsed)
        if rc != 0:
            return_code = rc
            print(f"[rdma-rc-client:{client_index}] exited with code {rc}")

    if timed_out:
        print(f"[rdma-rc-client] timed out after {args.client_timeout} seconds")
    return return_code, rows


def main():
    skip_reason = get_rdma_skip_reason()
    if skip_reason:
        print(f"[petps-skip] {skip_reason}")
        return RDMA_SKIP_EXIT_CODE

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-binary", required=True)
    parser.add_argument("--server-count", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=1)
    parser.add_argument("--thread-num", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--batch-keys", type=int, default=16)
    parser.add_argument("--value-size", type=int, default=16)
    parser.add_argument(
        "--op",
        choices=[
            "all",
            "put",
            "get",
            "async_get",
            "async_stream",
            "mixed",
        ],
        default="all",
    )
    parser.add_argument("--get-ratio", type=int, default=95)
    parser.add_argument("--async-depth", type=int, default=1)
    parser.add_argument(
        "--report-mode",
        choices=["summary", "per_round", "both"],
        default="summary",
    )
    parser.add_argument("--config-path")
    parser.add_argument("--max-kv-num-per-request", type=int)
    parser.add_argument("--client-timeout", type=int, default=120)
    parser.add_argument("--cluster-timeout", type=int, default=35)
    parser.add_argument(
        "--use-local-memcached",
        choices=["always", "auto", "never"],
        default="auto",
    )
    parser.add_argument("--memcached-host", default="127.0.0.1")
    parser.add_argument("--memcached-port", type=int, default=21211)
    parser.add_argument("--rdma-rc-qps-per-client-per-shard", type=int)
    parser.add_argument("--rdma-wait-timeout-ms", type=int)
    parser.add_argument("--rdma-rc-profile-interval-ms", type=int)
    parser.add_argument("--rdma-rc-server-coroutines-per-thread", type=int)
    parser.add_argument(
        "--rdma-rc-fake-get-mode",
        choices=["none", "status_only", "payload_memset"],
    )
    parser.add_argument("--rdma-rc-skip-client-copy", action="store_true")
    parser.add_argument("--verify-values", action="store_true")
    parser.add_argument("--verify-value-row-stride", type=int, default=1)
    parser.add_argument("--show-runner-logs", action="store_true")
    args = parser.parse_args()

    if args.server_count <= 0:
        raise ValueError("--server-count must be positive")
    if args.client_count <= 0:
        raise ValueError("--client-count must be positive")

    source_config_path = resolve_rdma_integration_config(
        args.server_count, args.config_path
    )
    if args.server_count == 1 and source_config_path is None:
        source_config_path = DEFAULT_RDMA_SINGLE_SHARD_CONFIG
    if args.server_count > 1 and source_config_path is None:
        source_config_path = DEFAULT_RDMA_MULTI_SHARD_CONFIG

    max_kv_num_per_request = (
        args.max_kv_num_per_request
        if args.max_kv_num_per_request is not None
        else max(1, args.batch_keys)
    )

    with tempfile.TemporaryDirectory(prefix="recstore_rdma_rc_benchmark_") as tmpdir:
        config_path = write_runtime_config(args, source_config_path, tmpdir)
        runner = PetPSClusterRunner(
            config_path=config_path,
            num_servers=args.server_count,
            num_clients=args.client_count,
            thread_num=args.thread_num,
            value_size=args.value_size,
            max_kv_num_per_request=max_kv_num_per_request,
            timeout=args.cluster_timeout,
            use_local_memcached=args.use_local_memcached,
            memcached_host=args.memcached_host,
            memcached_port=args.memcached_port,
            verbose=args.show_runner_logs,
            show_status_logs=args.show_runner_logs,
            show_memcached_logs=args.show_runner_logs,
            rdma_rc_qps_per_client_per_shard=(
                args.rdma_rc_qps_per_client_per_shard
            ),
            rdma_wait_timeout_ms=args.rdma_wait_timeout_ms,
            rdma_rc_profile_interval_ms=args.rdma_rc_profile_interval_ms,
            rdma_rc_server_coroutines_per_thread=(
                args.rdma_rc_server_coroutines_per_thread
            ),
            rdma_rc_fake_get_mode=args.rdma_rc_fake_get_mode,
            rdma_rc_skip_client_copy=args.rdma_rc_skip_client_copy,
        )

        summary_rows = []
        with runner.run():
            returncode, rows = run_benchmark_clients(runner, args)
            summary_rows.extend(rows)
            if returncode != 0:
                return returncode

    print_summary_table(summary_rows)
    print_aggregate_table(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
