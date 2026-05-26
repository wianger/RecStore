#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import threading

from petps_cluster_runner import PetPSClusterRunner, REPO_ROOT
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


def is_summary_line(line):
    return " phase=measure summary " in line or " phase=warmup summary " in line


def print_filtered_output(text, show_runner_logs, quiet):
    for line in text.splitlines():
        if quiet and not is_summary_line(line):
            continue
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
    if args.qps_per_client_per_shard is not None:
        cmd.append(
            "--rdma_rc_qps_per_client_per_shard="
            f"{args.qps_per_client_per_shard}"
        )
    if args.slots_per_qp is not None:
        cmd.append("--rdma_rc_slots_per_qp=" f"{args.slots_per_qp}")
    if args.rdma_wait_timeout_ms is not None:
        cmd.append(f"--rdma_wait_timeout_ms={args.rdma_wait_timeout_ms}")
    if args.profile_interval_ms is not None:
        cmd.append(
            "--rdma_rc_profile_interval_ms="
            f"{args.profile_interval_ms}"
        )
    if args.inline_bytes is not None:
        cmd.append("--rdma_rc_inline_bytes=" f"{args.inline_bytes}")
    if args.client_numa_id is not None:
        cmd.append("--rdma_rc_client_numa_id=" f"{args.client_numa_id}")
    if args.server_numa_id is not None:
        cmd.append("--rdma_rc_server_numa_id=" f"{args.server_numa_id}")
    if args.verify_values:
        cmd.append("--verify_values=true")
        cmd.append(
            f"--verify_value_row_stride={args.verify_value_row_stride}"
        )
    return cmd


def parse_client_numa_ids(value, client_count):
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    ids = [int(part) for part in parts]
    if len(ids) != client_count:
        raise ValueError(
            "--client-numa-ids must provide exactly one device id per client"
        )
    return ids


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


def _stream_process_output(
    client_index,
    stream,
    sink,
    show_runner_logs,
    is_stderr,
    log_path=None,
):
    prefix = f"[rdma-rc-client:{client_index}] "
    if is_stderr:
        prefix = f"[rdma-rc-client:{client_index}:stderr] "
    log_handle = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        for raw_line in iter(stream.readline, ""):
            line = raw_line.rstrip()
            sink.append(raw_line)
            if log_handle is not None:
                log_handle.write(raw_line)
                log_handle.flush()
            if show_runner_logs:
                print(prefix + line)
    finally:
        if log_handle is not None:
            log_handle.close()


def run_benchmark_clients(runner, args):
    if args.client_count == 1:
        completed = runner.run_client(
            build_benchmark_cmd(args),
            timeout=args.client_timeout,
            stream_output=args.show_runner_logs,
        )
        if not args.show_runner_logs:
            print_filtered_output(completed.stdout, args.show_runner_logs, args.quiet)
            print_filtered_output(completed.stderr, args.show_runner_logs, args.quiet)
        rows = collect_summary_rows(completed.stdout)
        for row in rows:
            row["client_index"] = 0
        return completed.returncode, rows

    env = runner.build_env()
    processes = []
    stdout_buffers = {}
    stderr_buffers = {}
    stream_threads = []
    log_dir = tempfile.mkdtemp(prefix="rdma_rc_client_logs_")
    deadline = time.monotonic() + args.client_timeout
    for client_index in range(args.client_count):
        cmd = runner.build_client_cmd(
            build_benchmark_cmd(args), client_index=client_index
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        processes.append((client_index, process))
        stdout_buffers[client_index] = []
        stderr_buffers[client_index] = []
        stdout_log_path = os.path.join(log_dir, f"client_{client_index}.stdout.log")
        stderr_log_path = os.path.join(log_dir, f"client_{client_index}.stderr.log")
        stdout_thread = threading.Thread(
            target=_stream_process_output,
            args=(
                client_index,
                process.stdout,
                stdout_buffers[client_index],
                args.show_runner_logs,
                False,
                stdout_log_path,
            ),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_stream_process_output,
            args=(
                client_index,
                process.stderr,
                stderr_buffers[client_index],
                args.show_runner_logs,
                True,
                stderr_log_path,
            ),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        stream_threads.extend([stdout_thread, stderr_thread])

    exit_codes = {}
    timed_out = False
    for client_index, process in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            break
        exit_codes[client_index] = process.returncode

    if timed_out:
        for _client_index, process in processes:
            terminate_process(process)

    for thread in stream_threads:
        thread.join(timeout=5)

    rows = []
    return_code = 124 if timed_out else 0
    for client_index, _process in processes:
        stdout = "".join(stdout_buffers.get(client_index, []))
        stderr = "".join(stderr_buffers.get(client_index, []))
        rc = exit_codes.get(client_index, 124 if timed_out else 0)
        if not args.show_runner_logs:
            print_filtered_output(stdout, args.show_runner_logs, args.quiet)
            print_filtered_output(stderr, args.show_runner_logs, args.quiet)
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
    parser.add_argument(
        "--qps-per-client-per-shard",
        "--rdma-rc-qps-per-client-per-shard",
        dest="qps_per_client_per_shard",
        type=int,
    )
    parser.add_argument(
        "--slots-per-qp",
        "--rdma-rc-slots-per-qp",
        dest="slots_per_qp",
        type=int,
    )
    parser.add_argument("--rdma-wait-timeout-ms", type=int)
    parser.add_argument(
        "--profile-interval-ms",
        "--rdma-rc-profile-interval-ms",
        dest="profile_interval_ms",
        type=int,
    )
    parser.add_argument(
        "--server-coroutines-per-thread",
        "--rdma-rc-server-coroutines-per-thread",
        dest="server_coroutines_per_thread",
        type=int,
    )
    parser.add_argument(
        "--inline-bytes",
        "--rdma-rc-inline-bytes",
        dest="inline_bytes",
        type=int,
    )
    parser.add_argument(
        "--client-numa-id",
        "--rdma-rc-client-numa-id",
        dest="client_numa_id",
        type=int,
    )
    parser.add_argument(
        "--client-numa-ids",
        dest="client_numa_ids",
        help="comma-separated RDMA device ids, one per client process",
    )
    parser.add_argument(
        "--server-numa-id",
        "--rdma-rc-server-numa-id",
        dest="server_numa_id",
        type=int,
    )
    parser.add_argument(
        "--fake-get-mode",
        "--rdma-rc-fake-get-mode",
        dest="fake_get_mode",
        choices=["none", "status_only", "payload_memset"],
    )
    parser.add_argument(
        "--skip-client-copy",
        "--rdma-rc-skip-client-copy",
        dest="skip_client_copy",
        action="store_true",
    )
    parser.add_argument("--verify-values", action="store_true")
    parser.add_argument("--verify-value-row-stride", type=int, default=1)
    parser.add_argument("--show-runner-logs", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress logs and print only the final aggregate summary",
    )
    args = parser.parse_args()

    if args.server_count <= 0:
        raise ValueError("--server-count must be positive")
    if args.client_count <= 0:
        raise ValueError("--client-count must be positive")
    if args.client_numa_id is not None and args.client_numa_ids is not None:
        raise ValueError(
            "--client-numa-id and --client-numa-ids are mutually exclusive"
        )
    if args.slots_per_qp is not None and args.slots_per_qp <= 0:
        raise ValueError("--slots-per-qp must be positive")
    client_numa_ids = parse_client_numa_ids(args.client_numa_ids, args.client_count)
    if args.op == "async_stream" and args.qps_per_client_per_shard is not None:
        slots_per_qp = args.slots_per_qp if args.slots_per_qp is not None else 1
        capacity = args.qps_per_client_per_shard * slots_per_qp
        if capacity < args.async_depth:
            raise ValueError(
                "async_stream requires qps_per_client_per_shard * slots_per_qp "
                ">= async_depth"
            )

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
            rdma_qps_per_client_per_shard=args.qps_per_client_per_shard,
            rdma_slots_per_qp=args.slots_per_qp,
            rdma_wait_timeout_ms=args.rdma_wait_timeout_ms,
            rdma_profile_interval_ms=args.profile_interval_ms,
            rdma_server_coroutines_per_thread=args.server_coroutines_per_thread,
            rdma_inline_bytes=args.inline_bytes,
            rdma_client_numa_id=args.client_numa_id,
            rdma_client_numa_ids=client_numa_ids,
            rdma_server_numa_id=args.server_numa_id,
            rdma_fake_get_mode=args.fake_get_mode,
            rdma_skip_client_copy=args.skip_client_copy,
        )

        summary_rows = []
        with runner.run():
            returncode, rows = run_benchmark_clients(runner, args)
            summary_rows.extend(rows)
            if returncode != 0:
                return returncode

    if not args.quiet:
        print_summary_table(summary_rows)
        print_aggregate_table(summary_rows)
    else:
        print_aggregate_table(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
