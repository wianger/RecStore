#!/bin/bash
# tools/env/test_ycsb.sh
# Purpose: Run diverse, fine-grained YCSB tests (KVDB, CCEH, multiple workloads).
# Usage: bash tools/env/test_ycsb.sh [install_prefix]

set -euo pipefail

# 1. Setup Environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

# Source the env script
source tools/env/init_ycsb_env.sh "${1:-}"

# 2. Define Test Runner Function
run_test() {
    local db_name="$1"      # e.g., kvdb, cceh
    local workload="$2"     # e.g., workloada, workloadc
    local test_id="$3"      # e.g., kvdb_a_min
    local desc="$4"         # e.g., "KVDB Minimal Resources - Update Heavy"
    local extra_args="$5"   # e.g., "-p ..."

    echo "------------------------------------------------------------"
    echo "Running Benchmark [${test_id}]: ${desc}"
    echo "------------------------------------------------------------"
    
    local LOG_FILE="${YCSB_LOG_DIR}/${test_id}.log"
    
    # Clean up previous data files to ensure isolation
    # Only remove known data files to avoid accidents
    rm -f "${YCSB_DATA_DIR}/shmvalue" "${YCSB_DATA_DIR}/ssdvalue"
    # Note: CCEH specific cleanup if needed (it usually uses the path directory)
    
    local prop_file="${db_name}.properties"
    if [ "$db_name" == "kvdb" ]; then
        prop_file="kv_db.properties"
    fi

    echo "Command: $YCSB_BIN -load -run -db $db_name -P $YCSB_WORKLOAD_DIR/$workload -P $YCSB_PROP_DIR/${prop_file} $extra_args"
    
    set +e
    "$YCSB_BIN" -load -run \
        -db "$db_name" \
        -P "${YCSB_WORKLOAD_DIR}/${workload}" \
        -P "${YCSB_PROP_DIR}/${prop_file}" \
        -p "hybridkv.path=${YCSB_DATA_DIR}" \
        -p "cceh.path=${YCSB_DATA_DIR}" \
        ${extra_args} \
        -s > "$LOG_FILE" 2>&1
    
    local exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        echo "✅ PASS: ${test_id}"
        # Print summary statistics (usually at the end of log)
        tail -n 10 "$LOG_FILE" | grep -E "Operations|AverageLatency" || true
    else
        echo "❌ FAIL: ${test_id} (Exit Code: $exit_code)"
        echo "Last 20 lines of log:"
        tail -n 20 "$LOG_FILE"
        exit $exit_code
    fi
    echo ""
}

# 3. Execute Tests

# === Test Suite 1: HybridKV (KVDB) ===

# Configuration for Benchmark (Minimal Resources but Higher Ops)
# 64MB RAM, 64MB SSD
# Record Count: 1000 (small to save space)
# Op Count: 100,000 (enough to get ~seconds of run to measure throughput stable)
KVDB_BENCH_ARGS="-p hybridkv.shmcapacity=67108864 -p hybridkv.ssdcapacity=67108864 -p recordcount=1000 -p operationcount=100000"

# [KVDB-A] Workload A: Update Heavy (50/50)
run_test "kvdb" "workloada" "kvdb_bench_workloada" \
    "HybridKV - Workload A (Update Heavy 50/50)" \
    "$KVDB_BENCH_ARGS"

# [KVDB-B] Workload B: Read Mostly (95/5)
run_test "kvdb" "workloadb" "kvdb_bench_workloadb" \
    "HybridKV - Workload B (Read Mostly 95/5)" \
    "$KVDB_BENCH_ARGS"

# [KVDB-C] Workload C: Read Only (100% Read)
run_test "kvdb" "workloadc" "kvdb_bench_workloadc" \
    "HybridKV - Workload C (Read Only 100%)" \
    "$KVDB_BENCH_ARGS"

# [KVDB-D] Workload D: Read Latest (Newest records)
run_test "kvdb" "workloadd" "kvdb_bench_workloadd" \
    "HybridKV - Workload D (Read Latest)" \
    "$KVDB_BENCH_ARGS"

# [KVDB-F] Workload F: Read-Modify-Write
run_test "kvdb" "workloadf" "kvdb_bench_workloadf" \
    "HybridKV - Workload F (Read-Modify-Write)" \
    "$KVDB_BENCH_ARGS"


# === Test Suite 2: CCEH ===
# CCEH requires SPDK/Hugepages, which might not be available in CI.
# Enable only if explicitly requested.
if [ "${ENABLE_CCEH_TEST:-0}" -eq 1 ]; then
    CCEH_MIN_ARGS="-p cceh.capacity=20000 -p cceh.io_backend_type=IOURING -p cceh.queue_cnt=512 -p recordcount=1000 -p operationcount=100000"

    # [CCEH-A] Workload A
    run_test "cceh" "workloada" "cceh_bench_workloada" \
        "CCEH - Workload A" \
        "$CCEH_MIN_ARGS"
else
    echo "⚠️  Skipping CCEH Benchmark (ENABLE_CCEH_TEST not set)"
fi


echo "🎉 All YCSB benchmarks completed successfully."
