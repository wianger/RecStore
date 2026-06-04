#!/bin/bash

# Usage:
# (with nohup/screen)
# ./run_single_day.sh --help
# Example:
# ./run_single_day.sh --custom --dataset-size 65535

DEFAULT_PROCESSED_DATASET_PATH="./processed_day_0_data"
DEFAULT_BATCH_SIZE=1024
DEFAULT_LEARNING_RATE=0.005
DEFAULT_EPOCHS=1
DEFAULT_DATASET_SIZE=4194304
DEFAULT_ENABLE_PREFETCH=true
DEFAULT_PREFETCH_DEPTH=2
DEFAULT_FUSE_EMB=true
DEFAULT_FUSE_K=30
DEFAULT_TRACE_FILE=""
DEFAULT_EMBEDDING_STORAGE="hbm"

DLRM_PATH="$(pwd)"
REPO_ROOT="$(cd "${DLRM_PATH}/../.." && pwd)"
TORCHREC_SCRIPT="${DLRM_PATH}/tests/dlrm_main_torchrec_single.py"
CUSTOM_SCRIPT="${DLRM_PATH}/tests/dlrm_main_single_day.py"
DEFAULT_PS_SERVER_PATH="${REPO_ROOT}/build/bin/ps_server"
DEFAULT_PS_LOG_DIR="/tmp"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: Neither python nor python3 is available in PATH" >&2
    exit 1
fi

DEFAULT_PS_CONFIG_PATH="$("$PYTHON_BIN" - "$REPO_ROOT" <<'PY'
import sys

repo_root = sys.argv[1]
sys.path.insert(0, repo_root)

from tools.config.recstore_config_path import resolve_recstore_config_path

print(resolve_recstore_config_path())
PY
)"

use_torchrec=false
use_random_dataset=false
start_ps=false
ps_server_path=$DEFAULT_PS_SERVER_PATH
ps_config_path=$DEFAULT_PS_CONFIG_PATH
ps_log_dir=$DEFAULT_PS_LOG_DIR
ps_server_pid=""
ps_server_started=false
dataset_size=$DEFAULT_DATASET_SIZE
processed_dataset_path=$DEFAULT_PROCESSED_DATASET_PATH
batch_size=$DEFAULT_BATCH_SIZE
learning_rate=$DEFAULT_LEARNING_RATE
epochs=$DEFAULT_EPOCHS
enable_prefetch=$DEFAULT_ENABLE_PREFETCH
prefetch_depth=$DEFAULT_PREFETCH_DEPTH
fuse_emb_tables=$DEFAULT_FUSE_EMB
fuse_k=$DEFAULT_FUSE_K
trace_file=$DEFAULT_TRACE_FILE
allow_tf32=false
embedding_storage=$DEFAULT_EMBEDDING_STORAGE
gin_config=""
gin_bindings=()

show_help() {
    echo "DLRM Training Script with Performance Metrics"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                  Show this help message and exit"
    echo ""
    echo "Configuration (defaults used if not specified):"
    echo "  --dataset-size SIZE         Dataset size (default: $DEFAULT_DATASET_SIZE)"
    echo "  --dataset-path PATH         Processed dataset path (default: $DEFAULT_PROCESSED_DATASET_PATH)"
    echo "  --random-dataset            Use generated random data instead of processed day_0 files"
    echo "  --batch-size SIZE           Batch size (default: $DEFAULT_BATCH_SIZE)"
    echo "  --learning-rate RATE        Learning rate (default: $DEFAULT_LEARNING_RATE)"
    echo "  --epochs COUNT              Number of epochs (default: $DEFAULT_EPOCHS)"
    echo "  --gin-config PATH           Load gin launch config file"
    echo "  --gin-binding EXPR          Override gin setting (repeatable)"
    echo ""
    echo "Mode Selection (choose one):"
    echo "  --torchrec                  Use TorchRec baseline (default: custom recstore)"
    echo "  --custom                    Use custom recstore (default behavior)"
    echo "  --ps                        Auto-start local ps_server for RecStore mode"
    echo ""
    echo "RecStore Prefetch (RecStore mode only):"
    echo "  --enable-prefetch           Enable async prefetch (default: $DEFAULT_ENABLE_PREFETCH)"
    echo "  --disable-prefetch          Disable async prefetch"
    echo "  --prefetch-depth N          Prefetch queue depth (default: $DEFAULT_PREFETCH_DEPTH)"
    echo ""
    echo "RecStore Fusion (RecStore mode only):"
    echo "  --enable-fuse-emb           Enable embedding table fusion (default: $DEFAULT_FUSE_EMB)"
    echo "  --disable-fuse-emb          Disable embedding table fusion"
    echo "  --fuse-k K                  Bit prefix shift k (default: $DEFAULT_FUSE_K)"
    echo "  --trace-file PATH           Chrome trace output file (optional)"
    echo "  --allow-tf32                Enable TF32 for MatMul"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --torchrec)
            use_torchrec=true
            shift
            ;;
        --custom)
            use_torchrec=false
            shift
            ;;
        --random-dataset)
            use_random_dataset=true
            shift
            ;;
        --ps)
            start_ps=true
            shift
            ;;
        --dataset-size)
            if [[ -n "$2" && "$2" != -* ]]; then
                dataset_size="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --dataset-path)
            if [[ -n "$2" && "$2" != -* ]]; then
                processed_dataset_path="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --batch-size)
            if [[ -n "$2" && "$2" != -* ]]; then
                batch_size="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --learning-rate)
            if [[ -n "$2" && "$2" != -* ]]; then
                learning_rate="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --epochs)
            if [[ -n "$2" && "$2" != -* ]]; then
                epochs="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --gin-config)
            if [[ -n "$2" && "$2" != -* ]]; then
                gin_config="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --gin-binding)
            if [[ -n "$2" && "$2" != -* ]]; then
                gin_bindings+=("$2")
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --enable-prefetch)
            enable_prefetch=true
            shift
            ;;
        --disable-prefetch|--no-prefetch)
            enable_prefetch=false
            shift
            ;;
        --prefetch-depth)
            if [[ -n "$2" && "$2" != -* ]]; then
                prefetch_depth="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --enable-fuse-emb)
            fuse_emb_tables=true
            shift
            ;;
        --disable-fuse-emb|--no-fuse-emb)
            fuse_emb_tables=false
            shift
            ;;
        --fuse-k)
            if [[ -n "$2" && "$2" != -* ]]; then
                fuse_k="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --trace-file)
            if [[ -n "$2" && "$2" != -* ]]; then
                trace_file="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --ps-host)
            if [[ -n "$2" && "$2" != -* ]]; then
                ps_host="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --ps-port)
            if [[ -n "$2" && "$2" != -* ]]; then
                ps_port="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --embedding-storage)
            if [[ -n "$2" && "$2" != -* ]]; then
                embedding_storage="$2"
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                show_help
                exit 1
            fi
            ;;
        --allow-tf32)
            allow_tf32=true
            shift
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            show_help
            exit 1
            ;;
    esac
done

if [ "$use_torchrec" = true ]; then
    script_to_run="$TORCHREC_SCRIPT"
    mode="TorchRec"
else
    script_to_run="$CUSTOM_SCRIPT"
    mode="RecStore"
fi

if [ "$mode" = "TorchRec" ] && [ "$start_ps" = true ]; then
    echo "Warning: --ps is only used in RecStore mode; ignoring it for TorchRec." >&2
    start_ps=false
fi

set_default_local_report_env() {
    export RECSTORE_REPORT_MODE="${RECSTORE_REPORT_MODE:-local}"
    export RECSTORE_REPORT_LOCAL_SINK="${RECSTORE_REPORT_LOCAL_SINK:-both}"
    export RECSTORE_REPORT_JSONL_PATH="${RECSTORE_REPORT_JSONL_PATH:-/tmp/recstore_report_events.jsonl}"
}

check_ps_ports_ready() {
    "$PYTHON_BIN" - "$ps_config_path" <<'PY'
import json
import socket
import sys

config_path = sys.argv[1]
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
except OSError:
    sys.exit(1)

servers = config.get("cache_ps", {}).get("servers", [])
if not servers:
    servers = config.get("distributed_client", {}).get("servers", [])
if not servers:
    sys.exit(1)

targets = []
for server in servers:
    host = server.get("host", "127.0.0.1")
    port = server.get("port")
    if port is not None:
        targets.append((host, int(port)))
targets = sorted(set(targets))
if not targets:
    sys.exit(1)

for host, port in targets:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            pass
    except OSError:
        sys.exit(1)
sys.exit(0)
PY
}

stop_ps_server() {
    if [ "$ps_server_started" != true ] || [ -z "$ps_server_pid" ]; then
        return
    fi
    if kill -0 "$ps_server_pid" 2>/dev/null; then
        echo "Stopping ps_server (pid: $ps_server_pid)..."
        kill "$ps_server_pid" 2>/dev/null || true
        wait "$ps_server_pid" 2>/dev/null || true
    fi
}

on_exit() {
    stop_ps_server
}

start_ps_server_if_requested() {
    if [ "$start_ps" != true ]; then
        return
    fi

    if [ ! -x "$ps_server_path" ]; then
        echo "Error: ps_server not found or not executable at $ps_server_path" >&2
        exit 1
    fi
    if [ ! -f "$ps_config_path" ]; then
        echo "Error: ps_server config not found at $ps_config_path" >&2
        exit 1
    fi

    set_default_local_report_env

    if check_ps_ports_ready; then
        echo "ps_server is already listening on configured ports; reusing it."
        return
    fi

    mkdir -p "$ps_log_dir"
    ps_log_file="${ps_log_dir}/run_single_day_ps_server.$(date +%Y%m%d%H%M%S).log"
    echo "Starting ps_server: $ps_server_path --config_path $ps_config_path"
    echo "ps_server log: $ps_log_file"
    "$ps_server_path" --config_path "$ps_config_path" > "$ps_log_file" 2>&1 &
    ps_server_pid=$!
    ps_server_started=true
    trap on_exit EXIT

    for _ in $(seq 1 60); do
        if ! kill -0 "$ps_server_pid" 2>/dev/null; then
            echo "Error: ps_server exited before becoming ready. Log tail:" >&2
            tail -n 80 "$ps_log_file" >&2
            exit 1
        fi
        if check_ps_ports_ready; then
            echo "ps_server is ready (pid: $ps_server_pid)"
            return
        fi
        sleep 1
    done

    echo "Error: timed out waiting for ps_server to listen on configured ports. Log tail:" >&2
    tail -n 80 "$ps_log_file" >&2
    exit 1
}

echo "=========================================="
echo "DLRM Training Configuration"
echo "=========================================="
echo "Mode:                     $mode"
echo "Auto-start ps_server:     $start_ps"
if [ "$use_random_dataset" = true ]; then
echo "Dataset Source:           Random synthetic"
else
echo "Dataset Source:           Processed day_0 files"
fi
echo "Dataset Size:             $dataset_size"
echo "Dataset Path:             $processed_dataset_path"
echo "Batch Size:               $batch_size"
echo "Learning Rate:            $learning_rate"
echo "Epochs:                   $epochs"
echo "Script Path:              $script_to_run"
echo "Python:                   $(command -v "$PYTHON_BIN")"
if [ -n "$gin_config" ]; then
echo "Gin Config:               $gin_config"
fi
if [ "$mode" = "RecStore" ]; then
echo "Enable Prefetch:         $enable_prefetch"
echo "Prefetch Depth:          $prefetch_depth"
echo "Fusion Enabled:          $fuse_emb_tables"
echo "Fusion k (shift):        $fuse_k"
if [ "$start_ps" = true ]; then
echo "PS Server Path:          $ps_server_path"
echo "PS Config Path:          $ps_config_path"
fi
if [ -n "$trace_file" ]; then
echo "Trace File:              $trace_file"
fi
fi
if [ "$mode" = "TorchRec" ]; then
echo "Embedding Storage:       $embedding_storage"
fi
echo "=========================================="

if [[ ! -f "$script_to_run" ]]; then
    echo "Error: Script not found at $script_to_run" >&2
    exit 1
fi

if [ "$use_random_dataset" != true ]; then
    if [ ! -d "$processed_dataset_path" ]; then
        echo "Error: Processed dataset not found at $processed_dataset_path"
        echo "Please run the preprocessing script first:"
        echo "bash scripts/process_single_day.sh <raw_data_dir> $processed_dataset_path"
        exit 1
    fi

    required_files=("day_0_dense.npy" "day_0_sparse.npy" "day_0_labels.npy")
    for file in "${required_files[@]}"; do
        if [ ! -f "$processed_dataset_path/$file" ]; then
            echo "Error: Required file $file not found in $processed_dataset_path"
            exit 1
        fi
    done

    echo "✓ All required data files found"

    if [ -f "$processed_dataset_path/day_0_labels.npy" ]; then
        detected_size=$($PYTHON_BIN -c "import numpy as np; print(np.load('${processed_dataset_path}/day_0_labels.npy', mmap_mode='r').shape[0])")
        if [ -n "$detected_size" ]; then
            echo "Auto-detected dataset size: $detected_size"
            dataset_size=$detected_size
        fi
    fi
else
    echo "✓ Random synthetic dataset mode enabled"
fi

echo ""

start_time=$(date +%s.%N)
start_seconds=$(date +%s)

start_ps_server_if_requested

echo "Starting training..."
extra_args=()
if [ "$allow_tf32" = true ]; then
    extra_args+=(--allow_tf32)
fi
if [ -n "$gin_config" ]; then
    extra_args+=(--gin_config "$gin_config")
fi
for gin_binding in "${gin_bindings[@]}"; do
    extra_args+=(--gin_binding "$gin_binding")
done
extra_args+=(--dataset-size "$dataset_size")
if [ "$use_random_dataset" = true ]; then
    extra_args+=(--random-dataset)
fi

if [ "$mode" = "RecStore" ]; then
    if [ "$enable_prefetch" = true ]; then
        extra_args+=(--enable_prefetch)
        extra_args+=(--prefetch_depth "$prefetch_depth")
    fi
    if [ "$fuse_emb_tables" = true ]; then
        extra_args+=(--fuse-emb-tables)
    else
        extra_args+=(--no-fuse-emb-tables)
    fi
    extra_args+=(--fuse-k "$fuse_k")
    if [ -n "$trace_file" ]; then
        extra_args+=(--trace_file "$trace_file")
    fi
    if [ -n "$ps_port" ]; then
        extra_args+=(--ps-port "$ps_port")
    fi
else 
    # TorchRec mode args
    if [ -n "$embedding_storage" ]; then
         extra_args+=(--embedding_storage "$embedding_storage")
    fi
    if [ -n "$trace_file" ]; then
        extra_args+=(--trace_file "$trace_file")
    fi
fi
data_args=(--single_day_mode)
if [ "$use_random_dataset" != true ]; then
    data_args+=(--in_memory_binary_criteo_path "$processed_dataset_path" --mmap_mode)
fi

launcher_args=(--nnodes 1 --nproc_per_node 1)
if [ "$start_ps" = true ]; then
    launcher_args+=(--standalone)
else
    launcher_args+=(--rdzv_backend c10d --rdzv_endpoint localhost --rdzv_id "run-$(date +%s)" --role trainer)
fi

echo "Executing command: $PYTHON_BIN -m torch.distributed.run ${launcher_args[*]} $script_to_run ${data_args[@]} --batch_size $batch_size --learning_rate $learning_rate --epochs $epochs --pin_memory --embedding_dim 128 ${extra_args[@]} --adagrad"

$PYTHON_BIN -m torch.distributed.run \
    "${launcher_args[@]}" \
    $script_to_run \
    "${data_args[@]}" \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --epochs $epochs \
    --pin_memory \
    --embedding_dim 128 \
    "${extra_args[@]}" \
    --adagrad > training_output.${dataset_size}.${mode}.pf$( [ "$enable_prefetch" = true ] && echo "$prefetch_depth" || echo 0 ).f$( [ "$fuse_emb_tables" = true ] && echo 1 || echo 0 ).$(date +%Y%m%d%H%M%S).log 2>&1
train_exit_code=$?


end_time=$(date +%s.%N)
end_seconds=$(date +%s)

duration=$((end_seconds - start_seconds))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "=========================================="
echo "Training Execution Summary"
echo "=========================================="
echo "Start Time:               $(date -d "@$(echo $start_time | cut -d. -f1)" '+%Y-%m-%d %H:%M:%S')"
echo "End Time:                 $(date -d "@$(echo $end_time | cut -d. -f1)" '+%Y-%m-%d %H:%M:%S')"
echo "Total Duration:           $(printf "%02d" $hours):$(printf "%02d" $minutes):$(printf "%02d" $seconds)"
echo "=========================================="


total_seconds=$(echo "$end_time - $start_time" | bc)
# uid="dlrm_${dataset_size}_${mode}_${epochs}_gpu"
echo "Uploading training duration: $total_seconds seconds"
# python report_uploader.py dlrm_training_metrics "$uid" training_duration_seconds "$total_seconds"
# if [ $? -ne 0 ]; then
#     echo "Warning: Data upload failed. Continuing with summary output."
# fi

if [ "$train_exit_code" -ne 0 ]; then
    echo "Training command failed with exit code: $train_exit_code" >&2
    exit "$train_exit_code"
fi
