#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_BINARY="${ROOT_DIR}/build/bin/rdma_rc_transport_benchmark"
RUNNER_SCRIPT="${ROOT_DIR}/src/test/scripts/run_rdma_rc_transport_benchmark.py"

if [[ ! -x "${BENCHMARK_BINARY}" ]]; then
  echo "benchmark binary not found: ${BENCHMARK_BINARY}" >&2
  echo "build it first:" >&2
  echo "  cmake --build ${ROOT_DIR}/build --target rdma_rc_transport_benchmark" >&2
  exit 1
fi

cd "${ROOT_DIR}"

exec python3 "${RUNNER_SCRIPT}" \
  --benchmark-binary "${BENCHMARK_BINARY}" \
  --server-count 1 \
  --client-count 8 \
  --thread-num 32 \
  --iterations 20 \
  --rounds 30 \
  --warmup-rounds 10 \
  --batch-keys 500 \
  --value-size 512 \
  --op async_stream \
  --async-depth 128 \
  --report-mode summary \
  --qps-per-client-per-shard 256 \
  --client-timeout 1800 \
  --cluster-timeout 120 \
  --use-local-memcached auto \
  "$@"
