#!/bin/bash
set -euo pipefail

# Generic runner for packed RecStore artifacts
# Usage:
#   tools/packaging/runner.sh <package_root> <binary_rel_path> [--ready-pattern PATTERN] [--timeout SECONDS] [--log LOG_FILE]
# Defaults:
#   ready-pattern: "listening on"
#   timeout: 180
#   log: <package_root>/../logs/run.log

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <package_root> <binary_rel_path> [--ready-pattern PATTERN] [--timeout SECONDS] [--log LOG_FILE]" >&2
  exit 2
fi

PKG_ROOT="$1"; shift
BIN_REL="$1"; shift

READY_PATTERN="listening on"
TIMEOUT=180
LOG_DIR="$(dirname "$PKG_ROOT")/../logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/run.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ready-pattern)
      READY_PATTERN="$2"; shift 2 ;;
    --timeout)
      TIMEOUT="$2"; shift 2 ;;
    --log)
      LOG_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

BIN_PATH="${PKG_ROOT}/${BIN_REL}"
if [[ ! -x "$BIN_PATH" ]]; then
  echo "Binary not found or not executable: $BIN_PATH" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${PKG_ROOT}/deps/lib:${PKG_ROOT}/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
ls -la "${PKG_ROOT}/deps/lib" || true

# show ldd and try rpath injection
echo "Preflight ldd: ${BIN_PATH}"
ldd "$BIN_PATH" || true
if ldd "$BIN_PATH" 2>&1 | grep -q "not found"; then
  if command -v patchelf >/dev/null 2>&1; then
    echo "Injecting rpath into ${BIN_PATH}"
    patchelf --set-rpath "\$ORIGIN/../deps/lib:\$ORIGIN/../lib" "$BIN_PATH" || true
    echo "New rpath:" && patchelf --print-rpath "$BIN_PATH" || true
    echo "Reflight ldd after rpath:" && ldd "$BIN_PATH" || true
  else
    echo "patchelf not available; skipping rpath injection"
  fi
fi

echo "Starting ${BIN_PATH} ... (log: ${LOG_FILE})"
"$BIN_PATH" >"${LOG_FILE}" 2>&1 &
PID=$!

trap 'if kill -0 ${PID} 2>/dev/null; then kill ${PID} || true; wait ${PID} 2>/dev/null || true; fi' EXIT

START_TS=$(date +%s)
while true; do
  if grep -q "$READY_PATTERN" "${LOG_FILE}"; then
    echo "Process ready (matched pattern: ${READY_PATTERN})"
    break
  fi
  if ! kill -0 ${PID} >/dev/null 2>&1; then
    echo "Process exited prematurely. Logs:" && cat "${LOG_FILE}"
    exit 1
  fi
  NOW_TS=$(date +%s)
  if (( NOW_TS - START_TS >= TIMEOUT )); then
    echo "Timeout (${TIMEOUT}s) waiting for readiness. Logs:" && cat "${LOG_FILE}"
    exit 1
  fi
  sleep 1
done

kill ${PID} || true
wait ${PID} 2>/dev/null || true
echo "Run completed successfully."
