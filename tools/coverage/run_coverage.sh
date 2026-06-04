#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build_coverage"
REPORT_DIR="${BUILD_DIR}/coverage"
CPP_REPORT_DIR="${REPORT_DIR}/cpp"
PY_REPORT_DIR="${REPORT_DIR}/python"

CPP_TEST_REGEX=""
SKIP_BUILD=0
SKIP_CPP=0
SKIP_PY=0

usage() {
  cat <<'EOF'
Usage: bash tools/coverage/run_coverage.sh [options]

Generate C++ and Python coverage reports in one command.

Options:
  --build-dir <dir>        Coverage build directory (default: build_coverage)
  --cpp-test-regex <regex> Run only C++ tests matching regex via ctest -R
  --skip-build             Reuse existing build directory without re-configure/rebuild
  --skip-cpp               Skip C++ coverage collection
  --skip-python            Skip Python coverage collection
  -h, --help               Show this help

Output:
  C++ HTML report   : build_coverage/coverage/cpp/index.html
  C++ XML report    : build_coverage/coverage/cpp/coverage.xml
  C++ text summary  : build_coverage/coverage/cpp/coverage.txt
  Python HTML report: build_coverage/coverage/python/index.html
  Python XML report : build_coverage/coverage/python/coverage.xml
  Python text report: build_coverage/coverage/python/coverage.txt
EOF
}

require_cmd() {
  local cmd="$1"
  local install_hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: ${cmd}"
    echo "Install hint: ${install_hint}"
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      if [[ $# -lt 2 ]]; then
        echo "--build-dir requires a value"
        exit 1
      fi
      BUILD_DIR="$2"
      shift 2
      ;;
    --cpp-test-regex)
      if [[ $# -lt 2 ]]; then
        echo "--cpp-test-regex requires a value"
        exit 1
      fi
      CPP_TEST_REGEX="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --skip-cpp)
      SKIP_CPP=1
      shift
      ;;
    --skip-python)
      SKIP_PY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "${SKIP_CPP}" -eq 1 && "${SKIP_PY}" -eq 1 ]]; then
  echo "Nothing to do: both --skip-cpp and --skip-python are set."
  exit 1
fi

REPORT_DIR="${BUILD_DIR}/coverage"
CPP_REPORT_DIR="${REPORT_DIR}/cpp"
PY_REPORT_DIR="${REPORT_DIR}/python"

if [[ "${SKIP_CPP}" -eq 0 ]]; then
  require_cmd gcovr "python3 -m pip install gcovr"
  require_cmd cmake "sudo apt-get install cmake"
fi

if [[ "${SKIP_PY}" -eq 0 ]]; then
  if ! python3 -m coverage --version >/dev/null 2>&1; then
    echo "Missing python package: coverage"
    echo "Install hint: python3 -m pip install coverage"
    exit 1
  fi
fi

mkdir -p "${CPP_REPORT_DIR}" "${PY_REPORT_DIR}"

if [[ "${SKIP_BUILD}" -eq 0 && "${SKIP_CPP}" -eq 0 ]]; then
  echo "[1/6] Configuring coverage build at ${BUILD_DIR}"
  CMAKE_ARGS=(
    -S "${REPO_ROOT}"
    -B "${BUILD_DIR}"
    -DENABLE_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.10 \
    -DCMAKE_C_FLAGS="--coverage -O0 -g" \
    -DCMAKE_CXX_FLAGS="--coverage -O0 -g"
  )
  if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
    CMAKE_ARGS+=(
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
      -DCUDAToolkit_ROOT=/usr/local/cuda
    )
  fi
  cmake "${CMAKE_ARGS[@]}"

  echo "[2/6] Building coverage targets"
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

if [[ "${SKIP_CPP}" -eq 0 ]]; then
  echo "[3/6] Running C++ tests for coverage"
  find "${BUILD_DIR}" -name '*.gcda' -delete || true
  if [[ -n "${CPP_TEST_REGEX}" ]]; then
    ctest --test-dir "${BUILD_DIR}" --output-on-failure -R "${CPP_TEST_REGEX}"
  else
    ctest --test-dir "${BUILD_DIR}" --output-on-failure
  fi

  echo "[4/6] Generating C++ coverage reports"
  eval "$(python3 "${REPO_ROOT}/tools/coverage/coverage_scope.py" \
    --labeler "${REPO_ROOT}/.github/labeler.yml" \
    --format gcovr)"
  gcovr \
    --root "${REPO_ROOT}" \
    --object-directory "${BUILD_DIR}" \
    "${COVERAGE_SCOPE_ARGS[@]}" \
    --exclude ".*third_party/.*" \
    --exclude "build_coverage/.*" \
    --exclude ".*/build_coverage/.*" \
    --exclude "build/.*" \
    --exclude ".*/build/.*" \
    --exclude-directories ".*/CMakeFiles/[0-9.]+/CompilerId.*" \
    --xml-pretty --xml "${CPP_REPORT_DIR}/coverage.xml" \
    --html-details "${CPP_REPORT_DIR}/index.html" \
    --txt "${CPP_REPORT_DIR}/coverage.txt" \
    --print-summary
fi

if [[ "${SKIP_PY}" -eq 0 ]]; then
  echo "[5/6] Running Python tests for coverage"
  LIB_PATH="${BUILD_DIR}/lib/lib_recstore_ops.so"
  if [[ ! -f "${LIB_PATH}" ]]; then
    echo "${LIB_PATH} does not exist. Build first or run without --skip-build."
    exit 1
  fi

  export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="${REPO_ROOT}/src/python/pytorch:${PYTHONPATH:-}"
  export PS_SERVER_PATH="${BUILD_DIR}/bin/ps_server"

  python3 -m coverage erase
  (
    cd "${REPO_ROOT}/src/test/framework/pytorch"
    python3 -m coverage run --parallel-mode test_client.py "${LIB_PATH}"
  )
  (
    cd "${REPO_ROOT}/src/python/pytorch"
    python3 -m coverage run --parallel-mode -m unittest recstore.unittest.test_dist_emb
  )

  echo "[6/6] Generating Python coverage reports"
  python3 -m coverage combine \
    "${REPO_ROOT}/src/test/framework/pytorch" \
    "${REPO_ROOT}/src/python/pytorch"
  eval "$(python3 "${REPO_ROOT}/tools/coverage/coverage_scope.py" \
    --labeler "${REPO_ROOT}/.github/labeler.yml" \
    --format coverage)"
  python3 -m coverage html "${COVERAGE_SCOPE_ARGS[@]}" -d "${PY_REPORT_DIR}"
  python3 -m coverage xml "${COVERAGE_SCOPE_ARGS[@]}" -o "${PY_REPORT_DIR}/coverage.xml"
  python3 -m coverage report "${COVERAGE_SCOPE_ARGS[@]}" -m | tee "${PY_REPORT_DIR}/coverage.txt"
fi

echo
echo "Coverage reports generated:"
if [[ "${SKIP_CPP}" -eq 0 ]]; then
  echo "  C++ HTML   : ${CPP_REPORT_DIR}/index.html"
  echo "  C++ XML    : ${CPP_REPORT_DIR}/coverage.xml"
  echo "  C++ Text   : ${CPP_REPORT_DIR}/coverage.txt"
fi
if [[ "${SKIP_PY}" -eq 0 ]]; then
  echo "  Python HTML: ${PY_REPORT_DIR}/index.html"
  echo "  Python XML : ${PY_REPORT_DIR}/coverage.xml"
  echo "  Python Text: ${PY_REPORT_DIR}/coverage.txt"
fi
