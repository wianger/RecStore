from __future__ import annotations

from pathlib import Path

from .common import REPO_ROOT


BUILD_BIN_DIR = REPO_ROOT / "build" / "bin"
PS_TRANSPORT_BENCHMARK_BIN = BUILD_BIN_DIR / "ps_transport_benchmark"
RECSTORE_MIXED_BENCHMARK_BIN = BUILD_BIN_DIR / "recstore_mixed_benchmark"
LOCAL_SHM_PS_SERVER_BIN = BUILD_BIN_DIR / "local_shm_ps_server"
BRPC_PS_SERVER_BIN = BUILD_BIN_DIR / "brpc_ps_server"
