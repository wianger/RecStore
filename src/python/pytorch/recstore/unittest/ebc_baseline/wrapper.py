import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from .cases import MultiProcessCase, PrecisionCase
from .config import TEST_SCRIPTS_PATH, resolve_ps_endpoint, resolve_repo_config_path

if str(TEST_SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(TEST_SCRIPTS_PATH))

from ps_server_helpers import (
    check_ps_server_running,
    get_server_config,
    should_skip_server_start,
)
from ps_server_runner import PSServerRunner


def require_torchrec(testcase) -> None:
    try:
        import importlib.util

        has_torchrec = importlib.util.find_spec("torchrec") is not None
    except Exception:
        has_torchrec = False

    if not has_torchrec:
        testcase.skipTest("torchrec is not installed in this test environment")


def build_multiprocess_command(
    case: MultiProcessCase,
    script_path: Path,
    ps_endpoint: tuple[str, int],
    python_executable: str = sys.executable,
) -> list[str]:
    host, port = ps_endpoint
    cmd = [
        python_executable,
        str(script_path),
        "--num-embeddings",
        str(case.num_embeddings),
        "--embedding-dim",
        str(case.embedding_dim),
        "--batch-size",
        str(case.batch_size),
        "--world-size",
        str(case.world_size),
        "--seed",
        str(case.seed),
        "--ps-host",
        str(host),
        "--ps-port",
        str(port),
    ]
    if case.cpu:
        cmd.append("--cpu")
    return cmd


class EBCPrecisionModuleHarness:
    def __init__(
        self,
        should_skip_server_start: Callable[[], tuple[bool, Optional[str]]] = should_skip_server_start,
        check_ps_server_running: Callable[[], tuple[bool, list[int]]] = check_ps_server_running,
        get_server_config: Callable[[], dict] = get_server_config,
        server_runner_factory: Callable[..., PSServerRunner] = PSServerRunner,
        ready_delay_seconds: float = 2.0,
    ):
        self._should_skip_server_start = should_skip_server_start
        self._check_ps_server_running = check_ps_server_running
        self._get_server_config = get_server_config
        self._server_runner_factory = server_runner_factory
        self._ready_delay_seconds = ready_delay_seconds
        self.server_runner = None

    def setup(self) -> None:
        skip_server, reason = self._should_skip_server_start()
        if skip_server:
            print(f"[{reason}] Running tests assuming ps_server is already running")
            is_running, open_ports = self._check_ps_server_running()
            if is_running:
                print(f"PS Server verified running on ports: {open_ports}")
            else:
                print("Warning: PS Server ports not responding; tests may fail")
            return

        config = self._get_server_config()
        self.server_runner = self._server_runner_factory(
            server_path=config["server_path"],
            config_path=config["config_path"],
            log_dir=config["log_dir"],
            timeout=config["timeout"],
            num_shards=config["num_shards"],
            verbose=True,
        )
        if not self.server_runner.start():
            raise RuntimeError("Failed to start PS Server")

        if self._ready_delay_seconds > 0:
            time.sleep(self._ready_delay_seconds)

    def teardown(self) -> None:
        if self.server_runner is None:
            return

        try:
            if self.server_runner.is_running():
                self.server_runner.stop()
        except Exception as error:
            print(f"Warning: PS Server teardown raised non-fatal exception: {error}")
        finally:
            self.server_runner = None


def run_precision_case(case: PrecisionCase, precision_main) -> None:
    args = case.to_namespace()
    precision_main(args)


def run_multiprocess_case(
    case: MultiProcessCase,
    script_path: Path,
    active_config_path: Optional[Path | str] = None,
) -> None:
    config_path = active_config_path or resolve_repo_config_path()
    cmd = build_multiprocess_command(
        case=case,
        script_path=script_path,
        ps_endpoint=resolve_ps_endpoint(config_path),
    )
    subprocess.check_call(cmd)
