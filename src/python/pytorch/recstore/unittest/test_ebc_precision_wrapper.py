import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

PYTORCH_ROOT = Path(__file__).resolve().parents[2]
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from recstore.unittest.ebc_baseline.cases import (
    cuda_precision_case,
    default_multiprocess_case,
    default_precision_cases,
)
from recstore.unittest.ebc_baseline.config import (
    resolve_ps_endpoint as _resolve_ps_endpoint,
    resolve_repo_config_path as _resolve_repo_config_path,
)
from recstore.unittest.ebc_baseline.wrapper import (
    EBCPrecisionModuleHarness,
    require_torchrec,
    run_multiprocess_case,
)


TEST_MODULE_PATH = Path(__file__).with_name("test_ebc_precision.py")
MP_TEST_MODULE_PATH = Path(__file__).with_name("test_ebc_precision_multiprocess.py")

_module_harness = EBCPrecisionModuleHarness()


def _lazy_import_test_module():
    spec = importlib.util.spec_from_file_location(
        "test_ebc_precision_module",
        TEST_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setUpModule():
    print("\n" + "=" * 70)
    print("SETUP MODULE: Initializing EBC Precision Test Suite")
    print("=" * 70)
    _module_harness.setup()


def tearDownModule():
    print("\n" + "=" * 70)
    print("Stopping PS Server")
    print("=" * 70)
    _module_harness.teardown()


class TestEBCPrecision(unittest.TestCase):
    def _run_case(self, case):
        require_torchrec(self)
        from recstore.unittest.ebc_baseline.single_process import run_precision

        run_precision(case.to_namespace())

    def test_basic_precision_cpu(self):
        self._run_case(default_precision_cases()[0])

    def test_small_batch_precision(self):
        self._run_case(default_precision_cases()[1])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_precision(self):
        require_torchrec(self)
        try:
            from recstore.unittest.ebc_baseline.single_process import run_precision

            run_precision(cuda_precision_case().to_namespace())
        except ImportError as error:
            self.skipTest(f"CUDA test skipped due to import error: {error}")

    def test_multiprocess_precision(self):
        require_torchrec(self)
        config_path = _resolve_repo_config_path()
        if _module_harness.server_runner and _module_harness.server_runner.config_path:
            config_path = str(_module_harness.server_runner.config_path)

        try:
            run_multiprocess_case(
                case=default_multiprocess_case(),
                script_path=MP_TEST_MODULE_PATH,
                active_config_path=config_path,
            )
        except subprocess.CalledProcessError as error:
            self.fail(
                "Multiprocess precision test failed with exit code "
                f"{error.returncode}"
            )


if __name__ == "__main__":
    unittest.main()
