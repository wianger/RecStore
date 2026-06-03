#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.benchmarks.run_benchmark_ps import *  # noqa: F401,F403
from tools.benchmarks.run_benchmark_ps import main


if __name__ == "__main__":
    raise SystemExit(main())
