#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.benchmarks.run_rdma_rc_transport_benchmark import *  # noqa: F401,F403
from tools.benchmarks.run_rdma_rc_transport_benchmark import (  # noqa: F401
    _stream_process_output,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
