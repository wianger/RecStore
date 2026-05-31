import argparse
import sys
from pathlib import Path

PYTORCH_ROOT = Path(__file__).resolve().parents[2]
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from recstore.unittest.ebc_baseline.multiprocess import (
    BARRIER_TIMEOUT_SECONDS,
    PROCESS_JOIN_TIMEOUT_SECONDS,
    ensure_spawn_start_method,
    generate_rank_batch,
    run_multiprocess_precision,
    worker,
)


ensure_spawn_start_method()


def main(args):
    run_multiprocess_precision(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiprocess precision test.")
    parser.add_argument("--num-embeddings", type=int, default=1000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--ps-host", type=str, default=None)
    parser.add_argument("--ps-port", type=int, default=None)

    main(parser.parse_args())
