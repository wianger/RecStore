import argparse
import sys
from pathlib import Path

import torch

PYTORCH_ROOT = Path(__file__).resolve().parents[2]
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from recstore.unittest.ebc_baseline.single_process import (
    LEARNING_RATE,
    NUM_TEST_ROUNDS,
    compare_tensors,
    generate_random_batch,
    get_eb_configs,
    run_precision,
)


def main(args):
    return run_precision(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-round precision test for RecStore EBC."
    )
    parser.add_argument("--num-embeddings", type=int, default=1000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=int(torch.rand(1)[0]))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--ps-host", type=str)
    parser.add_argument("--ps-port", type=int)

    main(parser.parse_args())
