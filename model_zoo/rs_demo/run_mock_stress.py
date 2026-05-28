#!/usr/bin/env python3
"""Backward-compatible entrypoint."""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = str(_THIS_DIR.parents[1])
_PKG_PARENT = str(_THIS_DIR.parent)
for _path in (_REPO_ROOT, _PKG_PARENT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rs_demo.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
