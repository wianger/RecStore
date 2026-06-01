from __future__ import annotations

import sys
from pathlib import Path


def ensure_legacy_test_script_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    legacy_dir = repo_root / "src" / "test" / "scripts"
    legacy_dir_str = str(legacy_dir)
    if legacy_dir_str not in sys.path:
        sys.path.insert(0, legacy_dir_str)
    return legacy_dir
