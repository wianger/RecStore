#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Optional


DEFAULT_RECSTORE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "recstore_config.json"
RECSTORE_CONFIG_FILENAME = "recstore_config.json"


def _existing_regular_file(path: Path) -> Optional[Path]:
    return path.resolve() if path.is_file() else None


def find_recstore_config_path(start_dir: Path | str | None = None) -> Optional[Path]:
    env_config = os.environ.get("RECSTORE_CONFIG")
    if env_config:
        config_path = _existing_regular_file(Path(env_config))
        if config_path is not None:
            return config_path

    current = Path(start_dir) if start_dir is not None else Path.cwd()
    current = current.resolve()
    while True:
        config_path = _existing_regular_file(current / RECSTORE_CONFIG_FILENAME)
        if config_path is not None:
            return config_path
        if current.parent == current:
            break
        current = current.parent

    config_path = _existing_regular_file(DEFAULT_RECSTORE_CONFIG_PATH)
    if config_path is not None:
        return config_path
    return None


def resolve_recstore_config_path(start_dir: Path | str | None = None) -> Path:
    config_path = find_recstore_config_path(start_dir)
    if config_path is None:
        start = Path(start_dir) if start_dir is not None else Path.cwd()
        raise FileNotFoundError(
            f"{RECSTORE_CONFIG_FILENAME} not found at {DEFAULT_RECSTORE_CONFIG_PATH} "
            f"or in any parent directory of {start}"
        )
    return config_path
