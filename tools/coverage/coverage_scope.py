#!/usr/bin/env python3
import argparse
import fnmatch
import shlex
from pathlib import Path
from typing import Iterable, List


COVERAGE_LABEL_PREFIXES = ("Compute", "Network", "Storage", "Optimizer")


def _load_labeler(labeler_path: Path):
    try:
        import yaml

        with labeler_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        return _load_labeler_without_yaml(labeler_path)


def _load_labeler_without_yaml(labeler_path: Path):
    data = {}
    current_label = None
    for raw_line in labeler_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        if raw_line and not raw_line.startswith((" ", "-")) and raw_line.rstrip().endswith(":"):
            current_label = raw_line.rstrip()[:-1]
            data[current_label] = [{"changed-files": [{"any-glob-to-any-file": []}]}]
            continue

        stripped = raw_line.strip()
        if current_label and stripped.startswith("- "):
            value = stripped[2:].strip()
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            if value and not value.endswith(":"):
                data[current_label][0]["changed-files"][0]["any-glob-to-any-file"].append(value)
    return data


def extract_scope_globs(labeler_path: Path) -> List[str]:
    data = _load_labeler(labeler_path)

    globs: List[str] = []
    for label, rules in data.items():
        if not str(label).startswith(COVERAGE_LABEL_PREFIXES):
            continue
        for rule in rules or []:
            changed_files = rule.get("changed-files", [])
            for item in changed_files:
                globs.extend(item.get("any-glob-to-any-file", []) or [])

    deduped: List[str] = []
    seen = set()
    for pattern in globs:
        if pattern not in seen:
            deduped.append(pattern)
            seen.add(pattern)
    return deduped


def glob_to_gcovr_filter(pattern: str) -> str:
    regex = fnmatch.translate(pattern)
    if regex.startswith("(?s:") and regex.endswith(")\\Z"):
        regex = regex[4:-3]
    regex = regex.replace("\\Z", "$")
    return regex


def glob_to_coverage_include(pattern: str) -> str:
    return pattern if pattern.startswith("*") else f"*/{pattern}"


def emit_shell_array(name: str, values: Iterable[str]) -> None:
    print(f"{name}=(")
    for value in values:
        print(f"  {shlex.quote(value)}")
    print(")")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeler", type=Path, default=Path(".github/labeler.yml"))
    parser.add_argument("--format", choices=("gcovr", "coverage"), required=True)
    args = parser.parse_args()

    globs = extract_scope_globs(args.labeler)
    if args.format == "gcovr":
        emit_shell_array(
            "COVERAGE_SCOPE_ARGS",
            (arg for pattern in globs for arg in ("--filter", glob_to_gcovr_filter(pattern))),
        )
    else:
        include_patterns = ",".join(glob_to_coverage_include(pattern) for pattern in globs)
        emit_shell_array("COVERAGE_SCOPE_ARGS", ("--include", include_patterns))


if __name__ == "__main__":
    main()
