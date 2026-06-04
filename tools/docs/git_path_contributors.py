#!/usr/bin/env python3

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show contributors and commit stats for a git path."
    )
    parser.add_argument("path", help="Repository-relative or absolute path to inspect")
    return parser.parse_args()


def run_git_command(args, cwd):
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout


def find_repo_root():
    return Path(
        run_git_command(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).strip()
    )


def normalize_target_path(repo_root, target_path):
    candidate = Path(target_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (repo_root / candidate).resolve()

    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside repository: {target_path}") from exc


def load_git_history(repo_root, relative_path):
    return run_git_command(
        [
            "git",
            "log",
            "--numstat",
            "--date=iso-strict",
            "--pretty=format:commit %H%nauthor %an%ntime %ai%n",
            "--",
            relative_path,
        ],
        cwd=repo_root,
    )


def safe_int(value):
    return int(value) if value.isdigit() else 0


def parse_commits(log_output):
    commits = []
    current = None

    for raw_line in log_output.splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            continue

        if line.startswith("commit "):
            if current is not None:
                commits.append(current)
            current = {
                "commit": line.split(" ", 1)[1],
                "author": "",
                "time": "",
                "added_lines": 0,
                "deleted_lines": 0,
                "files": [],
            }
            continue

        if current is None:
            continue

        if line.startswith("author "):
            current["author"] = line.split(" ", 1)[1]
            continue

        if line.startswith("time "):
            current["time"] = line.split(" ", 1)[1]
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            continue

        added, deleted, path = parts
        current["added_lines"] += safe_int(added)
        current["deleted_lines"] += safe_int(deleted)
        current["files"].append(path)

    if current is not None:
        commits.append(current)

    return commits


def group_commits_by_author(commits):
    grouped = defaultdict(list)
    for commit in commits:
        grouped[commit["author"]].append(commit)

    authors = []
    for author, author_commits in grouped.items():
        sorted_commits = sorted(author_commits, key=lambda item: item["time"], reverse=True)
        total_added = sum(item["added_lines"] for item in sorted_commits)
        total_deleted = sum(item["deleted_lines"] for item in sorted_commits)
        authors.append(
            {
                "author": author,
                "commit_count": len(sorted_commits),
                "total_added_lines": total_added,
                "total_deleted_lines": total_deleted,
                "total_lines": total_added + total_deleted,
                "commits": sorted_commits,
            }
        )

    return sorted(
        authors,
        key=lambda item: (item["total_lines"], item["commit_count"], item["author"]),
        reverse=True,
    )


def render_report(target_path, authors):
    lines = [f"Path: {target_path}", f"Authors: {len(authors)}", ""]

    for author in authors:
        lines.append(
            "Author: {author} | Commits: {commit_count} | Lines: +{added} -{deleted} | Total: {total}".format(
                author=author["author"],
                commit_count=author["commit_count"],
                added=author["total_added_lines"],
                deleted=author["total_deleted_lines"],
                total=author["total_lines"],
            )
        )
        for commit in author["commits"]:
            lines.append(
                "  Commit: {commit} | Time: {time} | Lines: +{added} -{deleted} | Files: {files}".format(
                    commit=commit["commit"],
                    time=commit["time"],
                    added=commit["added_lines"],
                    deleted=commit["deleted_lines"],
                    files=", ".join(commit["files"]),
                )
            )
        lines.append("")

    return "\n".join(lines).rstrip()


def main():
    args = parse_args()

    try:
        repo_root = find_repo_root()
        relative_path = normalize_target_path(repo_root, args.path)
    except (subprocess.CalledProcessError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        log_output = load_git_history(repo_root, relative_path)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or str(exc)
        print(message, file=sys.stderr)
        return 1

    commits = parse_commits(log_output)
    authors = group_commits_by_author(commits)
    print(render_report(relative_path, authors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
