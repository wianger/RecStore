#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "docs" / "git_path_contributors.py"
SPEC = importlib.util.spec_from_file_location("git_path_contributors", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SAMPLE_LOG = """commit abc123
author Alice
time 2024-01-02 03:04:05 +0000

10\t2\tsrc/a.py
3\t1\tsrc/b.py

commit def456
author Bob
time 2024-02-03 04:05:06 +0000

-\t-\tbinary/model.bin
4\t0\tsrc/a.py

"""


class GitPathContributorsTest(unittest.TestCase):
    def test_parse_commits_from_numstat_log(self):
        commits = MODULE.parse_commits(SAMPLE_LOG)

        self.assertEqual(2, len(commits))
        self.assertEqual("abc123", commits[0]["commit"])
        self.assertEqual("Alice", commits[0]["author"])
        self.assertEqual("2024-01-02 03:04:05 +0000", commits[0]["time"])
        self.assertEqual(13, commits[0]["added_lines"])
        self.assertEqual(3, commits[0]["deleted_lines"])
        self.assertEqual(["src/a.py", "src/b.py"], commits[0]["files"])

        self.assertEqual("def456", commits[1]["commit"])
        self.assertEqual(4, commits[1]["added_lines"])
        self.assertEqual(0, commits[1]["deleted_lines"])
        self.assertEqual(["binary/model.bin", "src/a.py"], commits[1]["files"])

    def test_group_commits_by_author_and_sort(self):
        commits = MODULE.parse_commits(SAMPLE_LOG)
        authors = MODULE.group_commits_by_author(commits)

        self.assertEqual(["Alice", "Bob"], [author["author"] for author in authors])
        self.assertEqual(16, authors[0]["total_lines"])
        self.assertEqual(4, authors[1]["total_lines"])
        self.assertEqual(["abc123"], [commit["commit"] for commit in authors[0]["commits"]])

    def test_render_report_contains_commit_details(self):
        commits = MODULE.parse_commits(SAMPLE_LOG)
        authors = MODULE.group_commits_by_author(commits)
        report = MODULE.render_report("src", authors)

        self.assertIn("Path: src", report)
        self.assertIn("Author: Alice", report)
        self.assertIn("Commit: abc123", report)
        self.assertIn("Time: 2024-01-02 03:04:05 +0000", report)
        self.assertIn("Lines: +13 -3", report)
        self.assertIn("Files: src/a.py, src/b.py", report)


if __name__ == "__main__":
    unittest.main()
