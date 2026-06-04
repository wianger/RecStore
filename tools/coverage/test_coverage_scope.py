import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "coverage"))

from coverage_scope import extract_scope_globs, glob_to_gcovr_filter


class CoverageScopeTest(unittest.TestCase):
    def test_extracts_only_coverage_owned_label_prefixes(self):
        labeler = """
Compute/PyTorch:
- changed-files:
  - any-glob-to-any-file:
    - 'src/framework/pytorch/**'
    - 'src/python/pytorch/**'

Network/gRPC:
- changed-files:
  - any-glob-to-any-file:
    - 'src/ps/grpc/**'

Storage/Engine:
- changed-files:
  - any-glob-to-any-file:
    - 'src/storage/**'

Optimizer:
- changed-files:
  - any-glob-to-any-file:
    - 'src/optimizer/**'

Testing:
- changed-files:
  - any-glob-to-any-file:
    - 'src/test/**'

Documentation:
- changed-files:
  - any-glob-to-any-file:
    - 'docs/**'
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeler.yml"
            path.write_text(labeler, encoding="utf-8")

            self.assertEqual(
                extract_scope_globs(path),
                [
                    "src/framework/pytorch/**",
                    "src/python/pytorch/**",
                    "src/ps/grpc/**",
                    "src/storage/**",
                    "src/optimizer/**",
                ],
            )

    def test_gcovr_filter_is_root_relative(self):
        self.assertEqual(glob_to_gcovr_filter("src/framework/op.cc"), "src/framework/op\\.cc")
        self.assertEqual(glob_to_gcovr_filter("src/framework/pytorch/**"), "src/framework/pytorch/.*")


if __name__ == "__main__":
    unittest.main()
