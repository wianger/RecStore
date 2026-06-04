import os
import shutil
import subprocess
import tarfile
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_SCRIPT = REPO_ROOT / "tools" / "packaging" / "pack_artifact.sh"


class PackArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.fakebin = self.root / "fakebin"
        self.fakebin.mkdir()
        self.deps = self.root / "deps"
        self.deps.mkdir()
        self.artifact = self.root / "ps_server"
        self.artifact.write_text("fake-elf", encoding="utf-8")
        os.chmod(self.artifact, 0o755)

        self._write_executable(
            "file",
            """#!/bin/sh
target="$2"
case "$target" in
  *ps_server) echo "ELF 64-bit LSB executable" ;;
  *) echo "ELF 64-bit LSB shared object" ;;
esac
""",
        )
        self._write_executable(
            "lddtree",
            f"""#!/bin/sh
cat <<'EOF'
{self.deps / "ld-linux-x86-64.so.2"}
{self.deps / "libc.so.6"}
{self.deps / "libcustom_dep.so"}
EOF
""",
        )
        self._write_executable(
            "patchelf",
            """#!/bin/sh
exit 0
""",
        )

        for name in ("ld-linux-x86-64.so.2", "libc.so.6", "libcustom_dep.so"):
            (self.deps / name).write_text(name, encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_executable(self, name: str, content: str) -> None:
        path = self.fakebin / name
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        os.chmod(path, 0o755)

    def test_excludes_host_runtime_libs_but_keeps_custom_deps(self) -> None:
        output_tar = self.root / "packed.tar.gz"
        env = os.environ.copy()
        env["PATH"] = f"{self.fakebin}:{env['PATH']}"

        completed = subprocess.run(
            ["bash", str(PACK_SCRIPT), str(output_tar), str(self.artifact)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

        with tarfile.open(output_tar, "r:gz") as archive:
            names = set(archive.getnames())

        self.assertIn("package/bin/ps_server", names)
        self.assertIn("package/deps/lib/libcustom_dep.so", names)
        self.assertNotIn("package/deps/lib/libc.so.6", names)
        self.assertNotIn("package/deps/lib/ld-linux-x86-64.so.2", names)


if __name__ == "__main__":
    unittest.main()
