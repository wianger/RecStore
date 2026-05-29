import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class TensorflowClientPathTest(unittest.TestCase):
    def test_default_library_path_matches_cmake_output_name(self):
        client_path = Path(__file__).resolve().with_name("client.py")
        spec = importlib.util.spec_from_file_location("tf_client_for_path_test", client_path)
        client_module = importlib.util.module_from_spec(spec)
        fake_tf = types.SimpleNamespace(
            Tensor=object,
            Operation=object,
            uint64="uint64",
            float32="float32",
            load_op_library=mock.Mock(),
            convert_to_tensor=lambda value, dtype=None: value,
        )

        with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
            spec.loader.exec_module(client_module)

        expected = (
            Path(__file__).resolve().parents[4]
            / "build"
            / "lib"
            / "lib_recstore_tf_ops.so"
        )

        with mock.patch.object(client_module.os.path, "exists", return_value=True):
            client_module.RecstoreClient()

        client_module.tf.load_op_library.assert_called_once_with(str(expected))


if __name__ == "__main__":
    unittest.main()
