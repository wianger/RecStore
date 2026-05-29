import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class TensorflowDlrmClientTest(unittest.TestCase):
    def setUp(self):
        self.module_name = "model_zoo.tensorflow_dlrm.client"
        sys.modules.pop(self.module_name, None)
        fake_tf = types.SimpleNamespace(
            Tensor=object,
            Operation=object,
            uint64="uint64",
            float32="float32",
            load_op_library=mock.Mock(),
            convert_to_tensor=lambda value, dtype=None: value,
        )
        self.tf_patch = mock.patch.dict(sys.modules, {"tensorflow": fake_tf})
        self.tf_patch.start()

    def tearDown(self):
        self.tf_patch.stop()
        sys.modules.pop(self.module_name, None)

    def test_default_library_path_is_repo_relative(self):
        client_module = importlib.import_module(self.module_name)
        expected = (
            Path(__file__).resolve().parents[3]
            / "build"
            / "lib"
            / "lib_recstore_tf_ops.so"
        )

        with mock.patch.object(client_module.os.path, "exists", return_value=True):
            client_module.RecstoreClient()

        client_module.tf.load_op_library.assert_called_once_with(str(expected))


if __name__ == "__main__":
    unittest.main()
