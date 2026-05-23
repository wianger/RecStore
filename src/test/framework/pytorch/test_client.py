import os
import sys
import unittest

import torch

from client import RecstoreClient

_server_runner = None
TEST_PHASE = os.environ.get("RECSTORE_CLIENT_TEST_PHASE", "full").lower()
RDMA_MEMCACHED_MODE = os.environ.get("RECSTORE_USE_LOCAL_MEMCACHED", "never").lower()
RDMA_GFLAG_PREFIXES = (
    "--global_id=",
    "--num_server_processes=",
    "--num_client_processes=",
    "--value_size=",
    "--max_kv_num_per_request=",
)


def log(message):
    print(message, flush=True)


def ensure_test_scripts_path():
    test_scripts_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../scripts")
    )
    if test_scripts_path not in sys.path:
        sys.path.insert(0, test_scripts_path)


def maybe_skip_for_missing_rdma():
    ensure_test_scripts_path()

    from ps_server_helpers import (
        RDMA_SKIP_EXIT_CODE,
        get_backend_type,
        get_rdma_skip_reason,
    )

    if get_backend_type() != "RDMA":
        return

    skip_reason = get_rdma_skip_reason()
    if skip_reason:
        log(f"[op-rdma][skip] {skip_reason}")
        raise SystemExit(RDMA_SKIP_EXIT_CODE)


def maybe_reexec_for_rdma_flags():
    maybe_skip_for_missing_rdma()

    if os.environ.get("RECSTORE_RDMA_BOOTSTRAPPED") == "1":
        log("[op-rdma] RDMA bootstrap already applied")
        return

    ensure_test_scripts_path()

    from ps_server_helpers import get_backend_type, get_rdma_runner_config

    if get_backend_type() != "RDMA":
        return

    rdma_config = get_rdma_runner_config()
    extra_flags = [
        f"--global_id={rdma_config['num_servers']}",
        f"--num_server_processes={rdma_config['num_servers']}",
        "--num_client_processes=1",
        f"--value_size={rdma_config['value_size']}",
        f"--max_kv_num_per_request={rdma_config['max_kv_num_per_request']}",
    ]

    new_env = os.environ.copy()
    new_env["RECSTORE_RDMA_BOOTSTRAPPED"] = "1"
    log(f"[op-rdma] reexec with flags: {' '.join(extra_flags)}")
    os.execve(
        sys.executable,
        [sys.executable, os.path.abspath(__file__), *sys.argv[1:], *extra_flags],
        new_env,
    )


def unittest_argv_without_cpp_gflags():
    return [sys.argv[0]] + [
        arg
        for arg in sys.argv[2:]
        if not any(arg.startswith(prefix) for prefix in RDMA_GFLAG_PREFIXES)
    ]


def start_server_if_needed():
    global _server_runner

    maybe_skip_for_missing_rdma()
    ensure_test_scripts_path()

    from ps_server_helpers import (
        get_backend_type,
        get_rdma_runner_config,
        get_server_config,
        should_skip_server_start,
    )

    backend = get_backend_type()
    if backend == "RDMA":
        if RDMA_MEMCACHED_MODE not in ("always", "auto", "never"):
            raise RuntimeError(
                "RECSTORE_USE_LOCAL_MEMCACHED must be one of: "
                "always, auto, never"
            )
        rdma_config = get_rdma_runner_config()
        log(f"\n{'=' * 70}")
        log("Starting PetPS Server for RDMA Client Tests")
        log(f"Config path: {os.environ.get('RECSTORE_CONFIG')}")
        log(f"Memcached mode: {RDMA_MEMCACHED_MODE}")
        log(f"{'=' * 70}\n")

        from petps_cluster_runner import PetPSClusterRunner

        _server_runner = PetPSClusterRunner(
            config_path=os.environ["RECSTORE_CONFIG"],
            num_servers=rdma_config["num_servers"],
            num_clients=1,
            thread_num=1,
            value_size=rdma_config["value_size"],
            max_kv_num_per_request=rdma_config["max_kv_num_per_request"],
            timeout=15,
            use_local_memcached=RDMA_MEMCACHED_MODE,
        )
        _server_runner.start()
        os.environ["RECSTORE_MEMCACHED_HOST"] = _server_runner.memcached_host
        os.environ["RECSTORE_MEMCACHED_PORT"] = str(_server_runner.memcached_port)
        os.environ["RECSTORE_MEMCACHED_TEXT_PROTOCOL"] = "1"
        if getattr(_server_runner, "memcached_namespace", ""):
            os.environ["RECSTORE_MEMCACHED_NAMESPACE"] = _server_runner.memcached_namespace
        log(
            "[op-rdma] effective memcached endpoint "
            f"{_server_runner.memcached_host}:{_server_runner.memcached_port}"
        )
        return

    skip_server, reason = should_skip_server_start()
    if skip_server:
        print(f"\n[{reason}] Running tests without starting ps_server\n")
        return

    config = get_server_config()

    print(f"\n{'=' * 70}")
    print("Starting PS Server for Client Tests")
    print(f"Server path: {config['server_path']}")
    print(f"{'=' * 70}\n")

    from ps_server_runner import PSServerRunner

    _server_runner = PSServerRunner(
        server_path=config["server_path"],
        config_path=config["config_path"],
        log_dir=config["log_dir"],
        timeout=config["timeout"],
        num_shards=config["num_shards"],
        verbose=True,
    )

    if not _server_runner.start():
        raise RuntimeError("Failed to start PS Server")


def stop_server():
    global _server_runner

    if _server_runner is not None:
        print(f"\n{'=' * 70}")
        print("Stopping PS Server")
        print(f"{'=' * 70}\n")
        _server_runner.stop()
        _server_runner = None


class TestRecstoreClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        maybe_reexec_for_rdma_flags()
        if len(sys.argv) < 2:
            raise ValueError("Usage: python test_client.py <path_to_custom_ops_library>")

        cls.library_path = sys.argv[1]
        print(f"Using custom ops library: {cls.library_path}")
        start_server_if_needed()
        cls.client = RecstoreClient(library_path=cls.library_path)
        cls.embedding_dim = 128
        cls.values_to_write = None

    @classmethod
    def tearDownClass(cls):
        stop_server()

    def test_00_init_and_read_write(self):
        print("\n--- Test 0: Init Embedding Table ---")
        ok = self.client.init_embedding_table("default", 10000, self.embedding_dim)
        self.assertTrue(ok, "init_embedding_table returned False")
        print("Init embedding table succeeded.")

        print("\n--- Test 1: Write and Read Verification ---")
        keys_to_write = torch.tensor([1001, 1002, 1003], dtype=torch.int64)
        values_to_write = torch.randn(3, self.embedding_dim, dtype=torch.float32)

        print(f"Writing embeddings for keys: {keys_to_write.tolist()}")
        self.client.emb_write(keys_to_write, values_to_write)
        print("Write call successful.")

        print(f"Reading embeddings for keys: {keys_to_write.tolist()}")
        read_values = self.client.emb_read(keys_to_write, self.embedding_dim)

        self.assertEqual(read_values.shape, values_to_write.shape, "Shape mismatch after read")
        self.assertTrue(torch.allclose(read_values, values_to_write), "Value mismatch after read")
        print("Read successful. Written values verified.")
        type(self).values_to_write = values_to_write

    def test_10_prefetch(self):
        if TEST_PHASE == "basic":
            self.skipTest("Skipping prefetch check in basic RDMA phase.")

        print("\n--- Test 2: Async Prefetch Read ---")
        prefetch_keys = torch.tensor([2001, 2002, 2003, 2004], dtype=torch.int64)
        prefetch_vals = torch.randn(4, self.embedding_dim, dtype=torch.float32)
        self.client.emb_write(prefetch_keys, prefetch_vals)

        pid = self.client.emb_prefetch(prefetch_keys)
        print(f"Issued prefetch id: {pid}")
        prefetched = self.client.emb_wait_result(pid, self.embedding_dim)

        print(f"Prefetched embeddings for keys: {prefetch_keys.tolist()}")
        print(f"Prefetch shape: {prefetched.shape}, expected shape: {(4, self.embedding_dim)}")
        print(f"Prefetched values(first 3): {prefetched.tolist()[:3]}")
        print(f"Expected values(first 3): {prefetch_vals.tolist()[:3]}")
        self.assertEqual(prefetched.shape, (4, self.embedding_dim), "Prefetch result shape mismatch")
        self.assertTrue(torch.allclose(prefetched, prefetch_vals), "Prefetch values mismatch")
        print("Async prefetch successful and values verified.")

    def test_20_update(self):
        if TEST_PHASE == "basic":
            self.skipTest("Skipping update check in basic RDMA phase.")

        self.assertIsNotNone(
            type(self).values_to_write,
            "Write/read test must run before update verification.",
        )

        print("\n--- Test 3: Table-aware Update (smoke) ---")
        update_keys = torch.tensor([1001, 1002], dtype=torch.int64)
        grads = torch.ones(2, self.embedding_dim, dtype=torch.float32)

        print("Reading values before update...")
        values_before_update = self.client.emb_read(update_keys, self.embedding_dim)
        print(f"Values before update (first 5 dims): {values_before_update[:, :5].tolist()}")
        print(f"Expected initial values (first 5 dims): {self.values_to_write[:2, :5].tolist()}")

        if not torch.allclose(values_before_update, self.values_to_write[:2]):
            print("WARNING: Values before update don't match values_to_write from Test 1!")
            print(f"  Max diff: {(values_before_update - self.values_to_write[:2]).abs().max()}")

        print(f"Updating embeddings for keys: {update_keys.tolist()}")
        self.client.emb_update_table("default", update_keys, grads)
        print("emb_update_table call succeeded.")

        lr = 0.01
        print(f"Reading updated values (backend lr={lr})...")
        updated_values = self.client.emb_read(update_keys, self.embedding_dim)
        expected_updated = self.values_to_write[:2] - (lr * grads)

        print(f"Updated values (first 5 dims): {updated_values[:, :5].tolist()}")
        print(f"Expected updated (first 5 dims): {expected_updated[:, :5].tolist()}")
        print(f"Difference (first 5 dims): {(updated_values - expected_updated)[:, :5].tolist()}")
        print(f"Max absolute difference: {(updated_values - expected_updated).abs().max():.6f}")
        print(
            "Max relative difference: "
            f"{((updated_values - expected_updated).abs() / (expected_updated.abs() + 1e-8)).max():.6f}"
        )

        if torch.allclose(updated_values, expected_updated, rtol=1e-4, atol=1e-6):
            print("Table-aware update verified successfully (with relaxed tolerance).")
            return
        if torch.allclose(updated_values, expected_updated):
            print("Table-aware update verified successfully.")
            return

        print("Table-aware update values mismatch!")
        print(f"  Expected: param = param - {lr} * grad")
        print(f"  Are all gradients 1.0? {torch.allclose(grads, torch.ones_like(grads))}")
        self.fail("Table-aware update values mismatch")


if __name__ == "__main__":
    unittest.main(argv=unittest_argv_without_cpp_gflags())
