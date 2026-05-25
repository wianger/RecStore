import sys
import unittest
from pathlib import Path

import torch

_PYTHON_ROOT = Path(__file__).resolve().parents[3]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from pytorch.torchrec_kv.EmbeddingBag import RecStoreEmbeddingBagCollection
from model_zoo.rs_demo.data.dlrm_source import build_sparse_features


class _FakeOps:
    def __init__(self):
        self._store = {}  # id(int)->tensor(1,D)
        self._next_handle = 1
        self._prefetch_buf = {}

    def emb_write(self, keys: torch.Tensor, values: torch.Tensor):
        keys = keys.to(torch.int64).cpu().contiguous()
        values = values.to(torch.float32).cpu().contiguous()
        assert values.dim() == 2
        for i in range(keys.numel()):
            self._store[int(keys[i].item())] = values[i:i+1].clone()

    def emb_read(self, keys: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        keys = keys.to(torch.int64).cpu().contiguous()
        out = torch.zeros((keys.numel(), int(embedding_dim)), dtype=torch.float32)
        for i in range(keys.numel()):
            kid = int(keys[i].item())
            if kid in self._store:
                row = self._store[kid]
                if row.size(1) != embedding_dim:
                    # simple reshape/pad-truncate to requested dim
                    if row.size(1) > embedding_dim:
                        out[i] = row[0, :embedding_dim]
                    else:
                        out[i, :row.size(1)] = row[0]
                else:
                    out[i] = row[0]
            else:
                # keep zeros for missing
                pass
        return out

    def emb_prefetch(self, keys: torch.Tensor) -> int:
        # For testing, precompute the result with default dim 4; the real wait will pass dim anyway.
        keys = keys.to(torch.int64).cpu().contiguous()
        handle = self._next_handle
        self._next_handle += 1
        self._prefetch_buf[handle] = keys.clone()
        return handle

    def emb_wait_result(self, handle: int, embedding_dim: int) -> torch.Tensor:
        keys = self._prefetch_buf.pop(int(handle))
        return self.emb_read(keys, embedding_dim)


class _FakeKVClient:
    def __init__(self, ops: _FakeOps) -> None:
        self.ops = ops
        self._tensor_meta = {}
        self.prefill_calls = []
        self.local_lookup_calls = []
        self.use_prefill_values_for_local_lookup = False

    def init_data(self, name, shape, dtype, base_offset: int = 0):
        self._tensor_meta[name] = {
            "shape": tuple(shape),
            "dtype": dtype,
            "base_offset": int(base_offset),
        }

    def pull(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        return self.ops.emb_read(ids, embedding_dim)

    def local_lookup_flat(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        embedding_dim = int(self._tensor_meta[name]["shape"][1])
        self.local_lookup_calls.append((name, ids.clone()))
        if self.use_prefill_values_for_local_lookup and self.prefill_calls:
            prefill_name, prefill_ids, prefill_values = self.prefill_calls[-1]
            self.assert_name = prefill_name
            index = {int(v.item()): i for i, v in enumerate(prefill_ids.cpu())}
            rows = [prefill_values[index[int(v.item())]] for v in ids.cpu()]
            return torch.stack(rows, dim=0).to(ids.device)
        return self.ops.emb_read(ids, embedding_dim).to(ids.device)

    def prefetch(self, ids: torch.Tensor) -> int:
        return int(self.ops.emb_prefetch(ids))

    def wait_and_get(self, handle: int, embedding_dim: int, device=torch.device("cpu")) -> torch.Tensor:
        out = self.ops.emb_wait_result(handle, embedding_dim)
        if device.type == "cuda":
            out = out.to(device)
        return out

    def prefill_gpu_cache(self, name: str, ids: torch.Tensor, values: torch.Tensor) -> None:
        self.prefill_calls.append((name, ids.detach().clone(), values.detach().clone()))

    def is_shared_local_shm_table(self) -> bool:
        return False


class TestFusedPrefetch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _build_features(self):
        # Batch size = 2
        # f1: lengths [2,1] -> values 3 ids
        # f2: lengths [1,2] -> values 3 ids
        keys = ["f1", "f2"]
        values_f1 = torch.tensor([1, 2, 3], dtype=torch.int64)
        lengths_f1 = torch.tensor([2, 1], dtype=torch.int32)
        values_f2 = torch.tensor([0, 4, 2], dtype=torch.int64)
        lengths_f2 = torch.tensor([1, 2], dtype=torch.int32)
        values = torch.cat([values_f1, values_f2], dim=0)
        lengths = torch.cat([lengths_f1, lengths_f2], dim=0)
        kjt = build_sparse_features(keys=keys, values=values, lengths=lengths)
        return kjt

    def test_fused_prefetch_matches_sync(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        # mirror init_data writes
        for idx, cfg in enumerate(configs):
            base_offset = (idx << 30)
            n, d = cfg["num_embeddings"], cfg["embedding_dim"]
            keys = torch.arange(n, dtype=torch.int64) + base_offset
            vals = torch.zeros((n, d), dtype=torch.float32)
            fake.emb_write(keys, vals)

        # Build features after initialization; rely on zero initialization for both tables.
        features = self._build_features()

        # Sync path (no prefetch)
        out_sync = ebc(features).values().detach().clone()

        # Fused prefetch path
        ebc.issue_fused_prefetch(features)
        out_prefetch = ebc(features).values().detach().clone()

        self.assertTrue(torch.allclose(out_sync, out_prefetch), "Fused prefetch output must match sync output (zero-init case)")

        stats = ebc.report_prefetch_stats(reset=True)
        self.assertGreaterEqual(stats.get("batches_prefetched", 0), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for GPU cache prefill")
    def test_fused_prefetch_prefills_gpu_cache_before_local_lookup(self):
        configs = [
            dict(name="t0", embedding_dim=4, num_embeddings=16, feature_names=["f1"]),
            dict(name="t1", embedding_dim=4, num_embeddings=16, feature_names=["f2"]),
        ]
        fake = _FakeOps()
        fake_client = _FakeKVClient(fake)
        fake_client.use_prefill_values_for_local_lookup = True
        ebc = RecStoreEmbeddingBagCollection(
            configs,
            enable_fusion=True,
            fusion_k=30,
            kv_client=fake_client,
        )
        ebc.enable_single_node_distributed_fast_path = True
        ebc.single_node_distributed_mode = "single_node"
        fake_client.is_shared_local_shm_table = lambda: True
        for idx, cfg in enumerate(configs):
            base_offset = (idx << 30)
            n, d = cfg["num_embeddings"], cfg["embedding_dim"]
            keys = torch.arange(n, dtype=torch.int64) + base_offset
            vals = torch.arange(n * d, dtype=torch.float32).view(n, d) + idx * 1000
            fake.emb_write(keys, vals)

        features = self._build_features()
        values = features._values.to("cuda")
        lengths = features._lengths.to("cuda")
        features = build_sparse_features(features.keys(), values, lengths)
        ebc.issue_fused_prefetch(features)
        out = ebc(features)

        self.assertEqual(len(fake_client.prefill_calls), 1)
        self.assertEqual(len(fake_client.local_lookup_calls), 1)
        name, prefill_ids, prefill_values = fake_client.prefill_calls[0]
        self.assertEqual(name, "t0")
        self.assertEqual(prefill_ids.dtype, torch.int64)
        self.assertEqual(prefill_values.dtype, torch.float32)
        self.assertEqual(prefill_ids.device.type, prefill_values.device.type)
        looked_up_ids = fake_client.local_lookup_calls[0][1]
        self.assertTrue(set(looked_up_ids.cpu().tolist()).issubset(set(prefill_ids.cpu().tolist())))
        self.assertEqual(out.values().shape, (2, 8))
        perf = ebc.consume_perf_stats(reset=True)
        self.assertEqual(perf["planned_gpu_cache_prefill_batches"], 1.0)
        self.assertGreater(perf["planned_gpu_cache_prefill_ids"], 0.0)


if __name__ == "__main__":
    unittest.main()
