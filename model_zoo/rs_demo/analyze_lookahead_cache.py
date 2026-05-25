from __future__ import annotations

import argparse
import csv
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from model_zoo.rs_demo.data.dlrm_source import build_train_dataloader, inject_project_paths


@dataclass(frozen=True)
class BatchIds:
    step: int
    ids: set[int]


def _parse_int_list(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    if not result:
        raise ValueError("expected at least one integer")
    return result


def _fused_unique_ids_from_sparse_batch(
    sparse_batch: torch.Tensor,
    *,
    fuse_k: int,
) -> set[int]:
    sparse = sparse_batch.to(torch.int64).cpu()
    chunks = []
    for table_idx in range(sparse.shape[1]):
        chunks.append(sparse[:, table_idx] + (table_idx << int(fuse_k)))
    if not chunks:
        return set()
    return set(int(v) for v in torch.unique(torch.cat(chunks)).tolist())


def load_batch_ids(
    *,
    repo_root: Path,
    data_dir: str,
    train_ratio: float,
    num_embeddings: int,
    batch_size: int,
    steps: int,
    seed: int,
    fuse_k: int,
) -> list[BatchIds]:
    inject_project_paths(repo_root)
    _, dataloader = build_train_dataloader(
        repo_root=repo_root,
        data_dir_rel=data_dir,
        train_ratio=train_ratio,
        num_embeddings=num_embeddings,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    data_iter = iter(dataloader)
    batches: list[BatchIds] = []
    for step in range(int(steps)):
        try:
            _, sparse_batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            _, sparse_batch, _ = next(data_iter)
        batches.append(
            BatchIds(
                step=step,
                ids=_fused_unique_ids_from_sparse_batch(sparse_batch, fuse_k=fuse_k),
            )
        )
    return batches


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_reuse(
    batches: list[BatchIds],
    *,
    depths: Iterable[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for depth in depths:
        depth = int(depth)
        for idx, batch in enumerate(batches):
            current = batch.ids
            future_batches = batches[idx + 1 : idx + depth + 1]
            future_union: set[int] = set()
            future_counts: Counter[int] = Counter()
            last_distance: dict[int, int] = {}
            for distance, future in enumerate(future_batches, start=1):
                future_union.update(future.ids)
                for emb_id in future.ids:
                    future_counts[emb_id] += 1
                    last_distance[emb_id] = distance

            reused = current & future_union
            ttl_distances = [last_distance[emb_id] for emb_id in reused]
            next_batch_reused = current & (future_batches[0].ids if future_batches else set())
            future_unique = len(future_union)
            rows.append(
                {
                    "step": batch.step,
                    "depth": depth,
                    "current_unique_ids": len(current),
                    "future_unique_ids": future_unique,
                    "reused_in_window_ids": len(reused),
                    "reused_in_window_ratio": len(reused) / len(current) if current else 0.0,
                    "next_batch_reused_ids": len(next_batch_reused),
                    "next_batch_reused_ratio": len(next_batch_reused) / len(current) if current else 0.0,
                    "prefetch_only_candidate_ids": len(current) - len(reused),
                    "avg_ttl_distance": sum(ttl_distances) / len(ttl_distances) if ttl_distances else 0.0,
                    "max_ttl_distance": max(ttl_distances) if ttl_distances else 0,
                    "future_occurrences_of_reused_ids": sum(future_counts[emb_id] for emb_id in reused),
                }
            )
    return rows


class LruCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.items: OrderedDict[int, None] = OrderedDict()

    def __contains__(self, key: int) -> bool:
        return key in self.items

    def touch(self, key: int) -> None:
        if self.capacity <= 0:
            return
        if key in self.items:
            self.items.move_to_end(key)
        else:
            self.items[key] = None
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)

    def discard(self, key: int) -> None:
        self.items.pop(key, None)

    def __len__(self) -> int:
        return len(self.items)


def simulate_lru(
    batches: list[BatchIds],
    *,
    capacity: int,
    bypass_min_rows: int | None = None,
    low_hit_ratio: float = 0.05,
) -> dict[str, object]:
    cache = LruCache(capacity)
    total_hits = 0
    total_requests = 0
    total_misses = 0
    cache_query_requests = 0
    bypassed_batches = 0
    queried_batches = 0
    bypass_enabled = False
    for batch in batches:
        ids = batch.ids
        total_requests += len(ids)
        if bypass_min_rows is not None and bypass_enabled and len(ids) >= bypass_min_rows:
            total_misses += len(ids)
            bypassed_batches += 1
            continue
        hits = sum(1 for emb_id in ids if emb_id in cache)
        cache_query_requests += len(ids)
        misses = len(ids) - hits
        total_hits += hits
        total_misses += misses
        queried_batches += 1
        for emb_id in ids:
            cache.touch(emb_id)
        if (
            bypass_min_rows is not None
            and len(ids) >= bypass_min_rows
            and len(ids) > 0
            and hits / len(ids) < low_hit_ratio
        ):
            bypass_enabled = True
            cache = LruCache(capacity)
    return {
        "policy": "lru_bypass" if bypass_min_rows is not None else "lru_all_insert",
        "capacity": capacity,
        "depth": 0,
        "requests": total_requests,
        "cache_query_requests": cache_query_requests,
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total_requests if total_requests else 0.0,
        "queried_batches": queried_batches,
        "bypassed_batches": bypassed_batches,
        "prefetch_ids": total_misses,
        "cache_insert_ids": total_misses,
        "evicted_ids": 0,
        "final_cache_size": len(cache),
    }


def simulate_hot_static(
    batches: list[BatchIds],
    *,
    capacity: int,
) -> dict[str, object]:
    counts: Counter[int] = Counter()
    for batch in batches:
        counts.update(batch.ids)
    hot = set(emb_id for emb_id, _ in counts.most_common(int(capacity)))
    requests = sum(len(batch.ids) for batch in batches)
    hits = sum(len(batch.ids & hot) for batch in batches)
    return {
        "policy": "hot_static_oracle",
        "capacity": capacity,
        "depth": 0,
        "requests": requests,
        "cache_query_requests": requests,
        "hits": hits,
        "misses": requests - hits,
        "hit_rate": hits / requests if requests else 0.0,
        "queried_batches": len(batches),
        "bypassed_batches": 0,
        "prefetch_ids": requests - hits,
        "cache_insert_ids": min(capacity, len(hot)),
        "evicted_ids": 0,
        "final_cache_size": len(hot),
    }


def simulate_lookahead_ttl(
    batches: list[BatchIds],
    *,
    capacity: int,
    depth: int,
) -> dict[str, object]:
    cache: dict[int, int] = {}
    evicted = 0
    hits = 0
    requests = 0
    prefetch_ids = 0
    cache_insert_ids = 0
    for idx, batch in enumerate(batches):
        expired = [emb_id for emb_id, ttl in cache.items() if ttl < idx]
        for emb_id in expired:
            del cache[emb_id]
            evicted += 1

        current = batch.ids
        future_batches = batches[idx + 1 : idx + int(depth) + 1]
        last_use: dict[int, int] = {}
        for offset, future in enumerate(future_batches, start=1):
            for emb_id in future.ids:
                last_use[emb_id] = idx + offset
        reusable = current & set(last_use)
        cache_hits = current & set(cache)
        hits += len(cache_hits)
        requests += len(current)

        for emb_id in reusable:
            ttl = last_use[emb_id]
            if emb_id not in cache:
                if len(cache) >= capacity:
                    # Evict the item with the nearest expiration. This is conservative:
                    # Bagpipe avoids this by shrinking lookahead or sizing the cache.
                    victim = min(cache.items(), key=lambda item: item[1])[0]
                    del cache[victim]
                    evicted += 1
                if len(cache) < capacity:
                    cache[emb_id] = ttl
                    cache_insert_ids += 1
            else:
                cache[emb_id] = max(cache[emb_id], ttl)

        prefetch_ids += len(current - cache_hits)

    misses = requests - hits
    return {
        "policy": "lookahead_ttl",
        "capacity": capacity,
        "depth": depth,
        "requests": requests,
        "cache_query_requests": requests,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / requests if requests else 0.0,
        "queried_batches": len(batches),
        "bypassed_batches": 0,
        "prefetch_ids": prefetch_ids,
        "cache_insert_ids": cache_insert_ids,
        "evicted_ids": evicted,
        "final_cache_size": len(cache),
    }


def summarize_rows(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    groups: dict[object, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda item: int(item[0])):
        numeric_keys = [
            name
            for name, value in items[0].items()
            if name != group_key and isinstance(value, (int, float))
        ]
        summary: dict[str, object] = {group_key: key, "rows": len(items)}
        for name in numeric_keys:
            vals = [float(item[name]) for item in items]
            summary[f"{name}_mean"] = sum(vals) / len(vals)
            summary[f"{name}_max"] = max(vals)
        out.append(summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze lookahead reuse and cache policies for rs_demo batches.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-dir", type=str, default="model_zoo/torchrec_dlrm/processed_day_0_data")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-embeddings", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--fuse-k", type=int, default=30)
    parser.add_argument("--depths", type=str, default="1,2,4,8,16")
    parser.add_argument("--capacities", type=str, default="1024,4096,8192,20000")
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/recstore_prefetch_exp/lookahead_cache"))
    args = parser.parse_args()

    depths = _parse_int_list(args.depths)
    capacities = _parse_int_list(args.capacities)
    batches = load_batch_ids(
        repo_root=args.repo_root,
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        num_embeddings=args.num_embeddings,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        fuse_k=args.fuse_k,
    )

    reuse_rows = analyze_reuse(batches, depths=depths)
    reuse_summary = summarize_rows(reuse_rows, "depth")
    sim_rows = []
    for capacity in capacities:
        sim_rows.append(simulate_lru(batches, capacity=capacity))
        sim_rows.append(simulate_lru(batches, capacity=capacity, bypass_min_rows=1024))
        sim_rows.append(simulate_hot_static(batches, capacity=capacity))
        for depth in depths:
            sim_rows.append(simulate_lookahead_ttl(batches, capacity=capacity, depth=depth))

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "lookahead_reuse_rows.csv", reuse_rows)
    write_csv(args.output_root / "lookahead_reuse_summary.csv", reuse_summary)
    write_csv(args.output_root / "cache_policy_simulation.csv", sim_rows)

    print(f"loaded_batches={len(batches)}")
    print(f"reuse_summary_csv={args.output_root / 'lookahead_reuse_summary.csv'}")
    print(f"simulation_csv={args.output_root / 'cache_policy_simulation.csv'}")
    print("reuse_summary")
    for row in reuse_summary:
        print(row)
    print("simulation")
    for row in sim_rows:
        print(row)


if __name__ == "__main__":
    main()
