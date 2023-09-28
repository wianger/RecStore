from cache_common import AbsEmb, ShmTensorStore, TorchNativeStdEmb, CacheShardingPolicy
from sharded_cache import KnownShardedCachedEmbedding, ShardedCachedEmbedding
from local_cache import LocalCachedEmbedding, KnownLocalCachedEmbedding
from utils import print_rank0


class CacheEmbFactory:
    @staticmethod
    def New(cache_type, emb, args) ->AbsEmb:
        if cache_type == "KnownShardedCachedEmbedding":
            cached_range = CacheShardingPolicy.generate_cached_range(
                emb, args['cache_ratio'])
            print_rank0(f"cache_range is {cached_range}")
            abs_emb = KnownShardedCachedEmbedding(
                emb, cached_range=cached_range)
        elif cache_type == "LocalCachedEmbedding":
            abs_emb = LocalCachedEmbedding(emb, cache_ratio=args['cache_ratio'],)
        elif cache_type == "KnownLocalCachedEmbedding":
            cached_range = CacheShardingPolicy.generate_cached_range(
                emb, args['cache_ratio'])
            print_rank0(f"cache_range is {cached_range}")
            abs_emb = KnownLocalCachedEmbedding(emb, cached_range=cached_range)
        else:
            assert False
        return abs_emb

