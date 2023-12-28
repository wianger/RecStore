from cache_common import AbsEmb, ShmTensorStore, TorchNativeStdEmb, CacheShardingPolicy,TorchNativeStdEmbDDP
from sharded_cache import KnownShardedCachedEmbedding, ShardedCachedEmbedding
from local_cache import LocalCachedEmbedding, KnownLocalCachedEmbedding
from utils import print_rank0


class CacheEmbFactory:
    @staticmethod
    def ReturnCachedRange(emb, args):
        cached_range = CacheShardingPolicy.generate_cached_range(
            emb.shape[0], args['cache_ratio'])
        return cached_range

    @staticmethod
    def New(cache_type, emb, args) -> AbsEmb:
        print_rank0(
            f"New CachedEmbedding, name={emb.name}, shape={emb.shape}, cache_type={cache_type}")

        cached_range = CacheShardingPolicy.generate_cached_range(
            emb.shape[0], args['cache_ratio'])

        # cached_range = CacheShardingPolicy.generate_cached_range_from_presampling()
        print_rank0(f"fixed cache_range is {cached_range}")

        if cache_type == "KnownSh ardedCachedEmbedding":
            abs_emb = KnownShardedCachedEmbedding(
                emb, cached_range=cached_range)
        elif cache_type == "LocalCachedEmbedding":
            abs_emb = LocalCachedEmbedding(
                emb, cache_ratio=args['cache_ratio'],)
        elif cache_type == "KnownLocalCachedEmbedding":
            abs_emb = KnownLocalCachedEmbedding(emb,
                                                cached_range=cached_range,
                                                kForwardItersPerStep=args['kForwardItersPerStep'],
                                                backward_mode=args['BackwardMode'],
                                                )
        elif cache_type == "NativeEmbedding":
            abs_emb = TorchNativeStdEmbDDP(emb.weight, device='cpu')
        else:
            assert False
        return abs_emb
