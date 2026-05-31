import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionCase:
    name: str
    num_embeddings: int
    embedding_dim: int
    batch_size: int
    seed: int
    cpu: bool = True

    def to_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            seed=self.seed,
            cpu=self.cpu,
        )


@dataclass(frozen=True)
class MultiProcessCase(PrecisionCase):
    world_size: int = 2

    def to_namespace(self) -> argparse.Namespace:
        namespace = super().to_namespace()
        namespace.world_size = self.world_size
        namespace.ps_host = None
        namespace.ps_port = None
        return namespace


def default_precision_cases() -> tuple[PrecisionCase, ...]:
    return (
        PrecisionCase(
            name="basic_cpu",
            num_embeddings=1000,
            embedding_dim=128,
            batch_size=64,
            seed=42,
            cpu=True,
        ),
        PrecisionCase(
            name="small_batch_cpu",
            num_embeddings=500,
            embedding_dim=128,
            batch_size=16,
            seed=42,
            cpu=True,
        ),
    )


def cuda_precision_case() -> PrecisionCase:
    return PrecisionCase(
        name="cuda",
        num_embeddings=1000,
        embedding_dim=128,
        batch_size=64,
        seed=42,
        cpu=False,
    )


def default_multiprocess_case() -> MultiProcessCase:
    return MultiProcessCase(
        name="multiprocess_cpu",
        num_embeddings=1000,
        embedding_dim=128,
        batch_size=32,
        seed=42,
        cpu=True,
        world_size=2,
    )

