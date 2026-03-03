from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "all-mpnet-base-v2"
    batch_size: int = 64
    normalize: bool = True


class EmbeddingEncoder:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize,
        )
        return np.asarray(emb, dtype=np.float32)


def encode_texts(
    texts: List[str],
    model_name: str = "all-mpnet-base-v2",
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    return EmbeddingEncoder(EncoderConfig(model_name, batch_size, normalize)).encode(texts)


def maybe_load_cached_embeddings(
    cache_dir: Path,
    cache_key: str,
    expected_n: int,
) -> Optional[np.ndarray]:
    path = cache_dir / f"{cache_key}.npy"
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.shape[0] != expected_n:
        return None
    return arr


def save_cached_embeddings(cache_dir: Path, cache_key: str, embeddings: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key}.npy"
    np.save(path, embeddings)
    return path
