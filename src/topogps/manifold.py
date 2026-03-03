from __future__ import annotations

from dataclasses import dataclass

import faiss  # type: ignore
import numpy as np


@dataclass
class SigmaKNNConfig:
    knn: int = 12
    scale: float = 1.25
    min_sigma: float = 0.05
    max_sigma: float = 1.25


def compute_local_sigmas(embeddings: np.ndarray, *, cfg: SigmaKNNConfig = SigmaKNNConfig()) -> np.ndarray:
    """Compute per-landmark sigma_i from kNN local scale in embedding space.

    Uses FAISS L2 kNN on the embedding vectors. If embeddings are unit-normalized,
    L2 distance is monotonic with cosine distance.

    sigma_i = clip(scale * mean_{kNN}(||x_i - x_j||), min_sigma, max_sigma)
    """
    E = np.asarray(embeddings, dtype=np.float32)
    n, d = E.shape
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    k = int(max(1, cfg.knn))
    k_eff = min(k + 1, n)  # include self

    index = faiss.IndexFlatL2(d)
    index.add(E)
    d2, _ = index.search(E, k_eff)

    if k_eff <= 1:
        return np.full((n,), float(cfg.min_sigma), dtype=np.float32)

    dists = np.sqrt(np.maximum(d2[:, 1:], 0.0))
    local_mean = np.mean(dists, axis=1).astype(np.float32)

    sigmas = float(cfg.scale) * local_mean
    sigmas = np.clip(sigmas, float(cfg.min_sigma), float(cfg.max_sigma)).astype(np.float32)
    return sigmas
