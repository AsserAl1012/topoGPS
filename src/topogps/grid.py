from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass
class GridCodeConfig:
    n_modules: int = 6
    d_per_module: int = 8
    lambdas: List[float] = None  # set in __post_init__
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.lambdas is None:
            self.lambdas = [0.45, 0.75, 1.25, 2.1, 3.5, 5.8]


def _random_proj(d_in: int, d_out: int, seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_in, d_out), dtype=np.float32)
    # normalize columns for stability
    W /= np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-8)
    return W


def encode_grid_codes(E: np.ndarray, cfg: GridCodeConfig) -> np.ndarray:
    """
    Encode embeddings into multi-module grid codes.
    For each module m, project E into d_per_module dims, then compute sinusoidal phase code:
      code = [cos(2π x/λ), sin(2π x/λ)] per dim (packed as 2*d_per_module),
    then flatten modules => (N, n_modules*2*d_per_module).
    """
    E = np.asarray(E, dtype=np.float32)
    if E.ndim != 2:
        raise ValueError("encode_grid_codes expects (N, D) array")
    n, d = E.shape
    M = int(cfg.n_modules)
    dd = int(cfg.d_per_module)

    W = _random_proj(d, dd, cfg.seed)
    X = E @ W  # (N, dd)

    feats = []
    for m in range(M):
        lam = float(cfg.lambdas[m % len(cfg.lambdas)])
        phase = (2.0 * np.pi / max(lam, 1e-6)) * X
        feats.append(np.cos(phase))
        feats.append(np.sin(phase))

    out = np.concatenate(feats, axis=1).astype(np.float32)
    return out


def grid_code_similarity(grid_codes: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between grid codes and query grid code.
    grid_codes: (N, G)
    q_grid: (G,) or (1, G)
    returns: (N,)
    """
    G = np.asarray(grid_codes, dtype=np.float32)
    q = np.asarray(q_grid, dtype=np.float32).reshape(1, -1)
    Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-8)
    qn = q / max(np.linalg.norm(q), 1e-8)
    return (Gn @ qn.T).reshape(-1).astype(np.float32)


def grid_attractor_grad(E: np.ndarray, grid_codes: np.ndarray, z: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
    """
    Additive gradient term that nudges z toward landmarks whose grid codes match q_grid.
    Approximation:
      w_i = relu(sim(grid_i, q_grid))
      grad = sum_i w_i * (z - E_i)   (normalized weights)
    """
    E = np.asarray(E, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    w = grid_code_similarity(grid_codes, q_grid)
    w = np.maximum(w, 0.0).astype(np.float32)  # only attractive
    s = float(w.sum())
    if s <= 1e-8:
        return np.zeros_like(z, dtype=np.float32)
    w = (w / s).reshape(-1, 1)
    grad = np.sum(w * (z.reshape(1, -1) - E), axis=0).astype(np.float32)
    return grad


class GridCode:
    """
    Thin OO wrapper used by core.py / energy.py.

    Supports BOTH call styles:
      - GridCode.random(dim=..., n_modules=..., d_per_module=..., lambdas=..., seed=...)
      - GridCode.random(D=..., cfg={...})   <-- your current core.py does this
    """

    def __init__(self, *, dim: int, cfg: GridCodeConfig):
        self.dim = int(dim)
        self.cfg = cfg
        # pre-sample projection so features() is deterministic and fast
        self._W = _random_proj(self.dim, int(self.cfg.d_per_module), self.cfg.seed)

    @staticmethod
    def random(
        *,
        dim: Optional[int] = None,
        D: Optional[int] = None,
        n_modules: Optional[int] = None,
        d_per_module: Optional[int] = None,
        lambdas: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
        cfg: Optional[Union[GridCodeConfig, Dict[str, Any]]] = None,
    ) -> "GridCode":
        """
        Create a deterministic grid-code encoder.

        Accepts:
          - dim=... (preferred)
          - D=...   (legacy / alternate name)
          - cfg=GridCodeConfig or cfg=dict with keys:
              n_modules, d_per_module, lambdas, seed
          - or direct n_modules/d_per_module/lambdas/seed
        """
        d = dim if dim is not None else D
        if d is None:
            raise TypeError("GridCode.random requires dim=... (or D=...)")

        # Start from cfg if provided
        if isinstance(cfg, GridCodeConfig):
            gc = GridCodeConfig(
                n_modules=int(cfg.n_modules),
                d_per_module=int(cfg.d_per_module),
                lambdas=list(cfg.lambdas),
                seed=int(cfg.seed) if cfg.seed is not None else None,
            )
        elif isinstance(cfg, dict):
            gc = GridCodeConfig(
                n_modules=int(cfg.get("n_modules", 6)),
                d_per_module=int(cfg.get("d_per_module", 8)),
                lambdas=list(cfg.get("lambdas", [0.45, 0.75, 1.25, 2.1, 3.5, 5.8])),
                seed=int(cfg["seed"]) if cfg.get("seed", None) is not None else None,
            )
        else:
            gc = GridCodeConfig()

        # Override with explicit args if provided
        if n_modules is not None:
            gc.n_modules = int(n_modules)
        if d_per_module is not None:
            gc.d_per_module = int(d_per_module)
        if lambdas is not None:
            gc.lambdas = [float(x) for x in lambdas]
        if seed is not None:
            gc.seed = int(seed)

        gc.__post_init__()  # ensure lambdas set
        return GridCode(dim=int(d), cfg=gc)

    def features(self, x: np.ndarray) -> np.ndarray:
        """
        Compute grid-code features.
          - if x is (D,) -> returns (G,)
          - if x is (N,D) -> returns (N,G)
        """
        X = np.asarray(x, dtype=np.float32)
        if X.ndim == 1:
            X2 = X.reshape(1, -1)
            out = self._features_2d(X2)
            return out.reshape(-1)
        if X.ndim == 2:
            return self._features_2d(X)
        raise ValueError("GridCode.features expects (D,) or (N,D)")

    def _features_2d(self, E: np.ndarray) -> np.ndarray:
        E = np.asarray(E, dtype=np.float32)
        if E.shape[1] != self.dim:
            raise ValueError(f"GridCode.features dim mismatch: got {E.shape[1]}, expected {self.dim}")

        dd = int(self.cfg.d_per_module)
        M = int(self.cfg.n_modules)

        X = E @ self._W  # (N, dd)

        feats = []
        for m in range(M):
            lam = float(self.cfg.lambdas[m % len(self.cfg.lambdas)])
            phase = (2.0 * np.pi / max(lam, 1e-6)) * X
            feats.append(np.cos(phase))
            feats.append(np.sin(phase))

        return np.concatenate(feats, axis=1).astype(np.float32)

    def similarity(self, grid_codes: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
        return grid_code_similarity(grid_codes, q_grid)

    def attractor_grad(self, E: np.ndarray, grid_codes: np.ndarray, z: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
        return grid_attractor_grad(E, grid_codes, z, q_grid)