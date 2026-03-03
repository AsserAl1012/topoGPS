from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DescentConfig:
    lr: float = 0.10
    steps: int = 120
    tol_grad: float = 1e-6
    tol_move: float = 1e-6
    clamp_norm: Optional[float] = None
    project_unit: bool = True


@dataclass(frozen=True)
class DescentTraceStep:
    step: int
    z: np.ndarray
    grad_norm: float
    move_norm: float


def _l2_normalize_vec(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)


def _cosine_sims(mat: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    vn = _l2_normalize_vec(v, eps=eps)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + eps)
    return (mn @ vn).astype(np.float32)


def fine_descent(
    q: np.ndarray,
    *,
    landmarks: np.ndarray,
    alpha: np.ndarray,
    sigmas: np.ndarray,
    cfg: DescentConfig,
    init_z: Optional[np.ndarray] = None,
    grid: Optional[object] = None,
    grid_landmarks: Optional[np.ndarray] = None,
    grid_weight: float = 0.0,
) -> Tuple[np.ndarray, List[DescentTraceStep]]:
    """
    Gradient flow over an attraction field induced by landmarks.

    We maximize a weighted mixture-of-Gaussians density by moving toward landmarks:
      grad ∝ Σ_i alpha_i * exp(-||z-li||^2/(2*sigma_i^2)) * (li - z)/sigma_i^2

    init_z: optional start state (used to make cue-routing still run fine settling).
    grid term: optional heuristic pull toward landmarks with high grid-code similarity
               (no backprop through grid features; used as additional attraction weighting).
    """
    E = np.asarray(landmarks, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32).reshape(-1)
    s = np.asarray(sigmas, dtype=np.float32).reshape(-1)

    n, d = E.shape
    if a.shape[0] != n or s.shape[0] != n:
        raise ValueError("alpha/sigmas length mismatch with landmarks")

    z = np.asarray(init_z if init_z is not None else q, dtype=np.float32).reshape(-1)
    if z.shape[0] != d:
        raise ValueError("q/init_z dim mismatch with landmarks")

    if cfg.project_unit:
        z = _l2_normalize_vec(z)

    # Work only on nonzero alpha indices
    idx = np.flatnonzero(a > 0)
    if idx.size == 0:
        return z, []

    Esub = E[idx]
    asub = a[idx]
    ssub = s[idx]

    eps = 1e-9
    trace: List[DescentTraceStep] = []

    # Precompute for speed
    s2 = (ssub * ssub + eps).astype(np.float32)

    for t in range(int(cfg.steps)):
        # Pull toward landmarks
        diff = (Esub - z.reshape(1, -1)).astype(np.float32)      # (m,d)
        d2 = np.sum(diff * diff, axis=1).astype(np.float32)      # (m,)
        g = np.exp(-0.5 * d2 / s2).astype(np.float32)            # (m,)
        w = (asub * g / s2).astype(np.float32)                   # (m,)

        grad = np.sum(diff * w.reshape(-1, 1), axis=0).astype(np.float32)

        # Optional grid attractor heuristic: add extra pull based on grid similarity.
        if grid_weight > 0.0 and grid is not None and grid_landmarks is not None:
            # grid.features(z) should return a vector (gdim,)
            gz = grid.features(z)
            if gz is not None:
                GL = np.asarray(grid_landmarks, dtype=np.float32)
                GLsub = GL[idx]
                gs = _cosine_sims(GLsub, np.asarray(gz, dtype=np.float32))
                # turn grid similarity into a soft weighting
                gs = gs - float(np.max(gs))
                gsw = np.exp(6.0 * gs).astype(np.float32)
                gsw = gsw / (float(np.sum(gsw)) + 1e-12)
                grad += float(grid_weight) * np.sum(diff * (gsw.reshape(-1, 1)), axis=0).astype(np.float32)

        grad_norm = float(np.linalg.norm(grad))
        move = (float(cfg.lr) * grad).astype(np.float32)
        move_norm = float(np.linalg.norm(move))

        z_new = (z + move).astype(np.float32)

        if cfg.clamp_norm is not None:
            cn = float(cfg.clamp_norm)
            nz = float(np.linalg.norm(z_new) + 1e-12)
            if nz > cn:
                z_new = (z_new * (cn / nz)).astype(np.float32)

        if cfg.project_unit:
            z_new = _l2_normalize_vec(z_new)

        trace.append(DescentTraceStep(step=t + 1, z=z_new.copy(), grad_norm=grad_norm, move_norm=move_norm))

        if grad_norm < float(cfg.tol_grad) or move_norm < float(cfg.tol_move):
            z = z_new
            break

        z = z_new

    return z, trace