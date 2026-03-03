from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .grid import GridCode


@dataclass
class DescentConfig:
    lr: float = 0.08
    steps: int = 40
    tol_grad: float = 1e-5
    tol_move: float = 1e-5
    clamp_norm: Optional[float] = None
    project_unit: bool = True


@dataclass
class DescentStep:
    step: int
    z: np.ndarray
    grad_norm: float
    move_norm: float


def _clamp_vec(z: np.ndarray, *, clamp_norm: Optional[float]) -> np.ndarray:
    if clamp_norm is None:
        return z
    cn = float(clamp_norm)
    if cn <= 0:
        return z
    n = float(np.linalg.norm(z))
    if n <= cn:
        return z
    return (z / max(n, 1e-12)) * cn


def _project_unit(z: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(z))
    if n <= 1e-12:
        return z
    return z / n


def _grid_attractor_grad(
    *,
    landmarks: np.ndarray,
    grid_landmarks: np.ndarray,
    z: np.ndarray,
    q_grid: np.ndarray,
) -> np.ndarray:
    """
    Approximate grid attractor gradient.

    w_i = relu(cos(grid_i, q_grid))
    grad = sum_i w_i * (z - landmark_i)
    """
    G = np.asarray(grid_landmarks, dtype=np.float32)
    q = np.asarray(q_grid, dtype=np.float32).reshape(1, -1)
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    E = np.asarray(landmarks, dtype=np.float32)

    Gn = G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-8)
    qn = q / max(float(np.linalg.norm(q)), 1e-8)

    w = (Gn @ qn.T).reshape(-1).astype(np.float32)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 1e-8:
        return np.zeros_like(z)

    w = (w / s).reshape(-1, 1)
    grad = np.sum(w * (z.reshape(1, -1) - E), axis=0).astype(np.float32)
    return grad


def fine_descent(
    q: np.ndarray,
    *,
    landmarks: np.ndarray,
    alpha: np.ndarray,
    sigmas: np.ndarray,
    cfg: DescentConfig,
    init_z: Optional[np.ndarray] = None,
    grid: Optional[GridCode] = None,
    grid_landmarks: Optional[np.ndarray] = None,
    grid_weight: float = 0.0,
) -> Tuple[np.ndarray, List[DescentStep]]:
    """
    Continuous descent in embedding space.

    Density:
      D(z) = sum_i alpha_i * exp(-||z - x_i||^2 / (2*sigma_i^2))

    We ASCEND density (equivalently descend -D).

    If grid/grid_landmarks are provided and grid_weight>0, we add an extra attractor
    term that nudges z toward landmarks whose grid code matches the query's grid code.
    """
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    z = np.asarray(init_z if init_z is not None else q, dtype=np.float32).reshape(-1).copy()

    E = np.asarray(landmarks, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32).reshape(-1)
    s = np.asarray(sigmas, dtype=np.float32).reshape(-1)

    if E.shape[0] != a.shape[0] or E.shape[0] != s.shape[0]:
        raise ValueError("landmarks/alpha/sigmas mismatch")

    use_grid = (grid is not None) and (grid_landmarks is not None) and (float(grid_weight) > 0.0)
    q_grid: Optional[np.ndarray] = None
    if use_grid:
        q_grid = grid.features(q)

    trace: List[DescentStep] = []
    prev_z = z.copy()

    for t in range(int(cfg.steps)):
        diffs = z.reshape(1, -1) - E  # (N, D)
        d2 = np.sum(diffs * diffs, axis=1)  # (N,)
        ss2 = np.maximum(s * s, 1e-10)

        w = a * np.exp(-0.5 * d2 / ss2)  # (N,)

        # gradient of density wrt z: sum_i w_i * (x_i - z) / sigma_i^2
        grad = np.sum((w / ss2).reshape(-1, 1) * (E - z.reshape(1, -1)), axis=0).astype(np.float32)

        if use_grid and q_grid is not None:
            grad_grid = _grid_attractor_grad(
                landmarks=E,
                grid_landmarks=np.asarray(grid_landmarks, dtype=np.float32),
                z=z,
                q_grid=q_grid,
            )
            # grad_grid is (z - E_i) style, so subtract to move toward E_i
            grad = grad + float(grid_weight) * (-grad_grid)

        gnorm = float(np.linalg.norm(grad))
        if gnorm <= float(cfg.tol_grad):
            trace.append(DescentStep(step=t, z=z.copy(), grad_norm=gnorm, move_norm=0.0))
            break

        z = z + float(cfg.lr) * grad
        z = _clamp_vec(z, clamp_norm=cfg.clamp_norm)
        if cfg.project_unit:
            z = _project_unit(z)

        move = float(np.linalg.norm(z - prev_z))
        trace.append(DescentStep(step=t, z=z.copy(), grad_norm=gnorm, move_norm=move))

        if move <= float(cfg.tol_move):
            break

        prev_z = z.copy()

    return z, trace