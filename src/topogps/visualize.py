from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

import plotly.graph_objects as go


@dataclass(frozen=True)
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: int = 42


def project_umap_3d(embeddings: np.ndarray, cfg: UMAPConfig = UMAPConfig()) -> np.ndarray:
    if umap is None:
        raise RuntimeError("umap-learn is not installed")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=cfg.n_neighbors,
        min_dist=cfg.min_dist,
        random_state=cfg.random_state,
        metric="cosine",
    )
    coords = reducer.fit_transform(embeddings)
    return np.asarray(coords, dtype=np.float32)


def plot_map_3d(
    coords_3d: np.ndarray,
    labels: List[str],
    *,
    path_coords: Optional[np.ndarray] = None,
    title: str = "TopoGPS semantic map",
) -> go.Figure:
    x, y, z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=3),
            name="landmarks",
        )
    )

    if path_coords is not None and len(path_coords) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=path_coords[:, 0],
                y=path_coords[:, 1],
                z=path_coords[:, 2],
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=6),
                name="retrieval path",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True,
    )
    return fig


def save_figure_html(fig: go.Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


def coords_for_path_by_nearest(
    path_z: Sequence[np.ndarray],
    embeddings: np.ndarray,
    coords_3d: np.ndarray,
) -> np.ndarray:
    """Map each z(t) to its nearest landmark, then return those 3D coords."""
    if len(path_z) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # embeddings are typically normalized. we use dot = cosine
    em = embeddings
    path = []
    for z in path_z:
        sims = em @ z.reshape(-1)
        idx = int(np.argmax(sims))
        path.append(coords_3d[idx])
    return np.asarray(path, dtype=np.float32)

from .core import QueryConfig, TopoGPS, TopoGPSWorkspace


def render_html(
    ws: TopoGPSWorkspace,
    *,
    out: Path,
    query: Optional[str] = None,
    cue_matching: bool = False,
    umap_seed: int = 42,
) -> Path:
    """CLI-facing convenience wrapper.

    Writes an interactive 3D HTML map. If `query` is provided, overlays the
    retrieval path (fine path if available, otherwise coarse path).
    """
    coords = project_umap_3d(ws.embeddings, UMAPConfig(random_state=umap_seed))
    path_coords = None

    title = "TopoGPS semantic map"
    if query:
        title = f"TopoGPS semantic map — {query}"
        qcfg = QueryConfig(enable_cue_matching=cue_matching)
        # try to request fine steps if the signature supports it
        try:
            res = TopoGPS.query(ws, query, cfg=qcfg, top_k=5, emit_fine_steps=True)
        except TypeError:
            res = TopoGPS.query(ws, query, cfg=qcfg, top_k=5)

        fine = getattr(res, "fine_path_z", None)
        if isinstance(fine, (list, tuple)) and len(fine) >= 2:
            path_coords = coords_for_path_by_nearest(fine, ws.embeddings, coords)
        else:
            coarse = getattr(res, "coarse", None)
            best_path = getattr(coarse, "best_path", []) if coarse is not None else []
            if best_path:
                path_coords = coords[np.asarray(best_path, dtype=int)]

    fig = plot_map_3d(coords, list(ws.labels), path_coords=path_coords, title=title)
    return save_figure_html(fig, Path(out))
