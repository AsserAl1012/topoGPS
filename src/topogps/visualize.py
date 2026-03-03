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

from .core import QueryConfig, TopoGPS, TopoGPSWorkspace
from .utils import l2_normalize


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
    # Embed JS so HTML works offline (paper-friendly).
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    return out


def coords_for_path_by_nearest(
    path_z: Sequence[np.ndarray],
    embeddings: np.ndarray,
    coords_3d: np.ndarray,
) -> np.ndarray:
    """Map each z(t) to its nearest landmark, then return those 3D coords."""
    if len(path_z) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    em = np.asarray(embeddings, dtype=np.float32)
    path = []
    for z in path_z:
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        sims = em @ z
        idx = int(np.argmax(sims))
        path.append(coords_3d[idx])
    return np.asarray(path, dtype=np.float32)


def _synthetic_query_vec_from_labels(ws: TopoGPSWorkspace, query: str) -> np.ndarray:
    """
    For synthetic indices (model_name='synthetic'), build the query vector from cue labels
    appearing literally in the query string, e.g. 'c0_000 and c1_000'.
    """
    q = (query or "").strip().lower()
    if not q:
        raise ValueError("Empty query for synthetic map rendering")

    label_to_idx = {lab.lower(): i for i, lab in enumerate(ws.labels)}

    # token-based fast path
    tokens = [t.strip() for t in q.replace(",", " ").replace("&", " ").split() if t.strip()]
    cue_idxs = []
    for t in tokens:
        if t in ("and", "or", "+"):
            continue
        if t in label_to_idx:
            cue_idxs.append(int(label_to_idx[t]))

    # fallback: substring match (still cheap at synthetic sizes)
    if not cue_idxs:
        for i, lab in enumerate(ws.labels):
            if lab.lower() in q:
                cue_idxs.append(i)

    cue_idxs = list(dict.fromkeys(cue_idxs))  # unique preserve order
    if not cue_idxs:
        raise ValueError(
            "Synthetic query did not contain any cue labels. "
            "Expected something like 'c0_000 and c1_000'."
        )

    v = np.mean(ws.embeddings[np.asarray(cue_idxs, dtype=int)], axis=0).astype(np.float32)
    if bool(ws.meta.get("normalized", True)):
        v = l2_normalize(v, axis=0).astype(np.float32)
    return v


def render_html(
    ws: TopoGPSWorkspace,
    *,
    out: Path,
    query: Optional[str] = None,
    cue_matching: bool = False,
    umap_seed: int = 42,
) -> Path:
    """Write an interactive 3D HTML map (optionally with a query path overlay)."""
    coords = project_umap_3d(ws.embeddings, UMAPConfig(random_state=umap_seed))
    path_coords = None
    title = "TopoGPS semantic map"

    if query:
        title = f"TopoGPS semantic map — {query}"
        qcfg = QueryConfig(enable_cue_matching=cue_matching)

        model_name = str(ws.meta.get("model_name", "")).strip().lower()
        if model_name == "synthetic":
            qvec = _synthetic_query_vec_from_labels(ws, query)
            res = TopoGPS.query_vec(ws, qvec, query=query, cfg=qcfg, top_k=5, emit_fine_steps=True)
        else:
            # Normal (HF-backed) path
            res = TopoGPS.query(ws, query, cfg=qcfg, top_k=5, emit_fine_steps=True)

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