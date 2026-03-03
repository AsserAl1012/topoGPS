from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import networkx as nx
import numpy as np


@dataclass
class BundlePaths:
    root: Path
    embeddings: Path
    labels: Path
    faiss_index: Path
    graph: Path
    meta: Path
    sigmas: Path
    grid_feats: Path


def bundle_paths(index_dir: Path) -> BundlePaths:
    root = Path(index_dir)
    return BundlePaths(
        root=root,
        embeddings=root / "embeddings.npy",
        labels=root / "labels.json",
        faiss_index=root / "index.faiss",
        graph=root / "graph.gpickle",
        meta=root / "meta.json",
        sigmas=root / "sigmas.npy",
        grid_feats=root / "grid_feats.npy",
    )


def save_bundle(
    index_dir: Path,
    *,
    embeddings: np.ndarray,
    labels: List[str],
    index: faiss.Index,
    graph: nx.Graph,
    meta: Dict[str, Any],
    sigmas: Optional[np.ndarray] = None,
    grid_feats: Optional[np.ndarray] = None,
) -> None:
    p = bundle_paths(index_dir)
    p.root.mkdir(parents=True, exist_ok=True)

    np.save(p.embeddings, np.asarray(embeddings, dtype=np.float32))
    p.labels.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    faiss.write_index(index, str(p.faiss_index))

    # networkx removed top-level write_gpickle/read_gpickle in recent versions.
    # Use stdlib pickle for maximum compatibility.
    with p.graph.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    p.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if sigmas is not None:
        np.save(p.sigmas, np.asarray(sigmas, dtype=np.float32))
    if grid_feats is not None:
        np.save(p.grid_feats, np.asarray(grid_feats, dtype=np.float32))


def load_bundle(index_dir: Path) -> Tuple[
    BundlePaths,
    np.ndarray,
    List[str],
    faiss.Index,
    nx.Graph,
    Dict[str, Any],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    p = bundle_paths(index_dir)

    E = np.load(p.embeddings).astype(np.float32)
    labels: List[str] = json.loads(p.labels.read_text(encoding="utf-8"))
    index = faiss.read_index(str(p.faiss_index))

    with p.graph.open("rb") as f:
        G = pickle.load(f)

    meta: Dict[str, Any] = json.loads(p.meta.read_text(encoding="utf-8"))

    sigmas = np.load(p.sigmas).astype(np.float32) if p.sigmas.exists() else None
    grid_feats = np.load(p.grid_feats).astype(np.float32) if p.grid_feats.exists() else None

    return p, E, labels, index, G, meta, sigmas, grid_feats