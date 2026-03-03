from __future__ import annotations

from pathlib import Path

import numpy as np

from topogps.core import QueryConfig, TopoGPS
from topogps.utils import l2_normalize


def test_build_from_embeddings_and_query_vec(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    labels = [f"c{i}" for i in range(40)]
    emb = rng.standard_normal((len(labels), 16)).astype(np.float32)
    emb = l2_normalize(emb, axis=1)

    index_dir = tmp_path / "index"
    ws = TopoGPS.build_from_embeddings(labels=labels, embeddings=emb, index_dir=index_dir, model_name="test")

    # Query exactly at a landmark.
    q_idx = 17
    q = emb[q_idx].copy()

    res = TopoGPS.query_vec(ws, q, query_text="c17", cfg=QueryConfig())

    assert res.final_idx == q_idx
    assert res.final_label == labels[q_idx]
    assert res.coarse.best_idx == q_idx
