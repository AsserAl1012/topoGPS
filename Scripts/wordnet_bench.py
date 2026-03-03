#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np

from topogps.core import BuildConfig, QueryConfig, TopoGPS
from topogps.embeddings import encode_texts
from topogps.utils import l2_normalize, seed_everything


def _ensure_wordnet():
    import nltk
    try:
        from nltk.corpus import wordnet as wn  # noqa
        _ = wn.synsets("dog")
    except Exception:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def build_wordnet_graph(max_words: int, seed: int) -> Tuple[List[str], nx.Graph]:
    from nltk.corpus import wordnet as wn

    rng = np.random.default_rng(seed)

    # collect lemma names (avoid multiword, keep simple)
    lemmas: List[str] = []
    for syn in wn.all_synsets(pos="n"):
        for l in syn.lemmas():
            w = l.name().lower()
            if "_" in w or "-" in w:
                continue
            if len(w) < 3 or len(w) > 18:
                continue
            lemmas.append(w)

    lemmas = sorted(set(lemmas))
    if len(lemmas) > max_words:
        pick = rng.choice(len(lemmas), size=max_words, replace=False)
        words = [lemmas[i] for i in pick]
    else:
        words = lemmas

    idx = {w: i for i, w in enumerate(words)}

    G = nx.Graph()
    for w, i in idx.items():
        G.add_node(i, label=w)

    # synonym + hypernym edges (lemma-level)
    for syn in wn.all_synsets(pos="n"):
        syn_words = [l.name().lower() for l in syn.lemmas()]
        syn_words = [w for w in syn_words if w in idx]

        # synonyms clique
        for i in range(len(syn_words)):
            for j in range(i + 1, len(syn_words)):
                G.add_edge(idx[syn_words[i]], idx[syn_words[j]], weight=1.0)

        # hypernym links
        hypers = syn.hypernyms()
        for h in hypers:
            h_words = [l.name().lower() for l in h.lemmas()]
            h_words = [w for w in h_words if w in idx]
            for w in syn_words:
                for hw in h_words:
                    G.add_edge(idx[w], idx[hw], weight=1.0)

    return words, G


def gen_hypernym_queries(
    words: List[str],
    G: nx.Graph,
    n: int,
    seed: int,
) -> List[Tuple[int, int, int, str]]:
    """
    Pick pairs of words that share a neighbor; target is that shared neighbor.
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(words)

    neigh = {i: list(G.neighbors(i)) for i in range(n_nodes)}

    queries: List[Tuple[int, int, int, str]] = []
    tries = 0
    while len(queries) < n and tries < n * 80:
        tries += 1
        a = int(rng.integers(0, n_nodes))
        b = int(rng.integers(0, n_nodes))
        if a == b:
            continue

        na = set(neigh.get(a, []))
        nb = set(neigh.get(b, []))
        inter = list(na.intersection(nb))
        if not inter:
            continue

        t = int(rng.choice(inter))
        qtext = f"{words[a]} {words[b]}"
        queries.append((a, b, t, qtext))

    return queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-words", type=int, default=2000)
    ap.add_argument("--n-queries", type=int, default=500)
    ap.add_argument("--model", default="all-mpnet-base-v2")
    ap.add_argument("--candidates", type=int, default=256)
    args = ap.parse_args()

    _ensure_wordnet()
    seed_everything(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    idx_dir = outdir / "index"

    words, G = build_wordnet_graph(args.max_words, args.seed)
    if len(words) < 10:
        raise SystemExit("Word list too small; increase --max-words.")

    # Warm up model download (do NOT call TopoGPS.build with a list)
    _ = encode_texts(words[:2], model_name=args.model, normalize=True).astype(np.float32)

    # Embed all words
    emb = encode_texts(words, model_name=args.model, normalize=True).astype(np.float32)

    cfg = BuildConfig(model_name=args.model, seed=args.seed, graph_knn=12, graph_min_sim=0.55)
    ws = TopoGPS.build_from_embeddings(labels=words, embeddings=emb, index_dir=idx_dir, cfg=cfg)

    # Override graph with WordNet graph
    ws.graph = G

    queries = gen_hypernym_queries(words, G, args.n_queries, args.seed)
    if not queries:
        raise SystemExit("Could not generate queries; try increasing --max-words or --n-queries.")

    methods = ["nn", "graph_ppr", "topogps", "topogps_cuepath", "topogps_no_cue"]

    qcfg = QueryConfig(enable_cue_matching=True, cue_path_only=False, faiss_candidates=args.candidates)
    qcfg_cuepath = QueryConfig(enable_cue_matching=True, cue_path_only=True, faiss_candidates=args.candidates)
    qcfg_no = QueryConfig(enable_cue_matching=False, cue_path_only=False, faiss_candidates=args.candidates)

    def nn_pred(qv: np.ndarray) -> int:
        return int(np.argmax(ws.embeddings @ qv.reshape(-1)))

    # NOTE: PPR is expensive per-query; acceptable as a baseline at 500 queries / 2k nodes.
    def graph_ppr(a: int, b: int) -> int:
        pers = {i: 0.0 for i in range(len(words))}
        pers[a] = 0.5
        pers[b] = 0.5
        pr = nx.pagerank(G, alpha=0.85, personalization=pers, weight="weight")
        best = None
        bestv = -1.0
        for i, v in pr.items():
            if i in (a, b):
                continue
            if v > bestv:
                bestv = float(v)
                best = int(i)
        return int(best) if best is not None else a

    out_csv = outdir / "results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["qid", "method", "success", "rt_ms", "coarse_visited", "fine_steps"],
        )
        w.writeheader()

        for m in methods:
            ok = 0
            rt = 0.0
            coarse = 0.0
            fine = 0.0

            for qid, (a, b, t, qtext) in enumerate(queries):
                qv = l2_normalize((ws.embeddings[a] + ws.embeddings[b]) * 0.5, axis=0)

                t0 = time.perf_counter()
                if m == "nn":
                    pred = nn_pred(qv)
                    cv, fs = 0.0, 0.0
                elif m == "graph_ppr":
                    pred = graph_ppr(a, b)
                    cv, fs = 0.0, 0.0
                elif m == "topogps":
                    res = TopoGPS.query_vec(ws, qv, query_text=qtext, cfg=qcfg, emit_fine_steps=False)
                    pred = int(res.final_idx)
                    cv, fs = float(res.coarse.visited), float(res.fine_steps)
                elif m == "topogps_cuepath":
                    res = TopoGPS.query_vec(ws, qv, query_text=qtext, cfg=qcfg_cuepath, emit_fine_steps=False)
                    pred = int(res.final_idx)
                    cv, fs = float(res.coarse.visited), float(res.fine_steps)
                elif m == "topogps_no_cue":
                    res = TopoGPS.query_vec(ws, qv, query_text=qtext, cfg=qcfg_no, emit_fine_steps=False)
                    pred = int(res.final_idx)
                    cv, fs = float(res.coarse.visited), float(res.fine_steps)
                else:
                    raise ValueError(m)

                t1 = time.perf_counter()
                rt_ms = (t1 - t0) * 1000.0

                s = int(pred == t)
                ok += s
                rt += rt_ms
                coarse += cv
                fine += fs

                w.writerow(
                    {
                        "qid": qid,
                        "method": m,
                        "success": s,
                        "rt_ms": rt_ms,
                        "coarse_visited": cv,
                        "fine_steps": fs,
                    }
                )

            n = max(1, len(queries))
            print(f"{m:<15} acc={ok/n:.3f}  rt_ms={rt/n:.2f}  coarse={coarse/n:.1f}  fine={fine/n:.1f}")

    print(f"\nWrote: {out_csv}")
    print(f"Index dir: {idx_dir}")


if __name__ == "__main__":
    main()