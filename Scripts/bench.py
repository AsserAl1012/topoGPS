#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from topogps.core import BuildConfig, GridConfig, QueryConfig, TopoGPS
from topogps.utils import l2_normalize, seed_everything


@dataclass
class QueryItem:
    qid: int
    cue_a: int
    cue_b: int
    target: int
    query_text: str
    q_vec: np.ndarray


def _cos_sims(E: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (E @ v.reshape(-1)).astype(np.float32)


def _method_nn(E: np.ndarray, q: np.ndarray) -> int:
    return int(np.argmax(_cos_sims(E, q)))


def _method_minsim(E: np.ndarray, cue_a: np.ndarray, cue_b: np.ndarray) -> np.ndarray:
    sa = _cos_sims(E, cue_a)
    sb = _cos_sims(E, cue_b)
    return np.minimum(sa, sb)


def _method_graph_ppr(
    G: nx.Graph, cue_a_idx: int, cue_b_idx: int, *, exclude: Optional[set] = None
) -> int:
    n = G.number_of_nodes()
    pers = {i: 0.0 for i in range(n)}
    pers[int(cue_a_idx)] = 0.5
    pers[int(cue_b_idx)] = 0.5
    pr = nx.pagerank(G, alpha=0.85, personalization=pers, weight="weight")
    exclude = exclude or set()
    best = None
    bestv = -1.0
    for i, v in pr.items():
        if i in exclude:
            continue
        if v > bestv:
            bestv = float(v)
            best = int(i)
    return int(best) if best is not None else int(cue_a_idx)


def _make_synthetic(
    *,
    task: str,
    n_clusters: int,
    per_cluster: int,
    dim: int,
    seed: int,
) -> Tuple[List[str], np.ndarray, nx.Graph, List[QueryItem]]:
    """
    Two synthetic tasks:

    bridge:
      - true bridge embedding is approx midpoint of cue clusters
      - minsim baseline often works (sanity + "easy" condition)

    associative:
      - bridge embedding is close to cueA but far from cueB
      - embedding-only baselines fail; graph path solves
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers = l2_normalize(centers, axis=1)

    labels: List[str] = []
    E_list: List[np.ndarray] = []
    cluster_nodes: List[List[int]] = []

    # base nodes
    for c in range(n_clusters):
        nodes = []
        for i in range(per_cluster):
            v = centers[c] + 0.10 * rng.standard_normal((dim,), dtype=np.float32)
            v = l2_normalize(v, axis=0)
            idx = len(labels)
            labels.append(f"c{c}_{i:03d}")
            E_list.append(v)
            nodes.append(idx)
        cluster_nodes.append(nodes)

    # add one bridge node per unordered cluster-pair
    bridge_for_pair: Dict[Tuple[int, int], int] = {}
    for a in range(n_clusters):
        for b in range(a + 1, n_clusters):
            if task == "bridge":
                v = centers[a] + centers[b] + 0.05 * rng.standard_normal((dim,), dtype=np.float32)
                v = l2_normalize(v, axis=0)
            elif task == "associative":
                # intentionally biased toward A (embedding-only multi-cue should fail)
                v = centers[a] + 0.02 * rng.standard_normal((dim,), dtype=np.float32)
                v = l2_normalize(v, axis=0)
            else:
                raise ValueError(f"Unknown task: {task}")

            idx = len(labels)
            labels.append(f"bridge_c{a}_c{b}_000")
            E_list.append(v)
            bridge_for_pair[(a, b)] = idx

    E = np.vstack(E_list).astype(np.float32)
    E = l2_normalize(E, axis=1)

    # base similarity graph (will be replaced by TopoGPS.build_from_embeddings, but we also inject edges)
    G = nx.Graph()
    for i in range(len(labels)):
        G.add_node(i, label=labels[i])

    # Inject explicit cue->bridge edges so cue-path is meaningful.
    # Weight 1.0 makes it always preferred.
    for (a, b), br in bridge_for_pair.items():
        # use representative cue nodes (index 0 in each cluster)
        cue_a = cluster_nodes[a][0]
        cue_b = cluster_nodes[b][0]
        G.add_edge(int(cue_a), int(br), weight=1.0)
        G.add_edge(int(br), int(cue_b), weight=1.0)

    # Queries: pick random cluster pairs and use those representative cues
    queries: List[QueryItem] = []
    qid = 0
    for _ in range(20000):  # generate pool; bench will slice n_queries
        a = int(rng.integers(0, n_clusters))
        b = int(rng.integers(0, n_clusters - 1))
        if b >= a:
            b += 1
        aa, bb = (a, b) if a < b else (b, a)
        br = bridge_for_pair[(aa, bb)]
        cue_a = cluster_nodes[a][0]
        cue_b = cluster_nodes[b][0]

        qv = l2_normalize((E[cue_a] + E[cue_b]) * 0.5, axis=0)
        qtext = f"{labels[cue_a]} {labels[cue_b]}"
        queries.append(QueryItem(qid=qid, cue_a=cue_a, cue_b=cue_b, target=br, query_text=qtext, q_vec=qv))
        qid += 1

    return labels, E, G, queries


def run_bench(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(args.seed))
    seed = int(args.seed)

    if args.task not in ("bridge", "associative"):
        raise SystemExit("--task must be: bridge | associative")

    n_clusters = int(args.n_clusters)
    per_cluster = int(args.per_cluster)
    dim = int(args.dim)

    labels, E, injected_graph, pool = _make_synthetic(
        task=args.task,
        n_clusters=n_clusters,
        per_cluster=per_cluster,
        dim=dim,
        seed=seed,
    )

    # Build TopoGPS workspace with synthetic embeddings
    idx_dir = outdir / "index"
    cfg = BuildConfig(
        model_name="synthetic",
        graph_knn=int(args.graph_knn),
        graph_min_sim=float(args.graph_min_sim),
        seed=seed,
        normalize_embeddings=True,
        grid=GridConfig(enabled=bool(args.grid)),
    )
    ws = TopoGPS.build_from_embeddings(
        labels=labels,
        embeddings=E,
        index_dir=idx_dir,
        cfg=cfg,
        meta_extra={"synthetic_task": args.task},
    )

    # Merge injected edges into ws.graph
    for u, v, d in injected_graph.edges(data=True):
        w = float(d.get("weight", 0.0))
        if ws.graph.has_edge(u, v):
            ws.graph[u][v]["weight"] = max(float(ws.graph[u][v].get("weight", 0.0)), w)
        else:
            ws.graph.add_edge(int(u), int(v), weight=w)

    # Prepare query list
    n_queries = int(args.n_queries)
    queries = pool[:n_queries]

    qcfg_full = QueryConfig(
        enable_cue_matching=True,
        cue_path_only=False,
        faiss_candidates=int(args.candidates),
        max_expansions=int(args.max_expansions),
        max_depth=int(args.max_depth),
    )
    qcfg_full.descent = qcfg_full.descent.__class__(lr=float(args.lr), steps=int(args.steps))

    qcfg_cuepath = QueryConfig(
        enable_cue_matching=True,
        cue_path_only=True,
        faiss_candidates=int(args.candidates),
        max_expansions=int(args.max_expansions),
        max_depth=int(args.max_depth),
    )
    qcfg_cuepath.descent = qcfg_cuepath.descent.__class__(lr=float(args.lr), steps=int(args.steps))

    qcfg_no_cue = QueryConfig(
        enable_cue_matching=False,
        cue_path_only=False,
        faiss_candidates=int(args.candidates),
        max_expansions=int(args.max_expansions),
        max_depth=int(args.max_depth),
    )
    qcfg_no_cue.descent = qcfg_no_cue.descent.__class__(lr=float(args.lr), steps=int(args.steps))

    rows: List[Dict[str, object]] = []

    def eval_method(name: str) -> Tuple[float, float, float, float]:
        correct = 0
        rt_ms_total = 0.0
        coarse_total = 0.0
        fine_total = 0.0

        for qi in queries:
            t0 = time.perf_counter()

            if name == "nn":
                pred = _method_nn(E, qi.q_vec)
                coarse_vis = 0
                fine_steps = 0
            elif name == "minsim":
                score = _method_minsim(E, E[qi.cue_a], E[qi.cue_b])
                pred = int(np.argmax(score))
                coarse_vis = 0
                fine_steps = 0
            elif name == "graph_ppr":
                pred = _method_graph_ppr(ws.graph, qi.cue_a, qi.cue_b, exclude={qi.cue_a, qi.cue_b})
                coarse_vis = 0
                fine_steps = 0
            elif name == "topogps":
                res = TopoGPS.query_vec(ws, qi.q_vec, query_text=qi.query_text, cfg=qcfg_full, emit_fine_steps=False)
                pred = int(res.final_idx)
                coarse_vis = float(res.coarse.visited)
                fine_steps = float(res.fine_steps)
            elif name == "topogps_cuepath":
                res = TopoGPS.query_vec(ws, qi.q_vec, query_text=qi.query_text, cfg=qcfg_cuepath, emit_fine_steps=False)
                pred = int(res.final_idx)
                coarse_vis = float(res.coarse.visited)
                fine_steps = float(res.fine_steps)
            elif name == "topogps_no_cue":
                res = TopoGPS.query_vec(ws, qi.q_vec, query_text=qi.query_text, cfg=qcfg_no_cue, emit_fine_steps=False)
                pred = int(res.final_idx)
                coarse_vis = float(res.coarse.visited)
                fine_steps = float(res.fine_steps)
            else:
                raise ValueError(name)

            t1 = time.perf_counter()
            rt_ms = (t1 - t0) * 1000.0

            ok = int(pred == qi.target)
            correct += ok
            rt_ms_total += rt_ms
            coarse_total += coarse_vis
            fine_total += fine_steps

            rows.append(
                {
                    "task": args.task,
                    "qid": qi.qid,
                    "method": name,
                    "cue_a": qi.cue_a,
                    "cue_b": qi.cue_b,
                    "target_idx": qi.target,
                    "pred_idx": pred,
                    "success": ok,
                    "rt_ms": rt_ms,
                    "coarse_visited": coarse_vis,
                    "fine_steps": fine_steps,
                }
            )

        acc = correct / max(1, len(queries))
        rt = rt_ms_total / max(1, len(queries))
        coarse_avg = coarse_total / max(1, len(queries))
        fine_avg = fine_total / max(1, len(queries))
        return acc, rt, coarse_avg, fine_avg

    methods = ["nn", "minsim", "graph_ppr", "topogps", "topogps_cuepath", "topogps_no_cue"]

    results_path = outdir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "qid",
                "method",
                "cue_a",
                "cue_b",
                "target_idx",
                "pred_idx",
                "success",
                "rt_ms",
                "coarse_visited",
                "fine_steps",
            ],
        )
        w.writeheader()

        summary_lines = []
        for m in methods:
            acc, rt, coarse_avg, fine_avg = eval_method(m)
            summary_lines.append((m, acc, rt, coarse_avg, fine_avg))
            print(f"{m:<15} acc={acc:.3f}  rt_ms={rt:.2f}  coarse={coarse_avg:.1f}  fine={fine_avg:.1f}")

        for r in rows:
            w.writerow(r)

    queries_path = outdir / "queries.jsonl"
    with queries_path.open("w", encoding="utf-8") as f:
        for qi in queries:
            f.write(
                json.dumps(
                    {
                        "qid": qi.qid,
                        "cue_a": int(qi.cue_a),
                        "cue_b": int(qi.cue_b),
                        "target_idx": int(qi.target),
                        "query_text": qi.query_text,
                        "q_vec": qi.q_vec.astype(float).tolist(),
                    }
                )
                + "\n"
            )

    print(f"\nWrote: {results_path}")
    print(f"Index dir: {idx_dir}")
    print(f"Queries:   {queries_path}")
    print("Next: python scripts/make_figures.py --run <outdir> --make-plots --make-html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--task", default="bridge", choices=["bridge", "associative"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--candidates", type=int, default=256)

    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--n-clusters", type=int, default=5)
    ap.add_argument("--per-cluster", type=int, default=500)
    ap.add_argument("--dim", type=int, default=64)

    ap.add_argument("--graph-knn", type=int, default=12)
    ap.add_argument("--graph-min-sim", type=float, default=0.55)

    ap.add_argument("--max-expansions", type=int, default=2500)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.08)

    args = ap.parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()