from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from topogps.core import BuildConfig, GridConfig, QueryConfig, TopoGPS
from topogps.utils import l2_normalize, seed_everything, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TopoGPS benchmark harness (paper-grade).")
    p.add_argument("--outdir", type=Path, default=Path("results/run"), help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--dim", type=int, default=64, help="Synthetic embedding dim")
    p.add_argument("--clusters", type=int, default=5, help="Number of clusters (c0..cK)")
    p.add_argument("--per-cluster", type=int, default=120, help="Nodes per cluster")
    p.add_argument("--noise", type=float, default=0.08, help="Cluster noise")
    p.add_argument("--bridge-noise", type=float, default=0.03, help="Bridge noise")
    p.add_argument("--query-noise", type=float, default=0.03, help="Query noise")
    p.add_argument("--cue-b-mix", type=float, default=0.15, help="Query = cueA + mix*cueB + noise")
    p.add_argument("--n-queries", type=int, default=2000, help="Number of random bridge queries to generate")
    p.add_argument("--task", type=str, default="associative", choices=["geometric", "associative"], help="Benchmark task")
    p.add_argument("--grid", action="store_true", help="Enable grid features in build")
    p.add_argument("--candidates", type=int, default=256, help="FAISS top-M candidates (0 disables)")
    p.add_argument("--steps", type=int, default=120, help="Fine descent steps")
    p.add_argument("--lr", type=float, default=0.10, help="Fine descent lr")
    p.add_argument("--beta", type=float, default=1.0, help="Coarse search beta")
    p.add_argument("--max-depth", type=int, default=4, help="Coarse max depth")
    p.add_argument("--max-expansions", type=int, default=2500, help="Coarse max expansions")
    p.add_argument(
        "--methods",
        type=str,
        default="nn,minsim,graph_ppr,topogps,topogps_cuepath,topogps_no_cue",
        help="Comma-separated methods: nn,minsim,graph_ppr,topogps,topogps_cuepath,topogps_no_cue",
    )
    return p.parse_args()


def make_synth_world(
    *,
    seed: int,
    dim: int,
    clusters: int,
    per_cluster: int,
    noise: float,
    bridge_noise: float,
    task: str,
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(seed)
    centers = l2_normalize(rng.standard_normal((clusters, dim)).astype(np.float32), axis=1)

    labels: List[str] = []
    E_list: List[np.ndarray] = []

    # cluster nodes
    for ci in range(clusters):
        for j in range(per_cluster):
            lab = f"c{ci}_{j:03d}"
            x = centers[ci] + noise * rng.standard_normal((dim,)).astype(np.float32)
            x = l2_normalize(x, axis=0).astype(np.float32)
            labels.append(lab)
            E_list.append(x)

    if task == "geometric":
        # one midpoint bridge per pair (minsim will dominate; kept as sanity task)
        for a in range(clusters):
            for b in range(a + 1, clusters):
                lab = f"bridge_c{a}_c{b}"
                x = 0.5 * (centers[a] + centers[b]) + bridge_noise * rng.standard_normal((dim,)).astype(np.float32)
                x = l2_normalize(x, axis=0).astype(np.float32)
                labels.append(lab)
                E_list.append(x)

    else:
        # associative: per-index connector node. Not a geometric midpoint.
        # Bridge vectors are biased toward cluster a (so embedding-only intersection fails).
        for a in range(clusters):
            for b in range(a + 1, clusters):
                for j in range(per_cluster):
                    lab = f"bridge_c{a}_c{b}_{j:03d}"
                    x = centers[a] + 0.02 * centers[b] + bridge_noise * rng.standard_normal((dim,)).astype(np.float32)
                    x = l2_normalize(x, axis=0).astype(np.float32)
                    labels.append(lab)
                    E_list.append(x)

    E = np.stack(E_list, axis=0).astype(np.float32)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    return labels, E, label_to_idx


def add_associative_edges(G: nx.Graph, label_to_idx: Dict[str, int], clusters: int, per_cluster: int) -> None:
    """
    Make the associative task well-defined:
      cueA_j -> bridge_ab_j -> cueB_j
    by adding explicit strong edges.
    """
    w = 0.99
    for a in range(clusters):
        for b in range(a + 1, clusters):
            for j in range(per_cluster):
                cue_a = label_to_idx[f"c{a}_{j:03d}"]
                cue_b = label_to_idx[f"c{b}_{j:03d}"]
                bridge = label_to_idx[f"bridge_c{a}_c{b}_{j:03d}"]
                G.add_edge(cue_a, bridge, weight=w)
                G.add_edge(bridge, cue_b, weight=w)


def gen_queries(rng: np.random.Generator, *, clusters: int, per_cluster: int, n_queries: int, task: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for k in range(n_queries):
        a = int(rng.integers(0, clusters))
        b = int(rng.integers(0, clusters - 1))
        if b >= a:
            b += 1
        aa, bb = (a, b) if a < b else (b, a)

        j = int(rng.integers(0, per_cluster))
        cue_a = f"c{a}_{j:03d}"
        cue_b = f"c{b}_{j:03d}"
        if task == "geometric":
            expected = f"bridge_c{aa}_c{bb}"
        else:
            expected = f"bridge_c{aa}_c{bb}_{j:03d}"
        q = f"{cue_a} and {cue_b}"
        out.append({"id": f"q{k:05d}", "cue_a": cue_a, "cue_b": cue_b, "expected": expected, "query": q})
    return out


def nn_pred(E: np.ndarray, q: np.ndarray) -> int:
    sims = (E @ q.reshape(-1)).astype(np.float32)
    return int(np.argmax(sims))


def minsim_pred(E: np.ndarray, cue_a_idx: int, cue_b_idx: int) -> int:
    va = E[cue_a_idx]
    vb = E[cue_b_idx]
    sims_a = (E @ va).astype(np.float32)
    sims_b = (E @ vb).astype(np.float32)
    score = np.minimum(sims_a, sims_b)
    score[cue_a_idx] = -np.inf
    score[cue_b_idx] = -np.inf
    return int(np.argmax(score))


def graph_ppr_pred(G: nx.Graph, cue_a_idx: int, cue_b_idx: int) -> int:
    personalization = {n: 0.0 for n in G.nodes}
    personalization[cue_a_idx] = 0.5
    personalization[cue_b_idx] = 0.5
    pr = nx.pagerank(G, alpha=0.85, personalization=personalization, weight="weight", max_iter=200, tol=1e-6)
    pr[cue_a_idx] = -1.0
    pr[cue_b_idx] = -1.0
    return int(max(pr, key=lambda k: pr[k]))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    labels, E, label_to_idx = make_synth_world(
        seed=args.seed,
        dim=args.dim,
        clusters=args.clusters,
        per_cluster=args.per_cluster,
        noise=args.noise,
        bridge_noise=args.bridge_noise,
        task=args.task,
    )

    # build TopoGPS index from embeddings (no HF)
    index_dir = outdir / "index"
    bcfg = BuildConfig(
        model_name="synthetic",
        graph_knn=12,
        graph_min_sim=0.55,
        seed=args.seed,
        normalize_embeddings=True,
        sigma_cfg=BuildConfig().sigma_cfg,
        grid=GridConfig(enabled=bool(args.grid)),
    )
    ws = TopoGPS.build_from_embeddings(labels=labels, embeddings=E, index_dir=index_dir, cfg=bcfg, model_name="synthetic")

    # critical: inject associative edges
    if args.task == "associative":
        add_associative_edges(ws.graph, label_to_idx, args.clusters, args.per_cluster)

    base_qcfg = QueryConfig(
        beta=float(args.beta),
        max_depth=int(args.max_depth),
        max_expansions=int(args.max_expansions),
        faiss_candidates=int(max(0, args.candidates)),
    )
    base_qcfg.descent = base_qcfg.descent.__class__(lr=float(args.lr), steps=int(args.steps))

    qcfg_topogps = QueryConfig(**{**base_qcfg.__dict__})
    qcfg_topogps.enable_cue_matching = True
    qcfg_topogps.cue_path_only = False
    qcfg_topogps.descent = base_qcfg.descent

    qcfg_cuepath = QueryConfig(**{**base_qcfg.__dict__})
    qcfg_cuepath.enable_cue_matching = True
    qcfg_cuepath.cue_path_only = True
    qcfg_cuepath.descent = base_qcfg.descent

    qcfg_no_cue = QueryConfig(**{**base_qcfg.__dict__})
    qcfg_no_cue.enable_cue_matching = False
    qcfg_no_cue.cue_path_only = False
    qcfg_no_cue.descent = base_qcfg.descent

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    rng = np.random.default_rng(args.seed + 777)
    queries = gen_queries(rng, clusters=args.clusters, per_cluster=args.per_cluster, n_queries=int(args.n_queries), task=args.task)
    write_jsonl(outdir / "queries.jsonl", queries)

    rows: List[Dict[str, object]] = []
    qnoise_rng = np.random.default_rng(args.seed + 999)

    for qobj in queries:
        qid = str(qobj["id"])
        cue_a = str(qobj["cue_a"])
        cue_b = str(qobj["cue_b"])
        expected = str(qobj["expected"])
        qstr = str(qobj["query"])

        ia = label_to_idx[cue_a]
        ib = label_to_idx[cue_b]
        ie = label_to_idx[expected]

        qv = E[ia] + float(args.cue_b_mix) * E[ib] + float(args.query_noise) * qnoise_rng.standard_normal((args.dim,)).astype(np.float32)
        qv = l2_normalize(qv.astype(np.float32), axis=0).astype(np.float32)

        for m in methods:
            t0 = time.perf_counter()
            pred_idx = None
            coarse_visited = 0
            fine_steps = 0

            if m == "nn":
                pred_idx = nn_pred(E, qv)
            elif m == "minsim":
                pred_idx = minsim_pred(E, ia, ib)
            elif m == "graph_ppr":
                pred_idx = graph_ppr_pred(ws.graph, ia, ib)
            elif m == "topogps":
                res = TopoGPS.query_vec(ws, qv, query=qstr, cfg=qcfg_topogps, top_k=1, emit_fine_steps=False)
                pred_idx = int(res.final_idx)
                coarse_visited = int(res.coarse.visited)
                fine_steps = int(res.fine_steps)
            elif m == "topogps_cuepath":
                res = TopoGPS.query_vec(ws, qv, query=qstr, cfg=qcfg_cuepath, top_k=1, emit_fine_steps=False)
                pred_idx = int(res.final_idx)
                coarse_visited = int(res.coarse.visited)
                fine_steps = int(res.fine_steps)
            elif m == "topogps_no_cue":
                res = TopoGPS.query_vec(ws, qv, query=qstr, cfg=qcfg_no_cue, top_k=1, emit_fine_steps=False)
                pred_idx = int(res.final_idx)
                coarse_visited = int(res.coarse.visited)
                fine_steps = int(res.fine_steps)
            else:
                raise SystemExit(f"Unknown method: {m}")

            dt_ms = (time.perf_counter() - t0) * 1000.0
            pred_lab = labels[int(pred_idx)]
            success = int(int(pred_idx) == ie)

            rows.append(
                {
                    "id": qid,
                    "task": args.task,
                    "method": m,
                    "cue_a": cue_a,
                    "cue_b": cue_b,
                    "expected": expected,
                    "pred": pred_lab,
                    "success": success,
                    "coarse_visited": coarse_visited,
                    "fine_steps": fine_steps,
                    "runtime_ms": round(dt_ms, 4),
                    "candidates": int(args.candidates),
                    "grid": int(bool(args.grid)),
                    "seed": int(args.seed),
                    "n_queries": int(args.n_queries),
                }
            )

    results_csv = outdir / "results.csv"
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_method: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(r)

    print(f"\nWrote: {results_csv}  (task={args.task})")
    for m, rr in by_method.items():
        acc = sum(int(x["success"]) for x in rr) / max(1, len(rr))
        avg_rt = sum(float(x["runtime_ms"]) for x in rr) / max(1, len(rr))
        avg_cv = sum(int(x["coarse_visited"]) for x in rr) / max(1, len(rr))
        avg_fs = sum(int(x["fine_steps"]) for x in rr) / max(1, len(rr))
        print(f"{m:16s} acc={acc:.3f}  rt_ms={avg_rt:.2f}  coarse={avg_cv:.1f}  fine={avg_fs:.1f}")

    print(f"\nIndex dir: {index_dir}")
    print(f"Queries:   {outdir / 'queries.jsonl'}")
    print("Next: python scripts/make_figures.py --run <outdir> --make-plots --make-html\n")


if __name__ == "__main__":
    main()