#!/usr/bin/env python3
"""
Scripts/run_experiments.py

One-file orchestrator that runs *all* TopoGPS paper experiments (multi-seed, multi-variant),
stores everything under ./experiments/, and writes aggregated summary tables you can paste
directly into the paper.

Design goals:
- Single entrypoint for running all benchmarks + ablations + corruption sweeps.
- Deterministic: each leaf run has an explicit seed and a frozen config snapshot.
- Reproducible outputs: raw per-query CSVs + per-seed summaries + cross-seed aggregates.
- Works even if internal APIs change: synthetic benches reuse existing Scripts/bench.py by default.
- Adds the missing "WordNet under topology corruption" benchmark (defined on true graph,
  executed on corrupted graph) in-file, no repo edits needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------
# Utilities
# ---------------------------

def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _short_hash(obj: Any, n: int = 10) -> str:
    h = hashlib.sha1(_stable_json(obj).encode("utf-8")).hexdigest()
    return h[:n]


def _slug(s: str) -> str:
    keep = []
    for ch in s.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("-")
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:80] if out else "run"


def _git_info(repo_root: Path) -> Dict[str, Any]:
    def _run(args: List[str]) -> Optional[str]:
        try:
            r = subprocess.run(
                args,
                cwd=str(repo_root),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if r.returncode != 0:
                return None
            return r.stdout.strip()
        except Exception:
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "head": head,
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _summarize_results_rows(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    """
    Expects per-query rows with at least:
      method, success, rt_ms, coarse_visited, fine_steps
    Returns per-method means.
    """
    by: Dict[str, Dict[str, float]] = {}
    n: Dict[str, int] = {}
    for row in rows:
        m = row["method"]
        n[m] = n.get(m, 0) + 1
        if m not in by:
            by[m] = {"acc": 0.0, "rt_ms": 0.0, "coarse_visited": 0.0, "fine_steps": 0.0}
        by[m]["acc"] += float(row.get("success", "0"))
        by[m]["rt_ms"] += float(row.get("rt_ms", "0"))
        by[m]["coarse_visited"] += float(row.get("coarse_visited", "0"))
        by[m]["fine_steps"] += float(row.get("fine_steps", "0"))
    for m, s in by.items():
        denom = max(1, n[m])
        s["acc"] /= denom
        s["rt_ms"] /= denom
        s["coarse_visited"] /= denom
        s["fine_steps"] /= denom
    return by


def _aggregate_across_seeds(seed_summaries: List[Tuple[int, Dict[str, Dict[str, float]]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    seed_summaries: [(seed, per_method_metrics)]
    Returns:
      method -> metric -> {mean, std}
    """
    if not seed_summaries:
        return {}
    methods = sorted({m for _, sm in seed_summaries for m in sm.keys()})
    metrics = ["acc", "rt_ms", "coarse_visited", "fine_steps"]
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in methods:
        out[m] = {}
        for k in metrics:
            vals = [sm[m][k] for _, sm in seed_summaries if m in sm]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            out[m][k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size >= 2 else 0.0,
                "n": float(arr.size),
            }
    return out


def _write_aggregate_csv(path: Path, agg: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    # rows: method, metric, mean, std, n
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "metric", "mean", "std", "n"])
        w.writeheader()
        for method in sorted(agg.keys()):
            for metric in sorted(agg[method].keys()):
                s = agg[method][metric]
                w.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "mean": f"{s['mean']:.6f}",
                        "std": f"{s['std']:.6f}",
                        "n": int(s["n"]),
                    }
                )


def _run_subprocess(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
) -> int:
    stdout_f = None
    stderr_f = None
    try:
        if stdout_path is not None:
            _ensure_dir(stdout_path.parent)
            stdout_f = stdout_path.open("w", encoding="utf-8")
        if stderr_path is not None:
            _ensure_dir(stderr_path.parent)
            stderr_f = stderr_path.open("w", encoding="utf-8")

        r = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdout=stdout_f if stdout_f is not None else None,
            stderr=stderr_f if stderr_f is not None else None,
            text=True,
        )
        return int(r.returncode)
    finally:
        if stdout_f is not None:
            stdout_f.close()
        if stderr_f is not None:
            stderr_f.close()


def _dict_cartesian_product(matrix: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    matrix = {"a":[1,2], "b":[3,4]} -> [{"a":1,"b":3}, {"a":1,"b":4}, ...]
    """
    if not matrix:
        return [{}]
    keys = list(matrix.keys())
    values = [matrix[k] for k in keys]
    out: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        d = {}
        for k, v in zip(keys, combo):
            d[k] = v
        out.append(d)
    return out


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    out.update(b)
    return out


# ---------------------------
# Plan schema
# ---------------------------

@dataclass(frozen=True)
class Job:
    kind: str  # "synthetic", "wordnet", "wordnet_corrupt"
    name: str
    seeds: List[int]
    base: Dict[str, Any]
    matrix: Dict[str, List[Any]]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Job":
        return Job(
            kind=str(d["kind"]),
            name=str(d.get("name") or d["kind"]),
            seeds=[int(x) for x in d.get("seeds", [])],
            base=dict(d.get("base", {})),
            matrix={str(k): list(v) for k, v in (d.get("matrix") or {}).items()},
        )


def _default_plan_paper_v1() -> Dict[str, Any]:
    """
    Minimal "paper v1" plan:
    - Synthetic bridge + associative, 5 seeds, descent steps {0, 120}, grid {off,on}
    - WordNet corruption sweep (true targets, corrupted topology), 3 seeds, dropout {0, .3, .5, .7}
      run with descent steps {0, 40} to demonstrate when fine descent helps.
    """
    return {
        "version": 1,
        "jobs": [
            {
                "kind": "synthetic",
                "name": "synthetic_bridge",
                "seeds": [0, 1, 2, 3, 4],
                "base": {
                    "task": "bridge",
                    "n_clusters": 5,
                    "per_cluster": 500,
                    "dim": 64,
                    "graph_knn": 12,
                    "graph_min_sim": 0.55,
                    "max_expansions": 2500,
                    "max_depth": 4,
                    "candidates": 256,
                    "n_queries": 2000,
                    "lr": 0.08,
                },
                "matrix": {
                    "grid": [0, 1],
                    "steps": [0, 120],
                },
            },
            {
                "kind": "synthetic",
                "name": "synthetic_associative",
                "seeds": [0, 1, 2, 3, 4],
                "base": {
                    "task": "associative",
                    "n_clusters": 5,
                    "per_cluster": 500,
                    "dim": 64,
                    "graph_knn": 12,
                    "graph_min_sim": 0.55,
                    "max_expansions": 2500,
                    "max_depth": 4,
                    "candidates": 256,
                    "n_queries": 2000,
                    "lr": 0.08,
                },
                "matrix": {
                    "grid": [0, 1],
                    "steps": [0, 120],
                },
            },
            {
                "kind": "wordnet_corrupt",
                "name": "wordnet_corrupt_dropout",
                "seeds": [0, 1, 2],
                "base": {
                    "model": "all-mpnet-base-v2",
                    "max_words": 2000,
                    "n_queries": 500,
                    "candidates": 256,
                    "rewire": 0.0,  # keep rewire off in v1; dropout alone tells the story
                    "lr": 0.08,
                },
                "matrix": {
                    "dropout": [0.0, 0.3, 0.5, 0.7],
                    "steps": [0, 40],
                    "cue_path_only": [0, 1],  # topogps vs topogps_cuepath mode at config-level
                },
            },
        ],
    }


def _load_plan(path: Optional[Path], preset: str) -> Dict[str, Any]:
    if path is None:
        if preset == "paper_v1":
            return _default_plan_paper_v1()
        if preset == "quick":
            p = _default_plan_paper_v1()
            # Trim for speed
            for j in p["jobs"]:
                j["seeds"] = j["seeds"][:2]
                if j["kind"] == "wordnet_corrupt":
                    j["base"]["max_words"] = 1000
                    j["base"]["n_queries"] = 200
                    j["matrix"]["dropout"] = [0.0, 0.5]
                    j["matrix"]["steps"] = [0, 40]
                    j["matrix"]["cue_path_only"] = [0]
            return p
        raise SystemExit(f"Unknown preset: {preset}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "jobs" not in data:
        raise SystemExit("Plan JSON must be an object with a top-level 'jobs' list.")
    return data


# ---------------------------
# External-script runners (synthetic + optional vanilla wordnet)
# ---------------------------

def _bench_py_args(params: Dict[str, Any], outdir: Path, seed: int) -> List[str]:
    # bench.py uses args with underscores; flags are with dashes.
    def add(flag: str, value: Any) -> List[str]:
        return [f"--{flag.replace('_', '-')}", str(value)]

    args: List[str] = []
    args += add("outdir", str(outdir))
    args += add("task", params["task"])
    args += add("seed", seed)

    if int(params.get("grid", 0)) == 1:
        args.append("--grid")

    for key in [
        "candidates",
        "n_queries",
        "n_clusters",
        "per_cluster",
        "dim",
        "graph_knn",
        "graph_min_sim",
        "max_expansions",
        "max_depth",
        "steps",
        "lr",
    ]:
        if key in params:
            args += add(key, params[key])

    return args


def _wordnet_bench_py_args(params: Dict[str, Any], outdir: Path, seed: int) -> List[str]:
    def add(flag: str, value: Any) -> List[str]:
        return [f"--{flag.replace('_', '-')}", str(value)]

    args: List[str] = []
    args += add("outdir", str(outdir))
    args += add("seed", seed)
    for key in ["max_words", "n_queries", "model", "candidates"]:
        if key in params:
            args += add(key, params[key])
    return args


# ---------------------------
# WordNet corruption benchmark (in-file)
# ---------------------------

def _ensure_wordnet() -> None:
    try:
        import nltk  # type: ignore
        from nltk.corpus import wordnet as wn  # type: ignore  # noqa: F401

        _ = wn.synsets("dog")
    except Exception:
        # Only try downloads if nltk is installed.
        try:
            import nltk  # type: ignore

            nltk.download("wordnet")
            nltk.download("omw-1.4")
        except Exception as e:
            raise RuntimeError(
                "NLTK WordNet not available. Install optional deps: pip install -e \".[paper]\""
            ) from e


def _build_wordnet_graph(max_words: int, seed: int) -> Tuple[List[str], Any]:
    import networkx as nx  # local import to keep base runtime lean

    from nltk.corpus import wordnet as wn  # type: ignore

    rng = np.random.default_rng(seed)
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
        words = [lemmas[int(i)] for i in pick]
    else:
        words = lemmas

    idx = {w: i for i, w in enumerate(words)}
    G = nx.Graph()
    for w, i in idx.items():
        G.add_node(i, label=w)

    # synonym + hypernym edges
    for syn in wn.all_synsets(pos="n"):
        syn_words = [l.name().lower() for l in syn.lemmas()]
        syn_words = [w for w in syn_words if w in idx]
        # synonyms clique
        for i in range(len(syn_words)):
            for j in range(i + 1, len(syn_words)):
                G.add_edge(idx[syn_words[i]], idx[syn_words[j]], weight=1.0)
        # hypernym links
        for h in syn.hypernyms():
            h_words = [l.name().lower() for l in h.lemmas()]
            h_words = [w for w in h_words if w in idx]
            for w in syn_words:
                for hw in h_words:
                    G.add_edge(idx[w], idx[hw], weight=1.0)

    return words, G


def _gen_hypernym_queries(words: List[str], G_true: Any, n: int, seed: int) -> List[Tuple[int, int, int, str]]:
    rng = np.random.default_rng(seed)
    n_nodes = len(words)
    neigh = {i: list(G_true.neighbors(i)) for i in range(n_nodes)}
    queries: List[Tuple[int, int, int, str]] = []
    tries = 0
    while len(queries) < n and tries < n * 120:
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


def _corrupt_graph(G_true: Any, *, dropout: float, rewire: float, seed: int) -> Any:
    """
    Returns a new graph with:
      - edge dropout: remove each edge with probability dropout
      - edge rewire: additionally remove rewire*m edges (after dropout) and replace each with random edge
    """
    import networkx as nx

    rng = np.random.default_rng(seed)
    dropout = float(max(0.0, min(1.0, dropout)))
    rewire = float(max(0.0, min(1.0, rewire)))

    G = nx.Graph()
    G.add_nodes_from(G_true.nodes(data=True))

    edges = list(G_true.edges(data=True))
    # dropout
    kept: List[Tuple[int, int, Dict[str, Any]]] = []
    for u, v, d in edges:
        if rng.random() < dropout:
            continue
        kept.append((int(u), int(v), dict(d)))
    for u, v, d in kept:
        G.add_edge(u, v, **d)

    # rewire: remove a fraction of remaining edges and add random edges back
    cur_edges = list(G.edges())
    m = len(cur_edges)
    n_rewire = int(round(rewire * m))
    if n_rewire > 0 and m > 0:
        rng.shuffle(cur_edges)
        to_remove = cur_edges[:n_rewire]
        for u, v in to_remove:
            if G.has_edge(u, v):
                G.remove_edge(u, v)

        nodes = list(G.nodes())
        n_nodes = len(nodes)
        # add new random edges
        added = 0
        attempts = 0
        while added < n_rewire and attempts < n_rewire * 50:
            attempts += 1
            u = int(nodes[int(rng.integers(0, n_nodes))])
            v = int(nodes[int(rng.integers(0, n_nodes))])
            if u == v:
                continue
            if G.has_edge(u, v):
                continue
            G.add_edge(u, v, weight=1.0)
            added += 1

    return G


def _pagerank_pick(G: Any, a: int, b: int) -> int:
    import networkx as nx

    n = G.number_of_nodes()
    pers = {i: 0.0 for i in range(n)}
    pers[int(a)] = 0.5
    pers[int(b)] = 0.5
    pr = nx.pagerank(G, alpha=0.85, personalization=pers, weight="weight")
    best = None
    bestv = -1.0
    for i, v in pr.items():
        if i in (a, b):
            continue
        if v > bestv:
            bestv = float(v)
            best = int(i)
    return int(best) if best is not None else int(a)


def _run_wordnet_corrupt_case(
    *,
    outdir: Path,
    seed: int,
    model: str,
    max_words: int,
    n_queries: int,
    candidates: int,
    dropout: float,
    rewire: float,
    steps: int,
    lr: float,
    cue_path_only: bool,
) -> None:
    """
    Builds WordNet *true* graph, samples queries and targets on that graph.
    Then corrupts the graph and runs methods on the corrupted topology.
    """
    from topogps.core import BuildConfig, QueryConfig, TopoGPS
    from topogps.embeddings import encode_texts
    from topogps.utils import l2_normalize, seed_everything

    _ensure_wordnet()
    seed_everything(int(seed))

    outdir = _ensure_dir(outdir)
    idx_dir = _ensure_dir(outdir / "index")

    words, G_true = _build_wordnet_graph(max_words=int(max_words), seed=int(seed))
    if len(words) < 50:
        raise RuntimeError("Word list too small; increase max_words.")
    queries = _gen_hypernym_queries(words, G_true, int(n_queries), int(seed))
    if not queries:
        raise RuntimeError("Could not generate queries; increase max_words or n_queries.")

    # Embed once.
    _ = encode_texts(words[:2], model_name=model, normalize=True).astype(np.float32)  # warmup
    emb = encode_texts(words, model_name=model, normalize=True).astype(np.float32)

    # Build TopoGPS FAISS bundle once; we'll overwrite ws.graph for each corruption.
    cfg = BuildConfig(model_name=model, seed=int(seed), graph_knn=12, graph_min_sim=0.55)
    ws = TopoGPS.build_from_embeddings(labels=words, embeddings=emb, index_dir=idx_dir, cfg=cfg)

    # Corrupt topology.
    G_corrupt = _corrupt_graph(G_true, dropout=float(dropout), rewire=float(rewire), seed=int(seed) + 13)
    ws.graph = G_corrupt

    # Configure query (cue matching on; but allow "cue path only" at case level).
    qcfg = QueryConfig(
        enable_cue_matching=True,
        cue_path_only=bool(cue_path_only),
        faiss_candidates=int(candidates),
    )
    # Overwrite descent settings (steps=0 => no fine descent).
    qcfg.descent = qcfg.descent.__class__(lr=float(lr), steps=int(steps))

    qcfg_no_cue = QueryConfig(
        enable_cue_matching=False,
        cue_path_only=False,
        faiss_candidates=int(candidates),
    )
    qcfg_no_cue.descent = qcfg_no_cue.descent.__class__(lr=float(lr), steps=int(steps))

    def nn_pred(qv: np.ndarray) -> int:
        return int(np.argmax(ws.embeddings @ qv.reshape(-1)))

    methods = ["nn", "graph_ppr", "topogps", "topogps_no_cue"]

    rows: List[Dict[str, Any]] = []
    for qid, (a, b, t, qtext) in enumerate(queries):
        qv = l2_normalize((ws.embeddings[a] + ws.embeddings[b]) * 0.5, axis=0)
        for m in methods:
            t0 = time.perf_counter()
            if m == "nn":
                pred = nn_pred(qv)
                cv, fs = 0.0, 0.0
            elif m == "graph_ppr":
                pred = _pagerank_pick(G_corrupt, a, b)
                cv, fs = 0.0, 0.0
            elif m == "topogps":
                res = TopoGPS.query_vec(ws, qv, query_text=qtext, cfg=qcfg, emit_fine_steps=False)
                pred = int(res.final_idx)
                cv, fs = float(res.coarse.visited), float(res.fine_steps)
            elif m == "topogps_no_cue":
                res = TopoGPS.query_vec(ws, qv, query_text=qtext, cfg=qcfg_no_cue, emit_fine_steps=False)
                pred = int(res.final_idx)
                cv, fs = float(res.coarse.visited), float(res.fine_steps)
            else:
                raise ValueError(m)
            t1 = time.perf_counter()
            rt_ms = (t1 - t0) * 1000.0
            s = int(pred == t)
            rows.append(
                {
                    "qid": qid,
                    "method": m if m != "topogps" else ("topogps_cuepath" if cue_path_only else "topogps"),
                    "success": s,
                    "rt_ms": rt_ms,
                    "coarse_visited": cv,
                    "fine_steps": fs,
                    "cue_a": int(a),
                    "cue_b": int(b),
                    "target_idx": int(t),
                    "pred_idx": int(pred),
                    "dropout": float(dropout),
                    "rewire": float(rewire),
                    "steps": int(steps),
                    "lr": float(lr),
                }
            )

    # Write raw rows.
    out_csv = outdir / "results.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "qid",
                "method",
                "success",
                "rt_ms",
                "coarse_visited",
                "fine_steps",
                "cue_a",
                "cue_b",
                "target_idx",
                "pred_idx",
                "dropout",
                "rewire",
                "steps",
                "lr",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write query metadata (for optional tracing/visualization).
    q_jsonl = outdir / "queries.jsonl"
    with q_jsonl.open("w", encoding="utf-8") as f:
        for qid, (a, b, t, qtext) in enumerate(queries):
            qv = l2_normalize((ws.embeddings[a] + ws.embeddings[b]) * 0.5, axis=0)
            f.write(
                json.dumps(
                    {
                        "qid": int(qid),
                        "cue_a": int(a),
                        "cue_b": int(b),
                        "target_idx": int(t),
                        "query_text": qtext,
                        "q_vec": qv.astype(float).tolist(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------
# Main runner
# ---------------------------

def _resolve_repo_root() -> Path:
    # This file lives at <repo>/Scripts/run_experiments.py
    return Path(__file__).resolve().parents[1]


def _write_leaf_summary(leaf_dir: Path) -> Dict[str, Dict[str, float]]:
    results_csv = leaf_dir / "results.csv"
    rows = _read_csv_rows(results_csv)
    summary = _summarize_results_rows(rows)
    _write_json(leaf_dir / "summary.json", summary)
    # also write a small csv (method rows)
    with (leaf_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "acc", "rt_ms", "coarse_visited", "fine_steps"])
        w.writeheader()
        for m in sorted(summary.keys()):
            s = summary[m]
            w.writerow(
                {
                    "method": m,
                    "acc": f"{s['acc']:.6f}",
                    "rt_ms": f"{s['rt_ms']:.6f}",
                    "coarse_visited": f"{s['coarse_visited']:.6f}",
                    "fine_steps": f"{s['fine_steps']:.6f}",
                }
            )
    return summary


def _run_all(
    plan: Dict[str, Any],
    *,
    run_root: Path,
    repo_root: Path,
    postprocess: bool,
    make_plots: bool,
    make_html: bool,
    overwrite: bool,
) -> None:
    jobs = [Job.from_dict(j) for j in plan.get("jobs", [])]
    if not jobs:
        raise SystemExit("No jobs in plan.")

    scripts_dir = repo_root / "Scripts"
    bench_py = scripts_dir / "bench.py"
    wordnet_py = scripts_dir / "wordnet_bench.py"
    make_fig_py = scripts_dir / "make_figures.py"

    meta = {
        "created_utc": _now_utc_iso(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": _git_info(repo_root),
        "run_root": str(run_root),
        "plan": plan,
    }
    _write_json(run_root / "run_meta.json", meta)

    expanded_jobs: List[Dict[str, Any]] = []

    for job in jobs:
        variants = _dict_cartesian_product(job.matrix)
        if not job.seeds:
            raise SystemExit(f"Job '{job.name}' has no seeds.")
        if not variants:
            variants = [{}]

        for var in variants:
            params = _merge_dicts(job.base, var)
            case_id = f"{_slug(job.name)}__{_short_hash({'kind': job.kind, 'params': params})}"
            for seed in job.seeds:
                leaf_dir = run_root / job.kind / case_id / f"seed_{seed}"
                if leaf_dir.exists() and not overwrite:
                    if (leaf_dir / "results.csv").exists():
                        expanded_jobs.append(
                            {
                                "kind": job.kind,
                                "job": job.name,
                                "case_id": case_id,
                                "seed": seed,
                                "params": params,
                                "outdir": str(leaf_dir),
                                "skipped": True,
                            }
                        )
                        continue

                _ensure_dir(leaf_dir)
                _write_json(
                    leaf_dir / "case_params.json",
                    {"kind": job.kind, "job": job.name, "seed": seed, "params": params},
                )

                if job.kind == "synthetic":
                    if not bench_py.exists():
                        raise SystemExit(f"Missing Scripts/bench.py at {bench_py}")
                    cmd = [sys.executable, str(bench_py)] + _bench_py_args(params, leaf_dir, seed)
                    rc = _run_subprocess(
                        cmd,
                        cwd=repo_root,
                        stdout_path=leaf_dir / "stdout.log",
                        stderr_path=leaf_dir / "stderr.log",
                    )
                    if rc != 0:
                        raise SystemExit(f"bench.py failed (rc={rc}) for {leaf_dir}")

                    if postprocess and make_fig_py.exists():
                        cmd2 = [sys.executable, str(make_fig_py), "--run", str(leaf_dir)]
                        if make_plots:
                            cmd2.append("--make-plots")
                        if make_html:
                            cmd2.append("--make-html")
                        rc2 = _run_subprocess(
                            cmd2,
                            cwd=repo_root,
                            stdout_path=leaf_dir / "post_stdout.log",
                            stderr_path=leaf_dir / "post_stderr.log",
                        )
                        if rc2 != 0:
                            raise SystemExit(f"make_figures.py failed (rc={rc2}) for {leaf_dir}")

                elif job.kind == "wordnet":
                    if not wordnet_py.exists():
                        raise SystemExit(f"Missing Scripts/wordnet_bench.py at {wordnet_py}")
                    cmd = [sys.executable, str(wordnet_py)] + _wordnet_bench_py_args(params, leaf_dir, seed)
                    rc = _run_subprocess(
                        cmd,
                        cwd=repo_root,
                        stdout_path=leaf_dir / "stdout.log",
                        stderr_path=leaf_dir / "stderr.log",
                    )
                    if rc != 0:
                        raise SystemExit(f"wordnet_bench.py failed (rc={rc}) for {leaf_dir}")

                elif job.kind == "wordnet_corrupt":
                    _run_wordnet_corrupt_case(
                        outdir=leaf_dir,
                        seed=seed,
                        model=str(params.get("model", "all-mpnet-base-v2")),
                        max_words=int(params.get("max_words", 2000)),
                        n_queries=int(params.get("n_queries", 500)),
                        candidates=int(params.get("candidates", 256)),
                        dropout=float(params.get("dropout", 0.0)),
                        rewire=float(params.get("rewire", 0.0)),
                        steps=int(params.get("steps", 40)),
                        lr=float(params.get("lr", 0.08)),
                        cue_path_only=bool(int(params.get("cue_path_only", 0))),
                    )
                else:
                    raise SystemExit(f"Unknown job kind: {job.kind}")

                _write_leaf_summary(leaf_dir)

                expanded_jobs.append(
                    {
                        "kind": job.kind,
                        "job": job.name,
                        "case_id": case_id,
                        "seed": seed,
                        "params": params,
                        "outdir": str(leaf_dir),
                        "skipped": False,
                    }
                )

    _write_json(run_root / "plan_resolved.json", {"expanded": expanded_jobs})

    # Aggregate across seeds per case_id
    agg_root = _ensure_dir(run_root / "aggregate")
    groups: Dict[Tuple[str, str], List[Path]] = {}
    for item in expanded_jobs:
        kind = item["kind"]
        case_id = item["case_id"]
        leaf = Path(item["outdir"])
        if not (leaf / "summary.json").exists():
            continue
        groups.setdefault((kind, case_id), []).append(leaf)

    agg_index: List[Dict[str, Any]] = []
    for (kind, case_id), leaves in sorted(groups.items()):
        seed_summaries: List[Tuple[int, Dict[str, Dict[str, float]]]] = []
        for leaf in sorted(leaves):
            seed = int(leaf.name.split("_", 1)[1])
            sm = json.loads((leaf / "summary.json").read_text(encoding="utf-8"))
            seed_summaries.append((seed, sm))
        agg = _aggregate_across_seeds(seed_summaries)
        out_dir = _ensure_dir(agg_root / kind / case_id)
        _write_json(out_dir / "aggregate.json", agg)
        _write_aggregate_csv(out_dir / "aggregate.csv", agg)
        agg_index.append(
            {
                "kind": kind,
                "case_id": case_id,
                "n_seeds": len(seed_summaries),
                "outdir": str(out_dir),
            }
        )

    _write_json(agg_root / "INDEX.json", {"cases": agg_index})

    lines = []
    lines.append("# Experiments run summary\n\n")
    lines.append(f"- Run root: `{run_root}`\n")
    lines.append(f"- Created (UTC): `{meta['created_utc']}`\n")
    git = meta.get("git", {})
    if git.get("head"):
        lines.append(f"- Git HEAD: `{git['head']}`\n")
    lines.append(f"- Dirty: `{git.get('dirty', False)}`\n\n")
    lines.append("## Aggregates\n\n")
    lines.append("Each case has per-method mean/std across seeds:\n\n")
    lines.append("`experiments/<run>/aggregate/<kind>/<case_id>/aggregate.csv`\n\n")
    (run_root / "RUN_SUMMARY.md").write_text("".join(lines), encoding="utf-8")


def _make_run_id(tag: Optional[str]) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if tag:
        return f"{ts}__{_slug(tag)}"
    return ts


def write_sample_plan(path: Path) -> None:
    _write_json(path, _default_plan_paper_v1())


def main() -> None:
    ap = argparse.ArgumentParser(description="TopoGPS one-file experiment runner (writes into ./experiments/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run experiments from a preset or a plan JSON.")
    runp.add_argument("--preset", default="paper_v1", choices=["paper_v1", "quick"])
    runp.add_argument("--plan", type=str, default=None, help="Path to plan JSON (overrides --preset).")
    runp.add_argument("--tag", type=str, default=None, help="Optional run tag (added to folder name).")
    runp.add_argument("--outroot", type=str, default="experiments", help="Output root directory.")
    runp.add_argument("--postprocess", action="store_true", help="Run Scripts/make_figures.py after synthetic bench runs.")
    runp.add_argument("--make-plots", action="store_true", help="With --postprocess, generate matplotlib plots if installed.")
    runp.add_argument("--make-html", action="store_true", help="With --postprocess, generate HTML maps.")
    runp.add_argument("--overwrite", action="store_true", help="Overwrite existing leaf runs (otherwise skips if results.csv exists).")

    samp = sub.add_parser("write-sample-plan", help="Write the default paper_v1 plan JSON to a file.")
    samp.add_argument("--out", type=str, required=True)

    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if args.cmd == "write-sample-plan":
        out = Path(args.out)
        write_sample_plan(out)
        print(f"Wrote sample plan to: {out}")
        return

    if args.cmd == "run":
        plan_path = Path(args.plan) if args.plan else None
        plan = _load_plan(plan_path, args.preset)

        outroot = _ensure_dir(repo_root / str(args.outroot))
        run_id = _make_run_id(args.tag)
        run_root = _ensure_dir(outroot / run_id)

        _run_all(
            plan,
            run_root=run_root,
            repo_root=repo_root,
            postprocess=bool(args.postprocess),
            make_plots=bool(args.make_plots),
            make_html=bool(args.make_html),
            overwrite=bool(args.overwrite),
        )
        print(f"Done. Results in: {run_root}")
        print(f"Top summary: {run_root / 'RUN_SUMMARY.md'}")
        print(f"Aggregates: {run_root / 'aggregate'}")
        return


if __name__ == "__main__":
    main()