#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from topogps.core import QueryConfig, TopoGPS
from topogps.visualize import (
    UMAPConfig,
    coords_for_path_by_nearest,
    plot_map_3d,
    project_umap_3d,
    save_figure_html,
)


def _read_results_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _read_queries_jsonl(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _summarize(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    by_method: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for r in rows:
        m = r["method"]
        counts[m] = counts.get(m, 0) + 1
        if m not in by_method:
            by_method[m] = {"acc": 0.0, "rt_ms": 0.0, "coarse": 0.0, "fine": 0.0}

        by_method[m]["acc"] += float(r["success"])
        by_method[m]["rt_ms"] += float(r["rt_ms"])
        by_method[m]["coarse"] += float(r["coarse_visited"])
        by_method[m]["fine"] += float(r["fine_steps"])

    for m, s in by_method.items():
        n = max(1, counts[m])
        s["acc"] /= n
        s["rt_ms"] /= n
        s["coarse"] /= n
        s["fine"] /= n

    return by_method


def _write_summary_md(out: Path, summary: Dict[str, Dict[str, float]]) -> None:
    lines = []
    lines.append("# Run summary\n")
    lines.append("| method | acc | rt_ms | coarse_visited | fine_steps |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for m in sorted(summary.keys()):
        s = summary[m]
        lines.append(
            f"| {m} | {s['acc']:.3f} | {s['rt_ms']:.2f} | {s['coarse']:.1f} | {s['fine']:.1f} |\n"
        )
    out.write_text("".join(lines), encoding="utf-8")


def _maybe_make_plots(figdir: Path, summary: Dict[str, Dict[str, float]], rows: List[Dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping plots. Install with: pip install matplotlib")
        return

    # Accuracy bar
    methods = list(summary.keys())
    accs = [summary[m]["acc"] for m in methods]
    plt.figure()
    plt.title("Bridge-query accuracy by method")
    plt.bar(methods, accs)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Accuracy")
    out1 = figdir / "bridge_accuracy.png"
    plt.tight_layout()
    plt.savefig(out1)
    plt.close()

    # Latency proxy scatter (TopoGPS variants)
    xs = []
    ys = []
    cs = []
    for r in rows:
        if not r["method"].startswith("topogps"):
            continue
        xs.append(float(r["coarse_visited"]))
        ys.append(float(r["fine_steps"]))
        cs.append(int(r["success"]))
    plt.figure()
    plt.title("Latency proxies (TopoGPS variants)")
    plt.scatter(xs, ys, c=cs)
    plt.xlabel("coarse_visited (proxy for search effort)")
    plt.ylabel("fine_steps (proxy for settling time)")
    out2 = figdir / "latency_proxies.png"
    plt.tight_layout()
    plt.savefig(out2)
    plt.close()

    print(f"Wrote plots into: {figdir}")


def _make_html_maps(run_dir: Path, figdir: Path) -> None:
    idx_dir = run_dir / "index"
    qpath = run_dir / "queries.jsonl"
    if not qpath.exists():
        print("No queries.jsonl found; skipping HTML maps.")
        return

    ws = TopoGPS.load(idx_dir)
    coords = project_umap_3d(ws.embeddings, UMAPConfig(random_state=42))

    queries = _read_queries_jsonl(qpath)
    # pick 2 queries (first two)
    pick = queries[:2] if len(queries) >= 2 else queries[:1]

    for k, q in enumerate(pick, start=1):
        q_vec = np.asarray(q["q_vec"], dtype=np.float32)
        q_text = str(q.get("query_text", f"q{k}"))

        qcfg = QueryConfig(enable_cue_matching=True)
        res = TopoGPS.query_vec(ws, q_vec, query_text=q_text, cfg=qcfg, emit_fine_steps=True)

        path_coords = None
        if res.fine_path_z and len(res.fine_path_z) >= 2:
            path_coords = coords_for_path_by_nearest(res.fine_path_z, ws.embeddings, coords)

        title = f"TopoGPS semantic map — {q_text}"
        fig = plot_map_3d(coords, list(ws.labels), path_coords=path_coords, title=title)
        out = figdir / f"map_path_{k}.html"
        save_figure_html(fig, out)
        print(f"Wrote: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory containing results.csv and index/")
    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument("--make-html", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run)
    figdir = run_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    rows = _read_results_csv(run_dir / "results.csv")
    summary = _summarize(rows)

    summary_md = figdir / "summary.md"
    _write_summary_md(summary_md, summary)
    print(f"Wrote: {summary_md}")

    if args.make_plots:
        _maybe_make_plots(figdir, summary, rows)

    if args.make_html:
        _make_html_maps(run_dir, figdir)

    print("Done.")


if __name__ == "__main__":
    main()