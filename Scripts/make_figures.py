from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from topogps.core import TopoGPS
from topogps.visualize import render_html


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper-ready figures from bench results.")
    p.add_argument("--run", type=Path, default=Path("results/run"), help="Bench run directory")
    p.add_argument("--results", type=Path, default=None, help="Override results.csv path")
    p.add_argument("--out", type=Path, default=None, help="Output figures dir (default: <run>/figures)")
    p.add_argument("--make-plots", action="store_true", help="Generate png plots (requires matplotlib)")
    p.add_argument("--make-html", action="store_true", help="Generate HTML path maps")
    return p.parse_args()


def read_results_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def summarize(rows: List[Dict[str, str]]) -> List[Tuple[str, Dict[str, float]]]:
    by_method: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    out: List[Tuple[str, Dict[str, float]]] = []
    for m, rr in sorted(by_method.items()):
        acc = sum(int(x["success"]) for x in rr) / max(1, len(rr))
        rt = sum(float(x["runtime_ms"]) for x in rr) / max(1, len(rr))
        cv = sum(int(x["coarse_visited"]) for x in rr) / max(1, len(rr))
        fs = sum(int(x["fine_steps"]) for x in rr) / max(1, len(rr))
        out.append(
            (
                m,
                {
                    "n": float(len(rr)),
                    "acc": float(acc),
                    "runtime_ms": float(rt),
                    "coarse_visited": float(cv),
                    "fine_steps": float(fs),
                },
            )
        )
    return out


def write_summary_md(path: Path, summary: List[Tuple[str, Dict[str, float]]]) -> None:
    lines = []
    lines.append("# TopoGPS Benchmark Summary\n\n")
    lines.append("| method | n | accuracy | avg runtime (ms) | avg coarse visited | avg fine steps |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for m, s in summary:
        lines.append(
            f"| `{m}` | {int(s['n'])} | {s['acc']:.3f} | {s['runtime_ms']:.2f} | {s['coarse_visited']:.1f} | {s['fine_steps']:.1f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def make_plots(figdir: Path, rows: List[Dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping plots. Install with: python -m pip install matplotlib")
        return

    methods = sorted(set(r["method"] for r in rows))
    accs = []
    for m in methods:
        rr = [x for x in rows if x["method"] == m]
        accs.append(sum(int(x["success"]) for x in rr) / max(1, len(rr)))

    plt.figure()
    plt.bar(methods, accs)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Bridge-query accuracy by method")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figdir / "bridge_accuracy.png", dpi=200)
    plt.close()

    # latency proxies scatter: show internal-dynamics methods only
    dyn = [r for r in rows if r["method"].startswith("topogps")]
    if dyn:
        x1 = [int(r["coarse_visited"]) for r in dyn if int(r["success"]) == 1]
        y1 = [int(r["fine_steps"]) for r in dyn if int(r["success"]) == 1]
        x0 = [int(r["coarse_visited"]) for r in dyn if int(r["success"]) == 0]
        y0 = [int(r["fine_steps"]) for r in dyn if int(r["success"]) == 0]

        plt.figure()
        if x0:
            plt.scatter(x0, y0, label="fail")
        if x1:
            plt.scatter(x1, y1, label="success")
        plt.xlabel("coarse_visited (proxy for search effort)")
        plt.ylabel("fine_steps (proxy for settling time)")
        plt.title("Latency proxies (TopoGPS variants)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / "latency_proxies.png", dpi=200)
        plt.close()

    print(f"Wrote plots into: {figdir}")


def make_html_maps(run_dir: Path, figdir: Path) -> None:
    index_dir = run_dir / "index"
    if not index_dir.exists():
        print(f"Missing index dir: {index_dir} (run bench.py first)")
        return

    ws = TopoGPS.load(index_dir)

    # Representative queries that always exist with our generator (same index on two clusters)
    queries = [
        "c0_000 and c1_000",
        "c1_015 and c3_015",
    ]

    for i, q in enumerate(queries, start=1):
        out = figdir / f"map_path_{i}.html"
        render_html(ws, out=out, query=q, cue_matching=True, umap_seed=42)
        print(f"Wrote: {out}")


def main() -> None:
    args = parse_args()
    run_dir = args.run
    results_csv = args.results if args.results is not None else (run_dir / "results.csv")
    figdir = args.out if args.out is not None else (run_dir / "figures")
    figdir.mkdir(parents=True, exist_ok=True)

    rows = read_results_csv(results_csv)
    summ = summarize(rows)

    summary_md = figdir / "summary.md"
    write_summary_md(summary_md, summ)
    print(f"Wrote: {summary_md}")

    if args.make_plots:
        make_plots(figdir, rows)

    if args.make_html:
        make_html_maps(run_dir, figdir)

    print("Done.")


if __name__ == "__main__":
    main()