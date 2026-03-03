from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .constraints import ConstraintSpec
from .core import BuildConfig, GridConfig, QueryConfig, TopoGPS
from .manifold import SigmaKNNConfig
from .trace import TraceWriter
from .utils import ensure_dir
from .visualize import UMAPConfig, coords_for_path_by_nearest, plot_map_3d, project_umap_3d, save_figure_html

app = typer.Typer(add_completion=False, help="TopoGPS prototype CLI")
console = Console()


@app.command()
def init(
    workspace: Path = typer.Argument(..., help="Workspace directory to create"),
) -> None:
    """Create a workspace with starter concepts."""
    workspace = Path(workspace)
    ensure_dir(workspace)
    ensure_dir(workspace / "index")

    concepts_path = workspace / "concepts.txt"
    sample = Path(__file__).resolve().parents[2] / "data" / "sample_concepts.txt"

    if not concepts_path.exists():
        if sample.exists():
            shutil.copyfile(sample, concepts_path)
        else:
            concepts_path.write_text("gravity\nblack hole\nentropy\n", encoding="utf-8")

    console.print(f"Workspace created: [bold]{workspace}[/bold]")
    console.print(f"Concepts: {concepts_path}")
    console.print(f"Index dir: {workspace / 'index'}")


@app.command()
def build(
    concepts: Path = typer.Argument(..., help="Text file with concepts (one per line; commas/whitespace supported)"),
    index_dir: Path = typer.Argument(..., help="Output directory for index artifacts"),
    model: str = typer.Option("all-mpnet-base-v2", help="SentenceTransformer model"),
    graph_knn: int = typer.Option(12, help="kNN degree for associative graph"),
    graph_min_sim: float = typer.Option(0.55, help="Minimum cosine similarity for an edge"),
    seed: int = typer.Option(42, help="Random seed"),
    no_normalize: bool = typer.Option(False, help="Disable embedding normalization"),
    # semantic manifold
    sigma_knn: int = typer.Option(12, "--sigma-knn", help="kNN for per-landmark sigma_i"),
    sigma_scale: float = typer.Option(1.25, "--sigma-scale", help="Scale factor for sigma_i"),
    sigma_min: float = typer.Option(0.05, "--sigma-min", help="Clamp min sigma_i"),
    sigma_max: float = typer.Option(1.25, "--sigma-max", help="Clamp max sigma_i"),
    # grid codes
    grid: bool = typer.Option(False, "--grid", help="Enable grid-code features"),
    grid_modules: int = typer.Option(6, "--grid-modules", help="Grid modules"),
    grid_dims: int = typer.Option(8, "--grid-dims", help="Grid dims per module"),
    grid_lambdas: str = typer.Option(
        "0.45,0.75,1.25,2.1,3.5,5.8",
        "--grid-lambdas",
        help="Comma-separated grid wavelengths",
    ),
    grid_seed: Optional[int] = typer.Option(None, "--grid-seed", help="Grid RNG seed (default: build seed)"),
    trace_out: Optional[Path] = typer.Option(None, help="Write build trace JSONL"),
) -> None:
    """Build embeddings, FAISS index, association graph, sigma_i, and optional grid codes."""
    lambdas = tuple(float(x.strip()) for x in str(grid_lambdas).split(",") if x.strip())
    cfg = BuildConfig(
        model_name=model,
        graph_knn=graph_knn,
        graph_min_sim=graph_min_sim,
        seed=seed,
        normalize_embeddings=not no_normalize,
        sigma_cfg=SigmaKNNConfig(knn=sigma_knn, scale=sigma_scale, min_sigma=sigma_min, max_sigma=sigma_max),
        grid=GridConfig(
            enabled=bool(grid),
            n_modules=int(grid_modules),
            d_per_module=int(grid_dims),
            lambdas=lambdas if lambdas else GridConfig().lambdas,
            seed=grid_seed,
        ),
    )

    trace_ctx = TraceWriter(trace_out) if trace_out else None
    try:
        ws = TopoGPS.build(concepts, index_dir, cfg=cfg, trace=trace_ctx)
    finally:
        if trace_ctx:
            trace_ctx.close()

    console.print(f"Built index: [bold]{index_dir}[/bold]")
    console.print(
        f"N landmarks: {len(ws.labels)} | dim: {ws.dim} | model: {ws.meta.get('model_name')} | "
        f"sigmas: {'yes' if ws.sigmas is not None else 'no'} | grid: {'yes' if ws.grid_feats is not None else 'no'}"
    )


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query/cue text"),
    index_dir: Path = typer.Argument(..., help="Index directory produced by build"),
    beta: float = typer.Option(1.0, help="Coarse search cue weight"),
    max_depth: int = typer.Option(4, help="Max graph depth (hops)"),
    max_expansions: int = typer.Option(2500, help="Max graph node expansions"),
    sigma: float = typer.Option(0.6, help="Fallback constant sigma"),
    use_local_sigmas: bool = typer.Option(True, help="Use per-landmark sigma_i if available"),
    sigma_scale: float = typer.Option(1.0, help="Multiply stored sigma_i by this"),
    steps: int = typer.Option(120, help="Fine descent steps"),
    lr: float = typer.Option(0.1, help="Fine descent learning rate"),
    seed: Optional[int] = typer.Option(None, help="Seed for determinism"),
    # NEW: candidate restriction
    candidates: int = typer.Option(256, "--candidates", help="FAISS top-M candidates for similarity (0 disables)"),
    # constraints
    starts_with: Optional[str] = typer.Option(None, help="Constraint: label starts with"),
    contains: Optional[str] = typer.Option(None, help="Constraint: label contains"),
    # misc
    cue_matching: bool = typer.Option(False, "--cue-matching", help="Enable multi-cue literal matching"),
    top_k: int = typer.Option(5, help="Show top-k by (possibly mixed) similarity"),
    trace_out: Optional[Path] = typer.Option(None, help="Write full trace JSONL"),
) -> None:
    """Run TopoGPS retrieval (coarse + fine) and print result."""
    ws = TopoGPS.load(index_dir)
    qcfg = QueryConfig(
        beta=beta,
        max_depth=max_depth,
        max_expansions=max_expansions,
        sigma=sigma,
        use_local_sigmas=use_local_sigmas,
        sigma_scale=sigma_scale,
        enable_cue_matching=cue_matching,
        faiss_candidates=int(max(0, candidates)),
    )
    qcfg.descent = qcfg.descent.__class__(lr=lr, steps=steps)

    c = ConstraintSpec(starts_with=starts_with, contains=contains)

    trace_ctx = TraceWriter(trace_out) if trace_out else None
    try:
        res = TopoGPS.query(ws, query_text, cfg=qcfg, constraints=c, trace=trace_ctx, seed=seed, top_k=top_k)
    finally:
        if trace_ctx:
            trace_ctx.close()

    console.print(f"\n[bold]Result[/bold]: {res.final_label} (idx={res.final_idx})")
    console.print(f"Start: {ws.labels[res.start_idx]} (idx={res.start_idx})")
    console.print(f"Coarse best: {ws.labels[res.coarse.best_idx]} (visited={res.coarse.visited})")
    console.print(f"Coarse path: {res.coarse.best_path}")
    console.print(f"Fine steps: {res.fine_steps}\n")

    table = Table(title="Top-K")
    table.add_column("label")
    table.add_column("score", justify="right")
    for lab, sc in res.topk:
        table.add_row(lab, f"{sc:.4f}")
    console.print(table)


@app.command()
def visualize(
    index_dir: Path = typer.Argument(..., help="Index directory produced by build"),
    out: Path = typer.Option(Path("map.html"), help="Output HTML"),
    reduce: str = typer.Option(
        "umap",
        "--reduce",
        help="3D reducer for the map. Supported: umap (default), none (use first 3 embedding dims).",
    ),
    query_text: Optional[str] = typer.Option(None, "--query", help="If set, overlay retrieval path"),
    cue_matching: bool = typer.Option(False, "--cue-matching", help="Use cue matching for overlay query"),
    n_neighbors: int = typer.Option(15, help="UMAP n_neighbors"),
    min_dist: float = typer.Option(0.1, help="UMAP min_dist"),
    seed: int = typer.Option(42, help="UMAP random_state"),
) -> None:
    """Generate an interactive 3D HTML map (optionally with a query path)."""
    ws = TopoGPS.load(index_dir)
    r = (reduce or "umap").strip().lower()

    if r == "umap":
        coords = project_umap_3d(
            ws.embeddings,
            UMAPConfig(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed),
        )
    elif r in ("none", "identity", "raw"):
        E = np.asarray(ws.embeddings, dtype=np.float32)
        if E.shape[1] < 3:
            raise typer.BadParameter(f"Embeddings dim={E.shape[1]} < 3; cannot use --reduce none")
        coords = E[:, :3].copy()
    else:
        raise typer.BadParameter("--reduce must be one of: umap, none")

    path_coords = None
    title = "TopoGPS semantic map"

    if query_text:
        qcfg = QueryConfig(enable_cue_matching=cue_matching)
        res = TopoGPS.query(ws, query_text, cfg=qcfg, emit_fine_steps=True)
        if res.fine_path_z:
            path_coords = coords_for_path_by_nearest(res.fine_path_z, ws.embeddings, coords)
        title = f"TopoGPS map + path: {query_text}"

    fig = plot_map_3d(coords, ws.labels, path_coords=path_coords, title=title)
    save_figure_html(fig, out)
    console.print(f"Wrote: [bold]{out}[/bold]")


if __name__ == "__main__":
    app()