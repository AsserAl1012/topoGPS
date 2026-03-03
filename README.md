# TopoGPS (prototype repo)

A working, testable implementation of the **TopoGPS** retrieval stack described in `topo.pdf`:

- Continuous semantic manifold **M** via sentence embeddings
- Landmark memory **L** stored in a **FAISS** vector index
- Associative topology **G** built as a weighted **NetworkX** graph
- Two-stage retrieval:
  1) **coarse** graph-guided navigation
  2) **fine** attractor settling by gradient descent in latent space
- Optional 3D projection + interactive visualization (UMAP + Plotly)
- Full trace logging for replay/debugging

## Quick start

### 1) Install

```bash
cd topogps_repo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

If you prefer a plain requirements install:

```bash
pip install -r requirements.txt
pip install -e .
```

### 2) Create a workspace

```bash
topogps init workspace
```

This creates:

- `workspace/concepts.txt` (starter concepts)
- `workspace/index/` (built artifacts will go here)

### 3) Build the map (embeddings + FAISS + graph)

```bash
topogps build workspace/concepts.txt workspace/index \
  --model all-mpnet-base-v2 \
  --graph-knn 12 \
  --graph-min-sim 0.55
```

### 4) Query + trace

```bash
topogps query "black hole entropy" workspace/index --trace-out workspace/trace.jsonl
```

### 5) Visualize in 3D (HTML)

```bash
topogps visualize workspace/index --query "black hole entropy" --out workspace/map.html
```

Open the generated HTML in a browser.

## Repo layout

```
src/topogps/
  cli.py           # Typer CLI
  core.py          # TopoGPS orchestrator
  embeddings.py    # encoder + caching
  store.py         # persistence format (index/graph/metadata)
  graph.py         # graph build + coarse navigation
  energy.py        # attractor energy + gradient descent
  constraints.py   # optional symbolic constraints C(z;q)
  visualize.py     # UMAP + Plotly 3D scatter + path overlay
  trace.py         # JSONL trace events
  utils.py         # math helpers

data/
  sample_concepts.txt

tests/
  test_build_and_query.py
```

## Notes / design choices

- **Landmarks** are embedding vectors with an isotropic basin `sigma` by default.
- **Cue activations** `alpha_i(q)` are normalized cosine similarities between query and landmarks.
- **Coarse search** is a bounded A* over the associative graph with a cue-aware heuristic.
- **Fine search** runs gradient descent on a mixture-of-Gaussians energy surface.

## Determinism

- For strict reproducibility, pass `--seed` to `build` and `query`.

## Next upgrades (if you want)

- learnable sigmas / Mahalanobis metric per region
- learned cue-to-activation network instead of cosine
- grid-module phase simulation (Φ) + drift correction
- bigger datasets + streaming build
- a small dashboard (stream trace + 3D map)
