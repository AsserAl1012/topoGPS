# TopoGPS PDF-to-code mapping

This repo is a direct implementation of the retrieval pipeline in `topo.pdf`.

## Manifold M, metric g
- **embeddings.py**: sentence-embedding coordinates for concepts and cues.

## Landmarks L (place-cell analog)
- **store.py**: persistence of embeddings/labels/index
- **core.py**: build landmarks from concept list, store to workspace

## Associative topology G
- **graph.py**: kNN graph construction + cue-aware coarse search

## Energy landscape / attractor dynamics
- **energy.py**: mixture-of-Gaussians energy + closed-form gradient + descent loop

## Constraints C(z;q)
- **constraints.py**: lightweight symbolic filters/reweights over landmarks

## Grid-cell-like coordinate system Φ(z)
- **grid.py**: periodic phase code θ_m(z) = (A_m z) mod 2π, multi-module features

## 3D visualization & path replay
- **visualize.py**: UMAP 3D projection + Plotly HTML + overlayed path
- CLI: `topogps visualize ...`
