from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import networkx as nx
import numpy as np

from .constraints import ConstraintSpec, constraint_mask, constraint_soft_weights
from .embeddings import encode_texts
from .energy import DescentConfig, fine_descent
from .graph import CoarseResult, build_association_graph, coarse_search
from .grid import GridCode
from .manifold import SigmaKNNConfig, compute_local_sigmas
from .store import load_bundle, save_bundle
from .trace import TraceWriter
from .utils import batched_cosine_sims, read_lines, seed_everything


def _stable_softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    xx = (x.astype(np.float64) * float(temp)).copy()
    m = np.max(xx)
    if not np.isfinite(m):
        return np.zeros_like(x, dtype=np.float64)
    xx -= m
    np.exp(xx, out=xx)
    s = float(xx.sum())
    if s <= 0.0:
        return np.zeros_like(x, dtype=np.float64)
    return xx / s


def _softmax_alpha(
    scores: np.ndarray,
    *,
    temp: float,
    top_n: int,
    force_keep: Optional[List[int]] = None,
) -> np.ndarray:
    """Turn scores into sparse [0,1] weights."""
    n = int(scores.shape[0])
    top_n = int(max(1, min(int(top_n), n)))

    probs = _stable_softmax(scores, temp=temp)

    mask = np.zeros(n, dtype=bool)
    if top_n < n:
        keep = np.argpartition(probs, -top_n)[-top_n:]
        mask[keep] = True
    else:
        mask[:] = True

    if force_keep:
        for idx in force_keep:
            ii = int(idx)
            if 0 <= ii < n:
                mask[ii] = True

    alpha = probs * mask.astype(np.float64)
    mx = float(alpha.max())
    if mx <= 0.0:
        return np.zeros(n, dtype=np.float32)

    alpha = alpha / (mx + 1e-12)
    return alpha.astype(np.float32)


def _extract_cue_indices(query: str, labels: List[str], *, max_cues: int = 3) -> List[int]:
    """Greedy longest-match cue extraction."""
    q = query.lower()
    spans: List[Tuple[int, int, int]] = []
    for i, lab in enumerate(labels):
        s = q.find(lab.lower())
        if s >= 0:
            spans.append((s, s + len(lab), i))

    spans.sort(key=lambda t: (-(t[1] - t[0]), t[0]))

    chosen: List[Tuple[int, int, int]] = []
    for s, e, i in spans:
        overlap = False
        for cs, ce, _ in chosen:
            if not (e <= cs or s >= ce):
                overlap = True
                break
        if overlap:
            continue
        chosen.append((s, e, i))
        if len(chosen) >= max_cues:
            break

    chosen.sort(key=lambda t: t[0])
    return [i for _, _, i in chosen]


def _edge_cost(u: int, v: int, d: Dict[str, Any]) -> float:
    if "cost" in d:
        return float(d["cost"])
    w = float(d.get("weight", 0.0))
    return 1.0 / (w + 1e-9)


def _cue_bridge_score(
    *,
    query_sims: np.ndarray,
    cue_sims: np.ndarray,
    cue_idxs: List[int],
    cue_mix: float,
    cue_penalty: float,
    balance_penalty: float,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Joint score for multi-cue retrieval."""
    sim_a = cue_sims[:, 0]
    sim_b = cue_sims[:, 1]
    inter = np.minimum(sim_a, sim_b)
    imbalance = np.abs(sim_a - sim_b)
    bridge = inter - balance_penalty * imbalance

    score = (1.0 - cue_mix) * query_sims + cue_mix * bridge
    if cue_idxs:
        score = score.copy()
        score[cue_idxs] -= float(cue_penalty)

    score = score.copy()
    score[~valid_mask] = -1e9
    return score


@dataclass
class GridConfig:
    enabled: bool = True
    n_modules: int = 6
    d_per_module: int = 8
    lambdas: Tuple[float, ...] = (0.45, 0.75, 1.25, 2.1, 3.5, 5.8)
    seed: Optional[int] = None


@dataclass
class BuildConfig:
    model_name: str = "all-mpnet-base-v2"
    graph_knn: int = 12
    graph_min_sim: float = 0.55
    seed: int = 42
    normalize_embeddings: bool = True

    sigma_cfg: SigmaKNNConfig = field(default_factory=SigmaKNNConfig)
    grid: GridConfig = field(default_factory=GridConfig)


@dataclass
class QueryConfig:
    beta: float = 1.0
    max_expansions: int = 2500
    max_depth: int = 4

    sigma: float = 0.6
    use_local_sigmas: bool = True
    sigma_scale: float = 1.0

    descent: DescentConfig = field(default_factory=DescentConfig)

    alpha_temp: float = 8.0
    alpha_top_n: int = 64

    grid_sim_weight: float = 0.25
    grid_attractor_weight: float = 0.08

    soft_constraints: bool = False
    soft_constraint_mismatch_weight: float = 0.12
    soft_constraint_penalty: float = 0.35

    enable_cue_matching: bool = False
    direct_hit_cos: float = 0.9995


@dataclass
class QueryResult:
    query: str
    start_idx: int
    coarse: CoarseResult
    final_idx: int
    final_label: str
    topk: List[Tuple[str, float]]
    fine_steps: int
    fine_path_z: List[np.ndarray]


class TopoGPSWorkspace:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        (
            self.paths,
            self.embeddings,
            self.labels,
            self.faiss_index,
            self.graph,
            self.meta,
            self.sigmas,
            self.grid_feats,
        ) = load_bundle(self.index_dir)

        self.grid: Optional[GridCode] = None
        if bool(self.meta.get("grid_enabled", False)):
            dim = int(self.embeddings.shape[1])
            n_modules = int(self.meta.get("grid_n_modules", 6))
            d_per_module = int(self.meta.get("grid_d_per_module", 8))
            lambdas = self.meta.get("grid_lambdas", None)
            if isinstance(lambdas, list):
                lambdas_list = [float(x) for x in lambdas]
            elif isinstance(lambdas, tuple):
                lambdas_list = [float(x) for x in lambdas]
            else:
                lambdas_list = list(GridConfig().lambdas)

            seed = int(self.meta.get("grid_seed", self.meta.get("seed", 42)))
            self.grid = GridCode.random(
                D=dim,
                cfg={
                    "n_modules": n_modules,
                    "d_per_module": d_per_module,
                    "lambdas": lambdas_list,
                    "seed": seed,
                },
            )
            if self.grid_feats is None:
                self.grid_feats = self.grid.features(self.embeddings)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


class TopoGPS:
    @staticmethod
    def build(
        concepts_path: Path,
        index_dir: Path,
        *,
        cfg: BuildConfig = BuildConfig(),
        trace: Optional[TraceWriter] = None,
    ) -> TopoGPSWorkspace:
        seed_everything(cfg.seed)
        concepts = read_lines(Path(concepts_path))
        if len(concepts) < 2:
            raise ValueError("Need at least 2 concepts")

        if trace:
            trace.emit(
                "build_start",
                model_name=cfg.model_name,
                n=len(concepts),
                graph_knn=cfg.graph_knn,
                graph_min_sim=cfg.graph_min_sim,
                seed=cfg.seed,
                normalize_embeddings=cfg.normalize_embeddings,
            )

        emb = encode_texts(
            concepts,
            model_name=cfg.model_name,
            batch_size=64,
            normalize=cfg.normalize_embeddings,
        ).astype(np.float32)

        return TopoGPS.build_from_embeddings(labels=concepts, embeddings=emb, index_dir=index_dir, cfg=cfg, trace=trace)

    @staticmethod
    def build_from_embeddings(
        *,
        labels: List[str],
        embeddings: np.ndarray,
        index_dir: Path,
        cfg: Optional[BuildConfig] = None,
        model_name: str = "custom",
        graph_knn: int = 12,
        graph_min_sim: float = 0.55,
        seed: int = 42,
        normalized: bool = True,
        meta_extra: Optional[Dict[str, Any]] = None,
        trace: Optional[TraceWriter] = None,
    ) -> TopoGPSWorkspace:
        if cfg is None:
            cfg = BuildConfig(
                model_name=model_name,
                graph_knn=graph_knn,
                graph_min_sim=graph_min_sim,
                seed=seed,
                normalize_embeddings=normalized,
            )

        seed_everything(cfg.seed)
        if embeddings.shape[0] != len(labels):
            raise ValueError("labels/embeddings mismatch")

        emb = np.asarray(embeddings, dtype=np.float32)
        d = int(emb.shape[1])

        index = faiss.IndexFlatIP(d)
        index.add(emb)

        G = build_association_graph(emb, labels, knn=cfg.graph_knn, min_sim=cfg.graph_min_sim)

        sigmas = compute_local_sigmas(emb, cfg=cfg.sigma_cfg)

        grid_feats: Optional[np.ndarray] = None
        grid_enabled = bool(cfg.grid.enabled)
        if grid_enabled:
            grid_seed = int(cfg.grid.seed if cfg.grid.seed is not None else cfg.seed)
            grid = GridCode.random(
                D=d,
                cfg={
                    "n_modules": int(cfg.grid.n_modules),
                    "d_per_module": int(cfg.grid.d_per_module),
                    "lambdas": list(cfg.grid.lambdas),
                    "seed": grid_seed,
                },
            )
            grid_feats = grid.features(emb)

        meta: Dict[str, Any] = {
            "model_name": cfg.model_name,
            "graph_knn": cfg.graph_knn,
            "graph_min_sim": cfg.graph_min_sim,
            "seed": cfg.seed,
            "normalized": bool(cfg.normalize_embeddings),
            "dim": d,
            "sigma_knn": int(cfg.sigma_cfg.knn),
            "sigma_scale": float(cfg.sigma_cfg.scale),
            "sigma_min": float(cfg.sigma_cfg.min_sigma),
            "sigma_max": float(cfg.sigma_cfg.max_sigma),
            "grid_enabled": grid_enabled,
        }
        if grid_enabled:
            meta.update(
                {
                    "grid_n_modules": int(cfg.grid.n_modules),
                    "grid_d_per_module": int(cfg.grid.d_per_module),
                    "grid_lambdas": list(cfg.grid.lambdas),
                    "grid_seed": int(cfg.grid.seed if cfg.grid.seed is not None else cfg.seed),
                }
            )
        if meta_extra:
            meta.update(meta_extra)

        save_bundle(
            index_dir,
            embeddings=emb,
            labels=labels,
            index=index,
            graph=G,
            meta=meta,
            sigmas=sigmas,
            grid_feats=grid_feats,
        )

        if trace:
            trace.emit("build_done", index_dir=str(Path(index_dir).resolve()))

        return TopoGPSWorkspace(Path(index_dir))

    @staticmethod
    def load(index_dir: Path) -> TopoGPSWorkspace:
        return TopoGPSWorkspace(Path(index_dir))

    @staticmethod
    def query(
        ws: TopoGPSWorkspace,
        query: str,
        *,
        cfg: QueryConfig = QueryConfig(),
        constraints: ConstraintSpec = ConstraintSpec(),
        trace: Optional[TraceWriter] = None,
        seed: Optional[int] = None,
        top_k: int = 5,
        emit_fine_steps: bool = True,
    ) -> QueryResult:
        if seed is not None:
            seed_everything(seed)

        if query is None:
            query = query_text

        model_name = str(ws.meta.get("model_name", "all-mpnet-base-v2"))
        q_emb = encode_texts([query], model_name=model_name, normalize=True)
        q = q_emb[0].astype(np.float32)

        return TopoGPS.query_vec(
            ws,
            q,
            query=query,
            cfg=cfg,
            constraints=constraints,
            trace=trace,
            seed=None,
            top_k=top_k,
            emit_fine_steps=emit_fine_steps,
        )

    @staticmethod
    def query_vec(
        ws: TopoGPSWorkspace,
        query_vec: np.ndarray,
        *,
        # Back-compat: some callers use query_text=..., others use query=...
        query_text: str = "<vector>",
        query: Optional[str] = None,
        cfg: QueryConfig = QueryConfig(),
        constraints: ConstraintSpec = ConstraintSpec(),
        trace: Optional[TraceWriter] = None,
        seed: Optional[int] = None,
        top_k: int = 5,
        emit_fine_steps: bool = True,
    ) -> QueryResult:
        if seed is not None:
            seed_everything(seed)

        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)

        sem_sims = batched_cosine_sims(ws.embeddings, q).astype(np.float32)
        sims = sem_sims

        if cfg.grid_sim_weight > 0.0 and ws.grid is not None and ws.grid_feats is not None:
            qg = ws.grid.features(q)
            grid_sims = batched_cosine_sims(ws.grid_feats, qg).astype(np.float32)
            w = float(max(0.0, min(1.0, cfg.grid_sim_weight)))
            sims = ((1.0 - w) * sem_sims + w * grid_sims).astype(np.float32)

        valid_mask = np.ones(len(ws.labels), dtype=bool)
        penalty = np.zeros(len(ws.labels), dtype=np.float32)

        if cfg.soft_constraints:
            wts = constraint_soft_weights(ws.labels, constraints, mismatch_weight=float(cfg.soft_constraint_mismatch_weight))
            penalty = (1.0 - wts.astype(np.float32)) * float(cfg.soft_constraint_penalty)
        else:
            valid_mask = constraint_mask(ws.labels, constraints)

        masked_sims = sims.copy()
        masked_sims -= penalty
        masked_sims[~valid_mask] = -np.inf

        best_idx = int(np.argmax(masked_sims))
        best_sim = float(masked_sims[best_idx])

        top_idx = np.argsort(masked_sims)[::-1]
        topk: List[Tuple[str, float]] = []
        for idx in top_idx[: max(1, int(top_k))]:
            sc = float(masked_sims[int(idx)])
            if not np.isfinite(sc):
                continue
            topk.append((ws.labels[int(idx)], sc))

        if trace:
            trace.emit("query_start", query=query, best_idx=best_idx, best_sim=best_sim)

        if best_sim >= float(cfg.direct_hit_cos):
            coarse = CoarseResult(best_idx=best_idx, best_path=[best_idx], visited=1)
            if trace:
                trace.emit("direct_hit", idx=best_idx, label=ws.labels[best_idx], cos=best_sim)
            return QueryResult(
                query=query,
                start_idx=best_idx,
                coarse=coarse,
                final_idx=best_idx,
                final_label=ws.labels[best_idx],
                topk=topk,
                fine_steps=0,
                fine_path_z=[ws.embeddings[best_idx].copy()],
            )

        cue_idxs: List[int] = []
        if cfg.enable_cue_matching and query != "<vector>":
            cue_idxs = _extract_cue_indices(query, ws.labels)
            cue_idxs = [i for i in cue_idxs if valid_mask[i]]

        start_idx = best_idx
        cue_mix = 0.55
        final_mix = 0.25
        score_prior = masked_sims

        if len(cue_idxs) >= 2:
            src = int(cue_idxs[0])
            dst = int(cue_idxs[1])

            cue_sims = np.stack(
                [batched_cosine_sims(ws.embeddings, ws.embeddings[i]) for i in cue_idxs[:2]],
                axis=1,
            ).astype(np.float32)

            score_prior = _cue_bridge_score(
                query_sims=masked_sims,
                cue_sims=cue_sims,
                cue_idxs=cue_idxs[:2],
                cue_mix=cue_mix,
                cue_penalty=0.18,
                balance_penalty=0.20,
                valid_mask=valid_mask,
            )

            try:
                coarse_path = nx.shortest_path(ws.graph, source=src, target=dst, weight=_edge_cost)
            except Exception:
                coarse_path = [src]

            force_keep = list(dict.fromkeys([best_idx] + list(coarse_path) + cue_idxs[:2]))
            alpha = _softmax_alpha(score_prior, temp=cfg.alpha_temp, top_n=cfg.alpha_top_n, force_keep=force_keep)

            best_on_path = int(max(coarse_path, key=lambda i: float(score_prior[int(i)])))
            coarse_best = best_on_path
            if len(coarse_path) > 2:
                internal = [int(i) for i in coarse_path[1:-1]]
                best_internal = int(max(internal, key=lambda i: float(score_prior[int(i)])))
                if float(score_prior[best_internal]) >= float(score_prior[best_on_path]) - 0.03:
                    coarse_best = best_internal
                if coarse_best in cue_idxs[:2] and float(score_prior[best_internal]) >= float(score_prior[best_on_path]) - 0.08:
                    coarse_best = best_internal

            coarse = CoarseResult(best_idx=coarse_best, best_path=[int(i) for i in coarse_path], visited=len(coarse_path))
            start_idx = src

            if emit_fine_steps:
                fine_path_z = [ws.embeddings[int(i)].copy() for i in coarse_path]
            else:
                fine_path_z = [ws.embeddings[coarse_best].copy()]

            if trace:
                trace.emit(
                    "query_done",
                    start_idx=start_idx,
                    coarse_best=coarse.best_idx,
                    final_idx=coarse_best,
                    final_label=ws.labels[coarse_best],
                    fine_steps=max(0, len(fine_path_z) - 1),
                )

            return QueryResult(
                query=query,
                start_idx=start_idx,
                coarse=coarse,
                final_idx=coarse_best,
                final_label=ws.labels[coarse_best],
                topk=topk,
                fine_steps=max(0, len(fine_path_z) - 1),
                fine_path_z=fine_path_z,
            )

        alpha = _softmax_alpha(score_prior, temp=cfg.alpha_temp, top_n=cfg.alpha_top_n, force_keep=[best_idx])
        coarse = coarse_search(
            ws.graph,
            start_idx=best_idx,
            alpha=alpha,
            beta=cfg.beta,
            max_expansions=cfg.max_expansions,
            max_depth=cfg.max_depth,
        )

        if cfg.use_local_sigmas and ws.sigmas is not None:
            sigmas = (ws.sigmas.astype(np.float32) * float(cfg.sigma_scale)).astype(np.float32)
        else:
            sigmas = np.full((len(ws.labels),), float(cfg.sigma), dtype=np.float32)

        normalized = bool(ws.meta.get("normalized", True))
        dcfg = DescentConfig(
            lr=cfg.descent.lr,
            steps=cfg.descent.steps,
            tol_grad=cfg.descent.tol_grad,
            tol_move=cfg.descent.tol_move,
            clamp_norm=cfg.descent.clamp_norm,
            project_unit=normalized,
        )

        z_final, descent_trace = fine_descent(
            q,
            landmarks=ws.embeddings,
            alpha=alpha,
            sigmas=sigmas,
            cfg=dcfg,
            grid=ws.grid if (cfg.grid_attractor_weight > 0.0 and ws.grid is not None) else None,
            grid_landmarks=ws.grid_feats if (cfg.grid_attractor_weight > 0.0 and ws.grid_feats is not None) else None,
            grid_weight=float(cfg.grid_attractor_weight),
        )

        final_sem_sims = batched_cosine_sims(ws.embeddings, z_final).astype(np.float32)
        final_sims = final_sem_sims
        if ws.grid is not None and ws.grid_feats is not None and cfg.grid_sim_weight > 0.0:
            zg = ws.grid.features(z_final)
            grid_sims_z = batched_cosine_sims(ws.grid_feats, zg).astype(np.float32)
            w = float(max(0.0, min(1.0, cfg.grid_sim_weight)))
            final_sims = ((1.0 - w) * final_sem_sims + w * grid_sims_z).astype(np.float32)

        final_sims -= penalty
        final_sims[~valid_mask] = -np.inf

        final_scores = ((1.0 - final_mix) * final_sims + final_mix * score_prior).astype(np.float32)
        final_scores[~valid_mask] = -np.inf
        final_idx = int(np.argmax(final_scores))

        fallback_used = False
        direct_idx = best_idx
        if direct_idx != final_idx:
            direct_sim = float(masked_sims[direct_idx])
            final_query_sim = float(masked_sims[final_idx])
            if direct_sim - final_query_sim >= 0.06:
                final_idx = int(direct_idx)
                fallback_used = True

        fine_path_z: List[np.ndarray] = []
        if emit_fine_steps:
            fine_path_z = [t.z.copy() for t in descent_trace]
            if fallback_used:
                fine_path_z.append(ws.embeddings[final_idx].copy())

        if trace:
            trace.emit(
                "query_done",
                start_idx=start_idx,
                coarse_best=coarse.best_idx,
                final_idx=final_idx,
                final_label=ws.labels[final_idx],
                fine_steps=len(descent_trace),
                fallback_used=fallback_used,
            )

        return QueryResult(
            query=query,
            start_idx=start_idx,
            coarse=coarse,
            final_idx=final_idx,
            final_label=ws.labels[final_idx],
            topk=topk,
            fine_steps=len(descent_trace),
            fine_path_z=fine_path_z,
        )
