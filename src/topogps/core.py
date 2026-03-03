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
from .utils import batched_cosine_sims, l2_normalize, read_lines, seed_everything


def _stable_softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    xx = (np.asarray(x, dtype=np.float64) * float(temp)).copy()
    m = np.max(xx)
    if not np.isfinite(m):
        return np.zeros_like(xx, dtype=np.float64)
    xx -= m
    np.exp(xx, out=xx)
    s = float(xx.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.zeros_like(xx, dtype=np.float64)
    return xx / s


def _softmax_alpha(
    scores: np.ndarray,
    *,
    temp: float,
    top_n: int,
    force_keep: Optional[List[int]] = None,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = int(scores.shape[0])
    top_n = int(max(1, min(int(top_n), n)))

    probs = _stable_softmax(scores, temp=temp).astype(np.float64)

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
    mx = float(alpha.max()) if alpha.size else 0.0
    if mx <= 0.0 or not np.isfinite(mx):
        return np.zeros(n, dtype=np.float32)

    alpha = alpha / (mx + 1e-12)
    return alpha.astype(np.float32)


def _is_word_boundary(q: str, start: int, end: int) -> bool:
    """
    Require non-alnum on both sides to avoid substring cue matches
    (e.g., 'man' accidentally matching inside 'human').
    """
    if start > 0 and q[start - 1].isalnum():
        return False
    if end < len(q) and q[end].isalnum():
        return False
    return True


def _extract_cue_indices(query: str, labels: List[str], *, max_cues: int = 3) -> List[int]:
    q = (query or "").lower()
    spans: List[Tuple[int, int, int]] = []

    for i, lab in enumerate(labels):
        lab_l = lab.lower()
        if not lab_l:
            continue

        # find ALL occurrences, keep only word-boundary matches
        pos = 0
        while True:
            s = q.find(lab_l, pos)
            if s < 0:
                break
            e = s + len(lab_l)
            if _is_word_boundary(q, s, e):
                spans.append((s, e, i))
            pos = s + 1

    # prefer longer matches, then earlier
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


def _edge_cost(_: int, __: int, d: Dict[str, Any]) -> float:
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

    descent: DescentConfig = field(default_factory=lambda: DescentConfig())

    alpha_temp: float = 8.0
    alpha_top_n: int = 64

    grid_sim_weight: float = 0.25
    grid_attractor_weight: float = 0.08

    soft_constraints: bool = False
    soft_constraint_mismatch_weight: float = 0.12
    soft_constraint_penalty: float = 0.35

    enable_cue_matching: bool = False
    cue_path_only: bool = False

    direct_hit_cos: float = 0.9995
    faiss_candidates: int = 256


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

        emb = encode_texts(
            concepts,
            model_name=cfg.model_name,
            batch_size=64,
            normalize=cfg.normalize_embeddings,
        ).astype(np.float32)

        return TopoGPS.build_from_embeddings(
            labels=concepts,
            embeddings=emb,
            index_dir=index_dir,
            cfg=cfg,
            trace=trace,
        )

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

        qstr = str(query or "")
        model_name = str(ws.meta.get("model_name", "all-mpnet-base-v2"))
        normalized = bool(ws.meta.get("normalized", True))

        q_emb = encode_texts([qstr], model_name=model_name, normalize=normalized).astype(np.float32)
        q = q_emb[0].reshape(-1)
        return TopoGPS.query_vec(
            ws,
            q,
            query_text=qstr,
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
        query_text: str = "",
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

        qstr = query if query is not None else query_text
        qstr = str(qstr or "")

        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        n = len(ws.labels)
        normalized = bool(ws.meta.get("normalized", True))

        # FAISS candidate restriction (optional)
        cand_idxs: Optional[np.ndarray] = None
        M = int(max(0, cfg.faiss_candidates))

        if normalized and ws.faiss_index is not None and 0 < M < n:
            qn = l2_normalize(q.reshape(1, -1), axis=1)[0].astype(np.float32)
            _, I = ws.faiss_index.search(qn.reshape(1, -1), min(M, n))
            cand = np.asarray(I[0], dtype=np.int64)
            cand = cand[cand >= 0]
            if cand.size > 0:
                cand_idxs = cand.astype(np.int64)
                sem_sims = np.full((n,), -np.inf, dtype=np.float32)
                sem_sims[cand_idxs] = (ws.embeddings[cand_idxs] @ qn).astype(np.float32)
            else:
                sem_sims = batched_cosine_sims(ws.embeddings, q).astype(np.float32)
        else:
            sem_sims = batched_cosine_sims(ws.embeddings, q).astype(np.float32)

        sims = sem_sims

        if cfg.grid_sim_weight > 0.0 and ws.grid is not None and ws.grid_feats is not None:
            qg = ws.grid.features(l2_normalize(q, axis=0) if normalized else q)
            if cand_idxs is not None:
                grid_sims = np.full((n,), -np.inf, dtype=np.float32)
                grid_sims[cand_idxs] = batched_cosine_sims(ws.grid_feats[cand_idxs], qg).astype(np.float32)
            else:
                grid_sims = batched_cosine_sims(ws.grid_feats, qg).astype(np.float32)

            w = float(max(0.0, min(1.0, cfg.grid_sim_weight)))
            sims = np.where(
                np.isfinite(sem_sims) & np.isfinite(grid_sims),
                ((1.0 - w) * sem_sims + w * grid_sims).astype(np.float32),
                sem_sims,
            )

        valid_mask = np.ones(n, dtype=bool)
        penalty = np.zeros(n, dtype=np.float32)

        if cfg.soft_constraints:
            wts = constraint_soft_weights(
                ws.labels,
                constraints,
                mismatch_weight=float(cfg.soft_constraint_mismatch_weight),
            )
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

        if best_sim >= float(cfg.direct_hit_cos):
            coarse = CoarseResult(best_idx=best_idx, best_path=[best_idx], visited=1)
            return QueryResult(
                query=qstr,
                start_idx=best_idx,
                coarse=coarse,
                final_idx=best_idx,
                final_label=ws.labels[best_idx],
                topk=topk,
                fine_steps=0,
                fine_path_z=[ws.embeddings[best_idx].copy()],
            )

        cue_idxs: List[int] = []
        if cfg.enable_cue_matching and qstr:
            cue_idxs = _extract_cue_indices(qstr, ws.labels)
        cue_idxs = [i for i in cue_idxs if valid_mask[i]]

        score_prior = masked_sims
        start_idx = best_idx
        cue_mode = False
        cue_internal: List[int] = []
        alpha: Optional[np.ndarray] = None
        coarse: Optional[CoarseResult] = None
        init_z: Optional[np.ndarray] = None

        # ---- cue mode ----
        if len(cue_idxs) >= 2:
            cue_mode = True
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
                cue_mix=0.55,
                cue_penalty=0.18,
                balance_penalty=0.20,
                valid_mask=valid_mask,
            )

            try:
                coarse_path = nx.shortest_path(ws.graph, source=src, target=dst, weight=_edge_cost)
            except Exception:
                coarse_path = [src]

            cue_internal = [int(i) for i in coarse_path[1:-1]]

            if cue_internal:
                coarse_best = int(max(cue_internal, key=lambda i: float(score_prior[i])))
            else:
                coarse_best = int(max(coarse_path, key=lambda i: float(score_prior[int(i)])))

            coarse = CoarseResult(
                best_idx=coarse_best,
                best_path=[int(i) for i in coarse_path],
                visited=len(coarse_path),
            )
            start_idx = src

            if cfg.cue_path_only:
                fine_path_z = (
                    [ws.embeddings[int(i)].copy() for i in coarse.best_path]
                    if emit_fine_steps
                    else [ws.embeddings[int(coarse_best)].copy()]
                )
                return QueryResult(
                    query=qstr,
                    start_idx=start_idx,
                    coarse=coarse,
                    final_idx=coarse_best,
                    final_label=ws.labels[coarse_best],
                    topk=topk,
                    fine_steps=0,
                    fine_path_z=fine_path_z,
                )

            # alpha only on internal nodes (avoid drifting back to cues)
            alpha = np.zeros(n, dtype=np.float32)
            if cue_internal:
                w_int = _stable_softmax(
                    np.asarray([score_prior[i] for i in cue_internal], dtype=np.float32),
                    temp=cfg.alpha_temp,
                ).astype(np.float32)
                for ii, wi in zip(cue_internal, w_int):
                    alpha[ii] = float(wi)
            else:
                alpha = _softmax_alpha(
                    score_prior, temp=cfg.alpha_temp, top_n=cfg.alpha_top_n, force_keep=[coarse_best]
                )

            mx = float(alpha.max())
            if mx > 0:
                alpha = (alpha / (mx + 1e-12)).astype(np.float32)

            # start fine descent from the cue location
            init_z = ws.embeddings[int(src)].copy()

        # ---- standard mode ----
        else:
            alpha = _softmax_alpha(score_prior, temp=cfg.alpha_temp, top_n=cfg.alpha_top_n, force_keep=[best_idx])
            coarse = coarse_search(
                ws.graph,
                start_idx=best_idx,
                alpha=alpha,
                beta=cfg.beta,
                max_expansions=cfg.max_expansions,
                max_depth=cfg.max_depth,
            )
            init_z = None

        # sigmas
        if cfg.use_local_sigmas and ws.sigmas is not None:
            sigmas = (ws.sigmas.astype(np.float32) * float(cfg.sigma_scale)).astype(np.float32)
        else:
            sigmas = np.full((n,), float(cfg.sigma), dtype=np.float32)

        dcfg = DescentConfig(
            lr=float(cfg.descent.lr),
            steps=int(cfg.descent.steps),
            tol_grad=float(cfg.descent.tol_grad),
            tol_move=float(cfg.descent.tol_move),
            clamp_norm=cfg.descent.clamp_norm,
            project_unit=normalized,
        )

        z_final, descent_trace = fine_descent(
            q,
            landmarks=ws.embeddings,
            alpha=alpha,
            sigmas=sigmas,
            cfg=dcfg,
            init_z=init_z,
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

        final_sims = final_sims - penalty
        final_sims[~valid_mask] = -np.inf

        # cue-mode readout: only within internal path nodes if available
        if cue_mode and cue_internal:
            cand = np.asarray([i for i in cue_internal if valid_mask[i]], dtype=int)
            if cand.size > 0:
                final_idx = int(cand[int(np.argmax(final_sims[cand]))])
            else:
                final_idx = int(np.argmax(final_sims))
        else:
            final_mix = 0.25
            final_scores = ((1.0 - final_mix) * final_sims + final_mix * score_prior).astype(np.float32)
            final_scores[~valid_mask] = -np.inf
            final_idx = int(np.argmax(final_scores))

        fine_path_z: List[np.ndarray] = []
        if emit_fine_steps:
            fine_path_z.extend([t.z.copy() for t in descent_trace])

        return QueryResult(
            query=qstr,
            start_idx=start_idx,
            coarse=coarse,
            final_idx=final_idx,
            final_label=ws.labels[final_idx],
            topk=topk,
            fine_steps=len(descent_trace),
            fine_path_z=fine_path_z,
        )