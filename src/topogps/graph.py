from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .utils import batched_cosine_sims


def build_association_graph(
    embeddings: np.ndarray,
    labels: List[str],
    *,
    knn: int = 12,
    min_sim: float = 0.55,
) -> nx.Graph:
    """Build a weighted undirected association graph over landmarks.

    Uses kNN to avoid O(N^2) full pairwise similarity.
    Weights are cosine similarity in [0,1].
    """
    if embeddings.shape[0] != len(labels):
        raise ValueError("embeddings and labels length mismatch")

    # cosine distance in sklearn: metric='cosine' yields 1 - cosine_sim.
    nn = NearestNeighbors(n_neighbors=min(knn + 1, len(labels)), metric="cosine")
    nn.fit(embeddings)
    dists, idxs = nn.kneighbors(embeddings, return_distance=True)

    G = nx.Graph()
    for i, label in enumerate(labels):
        G.add_node(i, label=label)

    for i in range(len(labels)):
        for jpos in range(1, idxs.shape[1]):
            j = int(idxs[i, jpos])
            sim = 1.0 - float(dists[i, jpos])
            if sim < min_sim:
                continue
            # Keep best weight if edge repeats.
            if G.has_edge(i, j):
                if sim > G[i][j]["weight"]:
                    G[i][j]["weight"] = sim
                continue
            G.add_edge(i, j, weight=sim)

    return G


@dataclass
class CoarseResult:
    best_idx: int
    best_path: List[int]
    visited: int


def coarse_search(
    G: nx.Graph,
    *,
    start_idx: int,
    alpha: np.ndarray,
    beta: float = 1.0,
    max_expansions: int = 2500,
    max_depth: int = 4,
) -> CoarseResult:
    """Bounded cue-aware graph search.

    We do not know the target node in advance; we search for nodes that are
    both reachable (low path cost) and cue-relevant (high alpha).

    Priority: f = path_cost - beta*log(alpha)
    Best node: maximize (beta*log(alpha) - path_cost)

    Edge traversal cost uses similarity weight: cost = 1/(weight+eps).
    """
    if start_idx not in G:
        raise ValueError("start_idx not in graph")
    if alpha.shape[0] != G.number_of_nodes():
        raise ValueError("alpha length mismatch")

    eps = 1e-9

    def node_util(node: int, g_cost: float) -> float:
        return beta * math.log(float(alpha[node]) + eps) - g_cost

    # (priority f, depth, g_cost, node, parent)
    heap: List[Tuple[float, int, float, int, Optional[int]]] = []
    heapq.heappush(heap, (0.0, 0, 0.0, start_idx, None))

    best = start_idx
    best_util = node_util(start_idx, 0.0)

    parents: Dict[int, Optional[int]] = {start_idx: None}
    best_at: Dict[int, float] = {start_idx: 0.0}

    expansions = 0
    while heap and expansions < max_expansions:
        f, depth, g_cost, node, parent = heapq.heappop(heap)
        if depth > max_depth:
            continue

        # Update best
        util = node_util(node, g_cost)
        if util > best_util:
            best_util = util
            best = node

        expansions += 1

        for nbr in G.neighbors(node):
            w = float(G[node][nbr].get("weight", 0.0))
            step_cost = 1.0 / (w + eps)
            new_g = g_cost + step_cost
            if nbr not in best_at or new_g < best_at[nbr] - 1e-12:
                best_at[nbr] = new_g
                parents[nbr] = node
                # A* like: f = g - beta*log(alpha)
                new_f = new_g - beta * math.log(float(alpha[nbr]) + eps)
                heapq.heappush(heap, (new_f, depth + 1, new_g, int(nbr), node))

    # Reconstruct path to best
    path: List[int] = []
    cur: Optional[int] = best
    while cur is not None:
        path.append(cur)
        cur = parents.get(cur)
    path.reverse()

    return CoarseResult(best_idx=best, best_path=path, visited=expansions)


def nearest_node(embeddings: np.ndarray, z: np.ndarray) -> int:
    sims = batched_cosine_sims(embeddings, z)
    return int(np.argmax(sims))
