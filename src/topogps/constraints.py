from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ConstraintSpec:
    """Simple textual constraints over landmark labels.

    - starts_with: label must start with this prefix (case-insensitive)
    - contains: label must contain this substring (case-insensitive)

    These are intentionally lightweight and deterministic.
    """

    starts_with: Optional[str] = None
    contains: Optional[str] = None


def constraint_mask(labels: List[str], c: ConstraintSpec) -> np.ndarray:
    """Hard boolean mask over labels."""
    if (c.starts_with is None) and (c.contains is None):
        return np.ones(len(labels), dtype=bool)

    sw = c.starts_with.lower() if c.starts_with else None
    ct = c.contains.lower() if c.contains else None

    mask = np.ones(len(labels), dtype=bool)
    for i, lab in enumerate(labels):
        ll = lab.lower()
        if sw is not None and not ll.startswith(sw):
            mask[i] = False
        if ct is not None and ct not in ll:
            mask[i] = False
    return mask


def constraint_soft_weights(
    labels: List[str],
    c: ConstraintSpec,
    *,
    mismatch_weight: float = 0.12,
) -> np.ndarray:
    """Soft weights in [0,1] that downweight constraint mismatches.

    Semantics:
      - if a constraint is satisfied => weight 1
      - if violated => weight (1 - mismatch_weight)

    When multiple constraints exist, weights multiply.
    """
    w = np.ones((len(labels),), dtype=np.float32)
    if (c.starts_with is None) and (c.contains is None):
        return w

    sw = c.starts_with.lower() if c.starts_with else None
    ct = c.contains.lower() if c.contains else None

    mw = float(max(0.0, min(1.0, mismatch_weight)))
    bad = 1.0 - mw

    for i, lab in enumerate(labels):
        ll = lab.lower()
        if sw is not None and not ll.startswith(sw):
            w[i] *= bad
        if ct is not None and ct not in ll:
            w[i] *= bad
    return w


# Backwards-compat (older experiments may import these names)
@dataclass
class LandmarkConstraints:
    require: List[str]
    avoid: List[str]

    def hard_mask(self, labels: List[str], candidate_idxs: np.ndarray) -> np.ndarray:
        req = [r.lower() for r in self.require]
        av = [a.lower() for a in self.avoid]

        mask = np.ones((len(candidate_idxs),), dtype=bool)
        for i, idx in enumerate(candidate_idxs):
            s = labels[int(idx)].lower()
            if req and not all(r in s for r in req):
                mask[i] = False
                continue
            if av and any(a in s for a in av):
                mask[i] = False
        return mask


@dataclass
class LandmarkSoftConstraints:
    require: List[str]
    avoid: List[str]


def legacy_constraint_soft_weights(labels: List[str], soft: LandmarkSoftConstraints) -> np.ndarray:
    req = [r.lower() for r in soft.require]
    av = [a.lower() for a in soft.avoid]

    w = np.zeros((len(labels),), dtype=np.float32)
    for i, lab in enumerate(labels):
        s = lab.lower()
        if req:
            miss = sum(1 for r in req if r not in s)
            w[i] += 0.15 * float(miss)
        if av:
            hits = sum(1 for a in av if a in s)
            w[i] += 0.35 * float(hits)
    return w
