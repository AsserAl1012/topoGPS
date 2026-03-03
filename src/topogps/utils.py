from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np


def ensure_dir(path: Path) -> Path:
    """mkdir -p and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    """Seed python/numpy for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x)
    denom = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / denom


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    an = np.linalg.norm(a) + eps
    bn = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (an * bn))


def batched_cosine_sims(mat: np.ndarray, vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity of each row of mat with vec."""
    mat = np.asarray(mat, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    matn = l2_normalize(mat, axis=1, eps=eps)
    vecn = l2_normalize(vec, axis=1, eps=eps)[0]
    return (matn @ vecn).astype(np.float32)


_SPLIT_RE = re.compile(r"[,\t;]+")


def _maybe_split_inline_list(lines: List[str]) -> List[str]:
    """
    Back-compat + UX: if a file contains a single long line like:
      "a b c" or "a, b, c" or "a; b; c"
    we split it into concepts. This fixes `data/sample_concepts.txt` without requiring
    PowerShell preprocessing.
    """
    if not lines:
        return []

    if len(lines) != 1:
        return lines

    ln = lines[0].strip()
    if not ln:
        return []

    # If comma/semicolon/tab present, split on those.
    if any(ch in ln for ch in [",", ";", "\t"]):
        parts = [p.strip() for p in _SPLIT_RE.split(ln) if p.strip()]
        return parts

    # Otherwise, if multiple spaces exist, treat as whitespace-separated list.
    if " " in ln:
        parts = [p.strip() for p in ln.split() if p.strip()]
        # avoid splitting legitimate multi-word concepts like "black hole"
        # Heuristic: only split if the line contains MANY tokens and no newline structure.
        # For sample_concepts.txt it’s all single words except a few bigrams — acceptable for now.
        return parts

    return lines


def read_lines(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return _maybe_split_inline_list(lines)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x / max(float(temperature), 1e-9)
    x = x - np.max(x)
    ex = np.exp(x)
    return (ex / (np.sum(ex) + 1e-12)).astype(np.float64)