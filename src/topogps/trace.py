from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np


@dataclass
class TraceEvent:
    t: float
    kind: str
    data: Dict[str, Any]


class TraceWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")

    def emit(self, kind: str, **data: Any) -> None:
        ev = TraceEvent(t=time.time(), kind=kind, data=_jsonable(data))
        self._f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TraceReader:
    """Simple NDJSON trace reader used by CLI/tools."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __iter__(self) -> Iterator[TraceEvent]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield TraceEvent(t=float(obj.get("t", 0.0)), kind=str(obj.get("kind", "")), data=dict(obj.get("data", {})))

    def read_all(self) -> List[TraceEvent]:
        return list(iter(self))


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x
