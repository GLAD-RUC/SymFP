from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(
    path: str | Path,
    payload,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, sort_keys=sort_keys)
    return resolved


def write_rows_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def write_key_value_csv(
    path: str | Path,
    rows: Iterable[tuple[str, object]],
) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in rows:
            writer.writerow([key, value])
    return resolved


__all__ = [
    "ensure_parent",
    "write_json",
    "write_key_value_csv",
    "write_rows_csv",
]

