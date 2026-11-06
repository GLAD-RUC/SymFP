"""Shared helpers for thin wrappers around official symmetry/shape tools."""

from __future__ import annotations

import json
import pickle
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


class ExternalToolUnavailableError(RuntimeError):
    """Raised when an official external tool is not installed."""


class ExternalToolExecutionError(RuntimeError):
    """Raised when an external tool call fails."""


def default_symbols(count: int, symbol: str = "H") -> tuple[str, ...]:
    """Return a constant symbol list for geometry-only baselines."""

    if int(count) <= 0:
        raise ValueError(f"Expected a positive atom count, got {count}.")
    return tuple(str(symbol) for _ in range(int(count)))


def ensure_positions(values: Any) -> np.ndarray:
    """Return coordinates as a float64 [N, 3] array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected coordinates with shape [N, 3], got {array.shape}.")
    return array


def write_xyz(
    path: str | Path,
    positions: Any,
    *,
    symbols: Sequence[str] | None = None,
    comment: str = "",
) -> Path:
    """Write a simple XYZ file."""

    coords = ensure_positions(positions)
    resolved_symbols = tuple(symbols) if symbols is not None else default_symbols(coords.shape[0])
    if len(resolved_symbols) != coords.shape[0]:
        raise ValueError(
            "The number of symbols must match the number of coordinates. "
            f"Got {len(resolved_symbols)} and {coords.shape[0]}."
        )

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as f:
        f.write(f"{coords.shape[0]}\n")
        f.write(f"{comment}\n")
        for symbol, point in zip(resolved_symbols, coords, strict=True):
            f.write(
                f"{symbol} {point[0]:.10f} {point[1]:.10f} {point[2]:.10f}\n"
            )
    return resolved_path


def resolve_executable(
    *,
    override: str | None,
    candidates: Sequence[str],
    install_hint: str,
) -> str:
    """Resolve an executable path from an override or a candidate list."""

    if override:
        resolved = shutil.which(str(override))
        if resolved is None:
            raise ExternalToolUnavailableError(
                f"Executable override not found: {override!r}. {install_hint}"
            )
        return resolved

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved

    raise ExternalToolUnavailableError(install_hint)


@contextmanager
def build_work_dir(
    root: str | Path | None,
    *,
    prefix: str,
) -> Iterator[Path]:
    """Yield a temporary or persistent working directory."""

    if root is None:
        with tempfile.TemporaryDirectory(prefix=prefix) as tmp_dir:
            yield Path(tmp_dir)
        return

    work_root = Path(root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    run_dir = work_root / f"{prefix}{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    yield run_dir


def parse_label_list(value: str | Sequence[str]) -> list[str]:
    """Parse a comma-separated or sequence-based label list."""

    if isinstance(value, str):
        labels = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    else:
        labels = [str(item).strip() for item in value if str(item).strip()]
    if not labels:
        raise ValueError("The label list must be non-empty.")

    deduplicated: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        deduplicated.append(label)
    return deduplicated


def parse_manual_center(
    value: str | Sequence[float] | None,
) -> tuple[float, float, float] | None:
    """Parse a manual center specification."""

    if value is None:
        return None
    if isinstance(value, str):
        parts = [chunk.strip() for chunk in value.split(",")]
    else:
        parts = [str(chunk).strip() for chunk in value]
    if len(parts) != 3:
        raise ValueError(f"manual center must contain three values, got {value!r}.")
    return float(parts[0]), float(parts[1]), float(parts[2])


def to_numpy(values: Any) -> np.ndarray:
    """Convert tensors and sequences to numpy arrays."""

    if isinstance(values, np.ndarray):
        return np.asarray(values)
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def record_species(record: dict[str, Any]) -> tuple[str, ...]:
    """Extract species labels from a toy-style record."""

    species = record.get("species")
    if species is not None:
        return tuple(str(value) for value in species)
    x = record.get("x")
    if x is None:
        raise KeyError("Record must contain either 'species' or 'x'.")
    atomic_numbers = to_numpy(x).reshape(-1)
    return tuple(str(int(value)) for value in atomic_numbers.tolist())


def record_to_sample(
    record: dict[str, Any],
    *,
    index: int | None = None,
    sample_id: str | None = None,
):
    """Convert a toy-style record dict into a labeled point cloud."""

    from .csom_modules.core import LabeledPointCloud

    if "pos" not in record:
        raise KeyError("Record must contain 'pos'.")

    resolved_id = sample_id or record.get("mid")
    if resolved_id is None:
        resolved_id = f"sample-{0 if index is None else index:04d}"
    true_group = record.get("point_group_hm") or record.get("point_group") or record.get("true_group")
    edge_index = record.get("edge_index")
    edge_index_np = None
    if edge_index is not None:
        edge_index_np = to_numpy(edge_index).astype(np.int64)
        if edge_index_np.ndim != 2:
            raise ValueError(f"Expected edge_index to be rank-2, got {edge_index_np.shape}.")
        if edge_index_np.shape[0] != 2 and edge_index_np.shape[1] == 2:
            edge_index_np = edge_index_np.T
        if edge_index_np.shape[0] != 2:
            raise ValueError(f"Expected edge_index with shape (2, E), got {edge_index_np.shape}.")

    return LabeledPointCloud(
        sample_id=str(resolved_id),
        species=record_species(record),
        positions=to_numpy(record["pos"]).astype(np.float64),
        true_group=None if true_group is None else str(true_group),
        edge_index=edge_index_np,
    )


def load_toy_samples(
    pkl_path: str | Path,
    *,
    split: str,
    limit: int | None = None,
):
    """Load toy samples from the current toy pickle."""

    resolved_path = Path(pkl_path).resolve()
    with resolved_path.open("rb") as f:
        dataset = pickle.load(f)
    if split not in dataset:
        raise KeyError(f"Split {split!r} not found in {resolved_path}.")

    records = list(dataset[split])
    if limit is not None:
        records = records[: int(limit)]
    return [
        record_to_sample(record, index=index)
        for index, record in enumerate(records)
    ]


def parse_candidate_groups(candidate_groups: str | Sequence[str]) -> list[str]:
    """Parse a candidate-group preset or explicit label list."""

    from .csom_modules.groups import TOY32_HM_GROUPS, normalize_group_label

    if isinstance(candidate_groups, str):
        if candidate_groups.strip().lower() == "toy32":
            return list(TOY32_HM_GROUPS)
        labels = [chunk.strip() for chunk in candidate_groups.split(",") if chunk.strip()]
    else:
        labels = [str(label).strip() for label in candidate_groups if str(label).strip()]

    if not labels:
        raise ValueError("candidate group list must be non-empty.")

    deduplicated: list[str] = []
    seen: set[str] = set()
    for label in labels:
        canonical = normalize_group_label(label)
        if canonical in seen:
            continue
        seen.add(canonical)
        deduplicated.append(label)
    return deduplicated


def find_newest_json(root: str | Path) -> tuple[Path | None, Any | None]:
    """Return the newest JSON file below a directory together with its parsed payload."""

    resolved_root = Path(root)
    json_paths = sorted(
        resolved_root.rglob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for json_path in json_paths:
        try:
            with json_path.open("r", encoding="utf-8") as f:
                return json_path, json.load(f)
        except Exception:
            continue
    return None, None


def find_numeric_value(
    payload: Any,
    *,
    preferred_keys: Sequence[str],
) -> float | None:
    """Recursively extract the first numeric field matching a preferred key order."""

    if isinstance(payload, dict):
        for key in preferred_keys:
            if key in payload and isinstance(payload[key], (int, float)):
                return float(payload[key])
        for value in payload.values():
            found = find_numeric_value(value, preferred_keys=preferred_keys)
            if found is not None:
                return found
        return None

    if isinstance(payload, list):
        for value in payload:
            found = find_numeric_value(value, preferred_keys=preferred_keys)
            if found is not None:
                return found
        return None

    if isinstance(payload, (int, float)):
        return float(payload)
    return None


def extract_last_float(text: str) -> float | None:
    """Extract the last float appearing in a text blob."""

    matches = _FLOAT_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def extract_table_value_after_header(text: str, *, header_token: str) -> float | None:
    """Extract the first table value following a header line containing a token."""

    lines = [line.rstrip() for line in text.splitlines()]
    header_index = None
    for index, line in enumerate(lines):
        if "Structure" in line and header_token in line:
            header_index = index
            break

    if header_index is not None:
        for line in lines[header_index + 1 :]:
            if not line.strip():
                continue
            matches = _FLOAT_RE.findall(line)
            if matches:
                return float(matches[-1])

    for line in lines:
        if header_token in line:
            matches = _FLOAT_RE.findall(line)
            if matches:
                return float(matches[-1])

    return extract_last_float(text)


def command_as_text(cmd: Sequence[str]) -> str:
    """Return a shell-like command preview."""

    return " ".join(str(part) for part in cmd)


def prepare_external_point_cloud(
    sample: Any,
    *,
    split_tag: str,
    center_mode: str,
    manual_center: Sequence[float] | None,
    cutoff: float | None,
    noise_alpha: float,
    noise_seed: int,
    cutoff_inclusive: bool,
) -> Any:
    """Prepare a point cloud while preserving external-tool preprocessing semantics."""

    from .csom_modules.core import PreparedPointCloud, add_relative_gaussian_noise

    points = ensure_positions(sample.positions)
    noisy_points, noise_scale = add_relative_gaussian_noise(
        points,
        sample_id=str(sample.sample_id),
        split_tag=str(split_tag),
        noise_alpha=float(noise_alpha),
        noise_seed=int(noise_seed),
    )

    if center_mode == "centroid":
        origin = np.mean(noisy_points, axis=0)
    elif center_mode == "first_atom":
        origin = np.asarray(noisy_points[0], dtype=float)
    elif center_mode == "manual":
        if manual_center is None:
            raise ValueError("manual_center is required when center_mode='manual'.")
        origin = np.asarray(tuple(float(value) for value in manual_center), dtype=float)
        if origin.shape != (3,):
            raise ValueError(f"manual_center must have shape (3,), got {origin.shape}.")
    elif center_mode == "none":
        origin = np.zeros(3, dtype=float)
    else:
        raise ValueError(f"Unsupported center_mode: {center_mode}")

    centered = noisy_points - origin[None, :]
    species = tuple(str(value) for value in sample.species)

    if cutoff is None:
        kept_points = centered
        kept_species = species
    else:
        cutoff_value = float(cutoff)
        if cutoff_value <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}.")
        radii = np.linalg.norm(centered, axis=1)
        if cutoff_inclusive:
            keep_mask = radii <= cutoff_value
        else:
            keep_mask = radii < cutoff_value
        if not np.any(keep_mask):
            raise ValueError("All nodes were removed by the cutoff.")
        kept_points = centered[keep_mask]
        kept_species = tuple(species[index] for index in np.flatnonzero(keep_mask))

    return PreparedPointCloud(
        sample_id=str(sample.sample_id),
        species=kept_species,
        positions=np.asarray(kept_points, dtype=float),
        true_group=None if getattr(sample, "true_group", None) is None else str(sample.true_group),
        origin=np.asarray(origin, dtype=float),
        noise_scale=float(noise_scale),
        num_nodes_before=int(points.shape[0]),
        num_nodes_after=int(kept_points.shape[0]),
    )


__all__ = [
    "ExternalToolExecutionError",
    "ExternalToolUnavailableError",
    "build_work_dir",
    "command_as_text",
    "default_symbols",
    "ensure_positions",
    "extract_last_float",
    "extract_table_value_after_header",
    "find_newest_json",
    "find_numeric_value",
    "load_toy_samples",
    "parse_candidate_groups",
    "parse_label_list",
    "parse_manual_center",
    "prepare_external_point_cloud",
    "record_species",
    "record_to_sample",
    "resolve_executable",
    "to_numpy",
    "write_xyz",
]
