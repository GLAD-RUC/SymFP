"""Point-group helpers for the standalone CSoM baseline."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.spatial.transform import Rotation


TOY32_HM_GROUPS = [
    "1",
    "-1",
    "2",
    "m",
    "2/m",
    "222",
    "mm2",
    "mmm",
    "4",
    "-4",
    "4/m",
    "422",
    "4mm",
    "-42m",
    "4/mmm",
    "3",
    "-3",
    "32",
    "3m",
    "-3m",
    "6",
    "-6",
    "6/m",
    "622",
    "6mm",
    "-6m2",
    "6/mmm",
    "23",
    "m-3",
    "432",
    "-43m",
    "m-3m",
]


@dataclass(frozen=True)
class CandidateTemplate:
    """Metadata for one candidate point group."""

    label_input: str
    label_schoenflies: str
    label_hm: str | None

    @classmethod
    def from_label(cls, label: str) -> "CandidateTemplate":
        label_schoenflies, label_hm = canonical_label_pair(label)
        return cls(
            label_input=str(label),
            label_schoenflies=label_schoenflies,
            label_hm=label_hm,
        )

SCHOENFLIES_TO_HM = {
    "C1": "1",
    "Ci": "-1",
    "C2": "2",
    "Cs": "m",
    "C2h": "2/m",
    "D2": "222",
    "C2v": "mm2",
    "D2h": "mmm",
    "C4": "4",
    "S4": "-4",
    "C4h": "4/m",
    "D4": "422",
    "C4v": "4mm",
    "D2d": "-42m",
    "D4h": "4/mmm",
    "C3": "3",
    "S6": "-3",
    "D3": "32",
    "C3v": "3m",
    "D3d": "-3m",
    "C6": "6",
    "C3h": "-6",
    "C6h": "6/m",
    "D6": "622",
    "C6v": "6mm",
    "D3h": "-6m2",
    "D6h": "6/mmm",
    "T": "23",
    "Th": "m-3",
    "O": "432",
    "Td": "-43m",
    "Oh": "m-3m",
}
HM_TO_SCHOENFLIES = {value: key for key, value in SCHOENFLIES_TO_HM.items()}
SCHOENFLIES_ALIASES = {
    "E": "C1",
    "e": "C1",
    "sigma": "Cs",
    "SIGMA": "Cs",
    "C3i": "S6",
}
SPECIAL_SCHOENFLIES = {
    "C1",
    "Ci",
    "Cs",
    "T",
    "Td",
    "Th",
    "O",
    "Oh",
    "I",
    "Ih",
}
POINT_GROUP_TO_SYSTEM = {
    "1": "triclinic",
    "-1": "triclinic",
    "2": "monoclinic",
    "m": "monoclinic",
    "2/m": "monoclinic",
    "222": "orthorhombic",
    "mm2": "orthorhombic",
    "mmm": "orthorhombic",
    "4": "tetragonal",
    "-4": "tetragonal",
    "4/m": "tetragonal",
    "422": "tetragonal",
    "4mm": "tetragonal",
    "-42m": "tetragonal",
    "4/mmm": "tetragonal",
    "3": "trigonal",
    "-3": "trigonal",
    "32": "trigonal",
    "3m": "trigonal",
    "-3m": "trigonal",
    "6": "hexagonal",
    "-6": "hexagonal",
    "6/m": "hexagonal",
    "622": "hexagonal",
    "6mm": "hexagonal",
    "-6m2": "hexagonal",
    "6/mmm": "hexagonal",
    "23": "cubic",
    "m-3": "cubic",
    "432": "cubic",
    "-43m": "cubic",
    "m-3m": "cubic",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_CSOM_OPERATIONS_DIR = (
    _PROJECT_ROOT
    / ".vendor"
    / "Continous-Symmetry-Operation-Measure-Program-main"
    / "Continous-Symmetry-Operation-Measure"
    / "Operations"
)


def normalize_group_label(label: str) -> str:
    """Return a canonical Schoenflies label."""

    cleaned = str(label).strip().replace("\r", "")
    if not cleaned:
        raise ValueError("Point-group label must be non-empty.")

    alias = SCHOENFLIES_ALIASES.get(cleaned, cleaned)
    if alias in HM_TO_SCHOENFLIES:
        return HM_TO_SCHOENFLIES[alias]
    if alias in SPECIAL_SCHOENFLIES:
        return alias
    if re.fullmatch(r"C(\d+)([hv])?", alias):
        return alias
    if re.fullmatch(r"D(\d+)([hd])?", alias):
        return alias
    if re.fullmatch(r"S(\d+)", alias):
        return alias

    raise KeyError(f"Unsupported point-group label: {label}")


def hm_label(label: str) -> Optional[str]:
    """Return the crystallographic HM alias when it exists."""

    normalized = normalize_group_label(label)
    return SCHOENFLIES_TO_HM.get(normalized)


def canonical_label_pair(label: str) -> tuple[str, Optional[str]]:
    """Return (Schoenflies, HM-or-None) for an input label."""

    normalized = normalize_group_label(label)
    return normalized, SCHOENFLIES_TO_HM.get(normalized)


def _identity() -> np.ndarray:
    return np.eye(3, dtype=float)


def reference_basis(crystal_system: str) -> np.ndarray:
    if crystal_system in {"trigonal", "hexagonal"}:
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [-0.5, math.sqrt(3.0) / 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    return np.eye(3, dtype=float)


def to_euclidean_operator(rotation: np.ndarray, crystal_system: str) -> np.ndarray:
    basis = reference_basis(crystal_system)
    basis_inv = np.linalg.inv(basis)
    operator_t = basis_inv @ np.asarray(rotation, dtype=float).T @ basis
    operator = operator_t.T
    operator[np.abs(operator) < 1e-12] = 0.0
    return operator


def rotation_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def reflection_horizontal() -> np.ndarray:
    return np.diag([1.0, 1.0, -1.0]).astype(float)


def reflection_vertical(phi: float = 0.0) -> np.ndarray:
    base = np.diag([1.0, -1.0, 1.0]).astype(float)
    rot = rotation_z(phi)
    return rot @ base @ rot.T


def inversion() -> np.ndarray:
    return -_identity()


def roto_reflection_z(theta: float) -> np.ndarray:
    return reflection_horizontal() @ rotation_z(theta)


def _project_to_o3(matrix: np.ndarray) -> np.ndarray:
    """Project a near-orthogonal matrix onto the closest O(3) operator."""

    candidate = np.asarray(matrix, dtype=float)
    u, _, vh = np.linalg.svd(candidate)
    projected = u @ vh

    input_det = float(np.linalg.det(candidate))
    desired_sign = -1.0 if input_det < 0.0 else 1.0
    if float(np.linalg.det(projected)) * desired_sign < 0.0:
        reflection_fix = np.eye(3, dtype=float)
        reflection_fix[-1, -1] = -1.0
        projected = u @ reflection_fix @ vh

    projected[np.abs(projected) < 1e-12] = 0.0
    return projected


def _deduplicate_matrices(
    matrices: Iterable[np.ndarray],
    atol: float = 1e-8,
) -> np.ndarray:
    unique: list[np.ndarray] = []
    for matrix in matrices:
        matrix = _project_to_o3(np.asarray(matrix, dtype=float))
        if any(np.allclose(matrix, existing, atol=atol) for existing in unique):
            continue
        unique.append(matrix)
    return np.stack(unique, axis=0)


def _expected_group_order(label: str) -> int:
    if label == "C1":
        return 1
    if label in {"Ci", "Cs"}:
        return 2
    if label == "T":
        return 12
    if label in {"Td", "Th"}:
        return 24
    if label == "O":
        return 24
    if label == "Oh":
        return 48
    if label == "I":
        return 60
    if label == "Ih":
        return 120

    match = re.fullmatch(r"C(\d+)", label)
    if match:
        return int(match.group(1))
    match = re.fullmatch(r"C(\d+)[hv]", label)
    if match:
        return 2 * int(match.group(1))
    match = re.fullmatch(r"D(\d+)", label)
    if match:
        return 2 * int(match.group(1))
    match = re.fullmatch(r"D(\d+)[hd]", label)
    if match:
        return 4 * int(match.group(1))
    match = re.fullmatch(r"S(\d+)", label)
    if match:
        return int(match.group(1))

    raise KeyError(f"Unsupported Schoenflies label: {label}")


def _official_pointgroup_csv_path(label: str) -> Path | None:
    stem = "E" if label == "C1" else label
    path = _OFFICIAL_CSOM_OPERATIONS_DIR / f"pointgroup_{stem}.csv"
    if path.is_file():
        return path
    return None


@lru_cache(maxsize=None)
def _official_group_matrices(label: str) -> tuple[np.ndarray, ...] | None:
    normalized = normalize_group_label(label)
    csv_path = _official_pointgroup_csv_path(normalized)
    if csv_path is None:
        return None

    matrices: list[np.ndarray] = [_identity()]
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            if len(row) < 10:
                raise RuntimeError(
                    "Official CSoM operation file has an unexpected format. "
                    f"path={csv_path}, row={row!r}."
                )
            matrix = np.asarray(row[1:10], dtype=float).reshape(3, 3)
            matrix[np.abs(matrix) < 1e-12] = 0.0
            matrices.append(matrix)

    deduplicated = _deduplicate_matrices(matrices)
    expected = _expected_group_order(normalized)
    if deduplicated.shape[0] != expected:
        return None
    return tuple(np.asarray(matrix, dtype=float) for matrix in deduplicated)


def _proper_rotation_group(label: str) -> np.ndarray:
    if label in {"C1", "Ci", "Cs"}:
        return np.stack([_identity()], axis=0)

    if label == "T":
        return Rotation.create_group("T").as_matrix()
    if label == "O":
        return Rotation.create_group("O").as_matrix()
    if label == "I":
        return Rotation.create_group("I").as_matrix()

    match = re.fullmatch(r"C(\d+)", label)
    if match:
        n = int(match.group(1))
        if n == 1:
            return np.stack([_identity()], axis=0)
        return Rotation.create_group(f"C{n}", axis="Z").as_matrix()

    match = re.fullmatch(r"C(\d+)[hv]", label)
    if match:
        n = int(match.group(1))
        return _proper_rotation_group(f"C{n}")

    match = re.fullmatch(r"D(\d+)", label)
    if match:
        n = int(match.group(1))
        return Rotation.create_group(f"D{n}", axis="Z").as_matrix()

    match = re.fullmatch(r"D(\d+)[hd]", label)
    if match:
        n = int(match.group(1))
        return _proper_rotation_group(f"D{n}")

    if label in {"Td", "Th"}:
        return _proper_rotation_group("T")
    if label == "Oh":
        return _proper_rotation_group("O")
    if label == "Ih":
        return _proper_rotation_group("I")

    match = re.fullmatch(r"S(\d+)", label)
    if match:
        n = int(match.group(1))
        if n % 2 != 0:
            raise ValueError(f"Only even-order improper rotations are supported, got {label}.")
        return _proper_rotation_group(f"C{n // 2}")

    raise KeyError(f"Unsupported Schoenflies label: {label}")


def _improper_generator(label: str) -> np.ndarray | None:
    if label == "Ci":
        return inversion()
    if label == "Cs":
        return reflection_vertical(0.0)
    if label in {"Th", "Oh", "Ih"}:
        return inversion()
    if label == "Td":
        return roto_reflection_z(math.pi / 2.0)

    match = re.fullmatch(r"C(\d+)h", label)
    if match:
        return reflection_horizontal()

    match = re.fullmatch(r"C(\d+)v", label)
    if match:
        return reflection_vertical(0.0)

    match = re.fullmatch(r"D(\d+)h", label)
    if match:
        return reflection_horizontal()

    match = re.fullmatch(r"D(\d+)d", label)
    if match:
        n = int(match.group(1))
        return reflection_vertical(math.pi / (2.0 * n))

    match = re.fullmatch(r"S(\d+)", label)
    if match:
        n = int(match.group(1))
        return roto_reflection_z(2.0 * math.pi / n)

    return None


@lru_cache(maxsize=None)
def group_matrices(label: str) -> tuple[np.ndarray, ...]:
    """Return all O(3) operators of a point group as 3x3 matrices."""

    normalized = normalize_group_label(label)
    official = _official_group_matrices(normalized)
    if official is not None:
        return official

    hm = SCHOENFLIES_TO_HM.get(normalized)
    if hm is not None:
        from pymatgen.symmetry.groups import PointGroup

        crystal_system = POINT_GROUP_TO_SYSTEM[hm]
        matrices = [
            to_euclidean_operator(np.asarray(op.rotation_matrix, dtype=float), crystal_system)
            for op in PointGroup(hm).symmetry_ops
        ]
        deduplicated = _deduplicate_matrices(matrices)
        expected = _expected_group_order(normalized)
        if deduplicated.shape[0] != expected:
            raise RuntimeError(
                "Failed to construct the expected point-group order. "
                f"label={normalized}, expected={expected}, got={deduplicated.shape[0]}."
            )
        return tuple(np.asarray(matrix, dtype=float) for matrix in deduplicated)

    proper = _proper_rotation_group(normalized)
    generator = _improper_generator(normalized)
    matrices: list[np.ndarray] = [matrix for matrix in proper]
    if generator is not None:
        matrices.extend(generator @ matrix for matrix in proper)
        matrices.extend(matrix @ generator for matrix in proper)
    deduplicated = _deduplicate_matrices(matrices)
    expected = _expected_group_order(normalized)
    if deduplicated.shape[0] != expected:
        raise RuntimeError(
            "Failed to construct the expected point-group order. "
            f"label={normalized}, expected={expected}, got={deduplicated.shape[0]}."
        )
    return tuple(np.asarray(matrix, dtype=float) for matrix in deduplicated)


def group_order(label: str) -> int:
    """Return the group order."""

    return len(group_matrices(label))


def csom_operation_matrices(
    label: str,
    *,
    exclude_identity: bool = True,
    atol: float = 1e-8,
) -> tuple[np.ndarray, ...]:
    """Return the operator list used by the CSoM deviation objective."""

    matrices = list(group_matrices(label))
    if exclude_identity and len(matrices) > 1:
        filtered = [
            matrix
            for matrix in matrices
            if not np.allclose(matrix, np.eye(3, dtype=float), atol=atol)
        ]
        if filtered:
            matrices = filtered
    return tuple(np.asarray(matrix, dtype=float) for matrix in matrices)


__all__ = [
    "CandidateTemplate",
    "HM_TO_SCHOENFLIES",
    "SCHOENFLIES_TO_HM",
    "TOY32_HM_GROUPS",
    "canonical_label_pair",
    "csom_operation_matrices",
    "group_matrices",
    "group_order",
    "hm_label",
    "normalize_group_label",
]
