from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


def normalize_group_label(label: str) -> str:
    cleaned = str(label).strip().replace("\r", "")
    if not cleaned:
        raise ValueError("Point-group label must be non-empty.")

    alias_map = {
        "E": "C1",
        "sigma": "Cs",
        "SIGMA": "Cs",
    }
    return alias_map.get(cleaned, cleaned)


def _identity() -> np.ndarray:
    return np.eye(3, dtype=float)


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


def _deduplicate_matrices(
    matrices: Iterable[np.ndarray],
    atol: float = 1e-8,
) -> np.ndarray:
    unique: list[np.ndarray] = []
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=float)
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
    normalized = normalize_group_label(label)
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
    return tuple(deduplicated)


def group_order(label: str) -> int:
    return len(group_matrices(label))


def so3_character(ell: int, theta: float, eps: float = 1e-10) -> float:
    if abs(theta) < eps:
        return float(2 * ell + 1)
    if abs(theta - math.pi) < eps:
        return float((-1) ** ell)
    return math.sin((ell + 0.5) * theta) / math.sin(theta / 2.0)


def o3_character(ell: int, parity: int, matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    det = int(round(float(np.linalg.det(matrix))))
    if det not in (-1, 1):
        raise ValueError(f"Expected determinant +/-1, got {det}.")

    if det == 1:
        rotation = matrix
        parity_factor = 1.0
    else:
        rotation = -matrix
        parity_factor = float(parity)

    trace = float(np.trace(rotation))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    return parity_factor * so3_character(ell, theta)


def parity_to_int(parity: int | str) -> int:
    if parity in {1, "e"}:
        return 1
    if parity in {-1, "o"}:
        return -1
    raise ValueError(f"Unsupported parity specifier: {parity}")


def natural_parity_label(ell: int) -> str:
    return "e" if ell % 2 == 0 else "o"


@lru_cache(maxsize=None)
def invariant_multiplicity(label: str, ell: int, parity: int | str) -> int:
    matrices = group_matrices(label)
    parity_int = parity_to_int(parity)
    chars = [
        o3_character(ell=ell, parity=parity_int, matrix=matrix)
        for matrix in matrices
    ]
    return int(round(sum(chars) / len(chars)))


@lru_cache(maxsize=None)
def natural_parity_multiplicities(label: str, max_ell: int) -> tuple[int, ...]:
    normalized = normalize_group_label(label)
    return tuple(
        invariant_multiplicity(normalized, ell=ell, parity=natural_parity_label(ell))
        for ell in range(int(max_ell) + 1)
    )


__all__ = [
    "group_matrices",
    "group_order",
    "invariant_multiplicity",
    "natural_parity_label",
    "natural_parity_multiplicities",
    "normalize_group_label",
    "o3_character",
    "parity_to_int",
    "so3_character",
]
