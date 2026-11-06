"""Point-group characters and fixed-subspace dimensions."""
import math
import numpy as np
from pymatgen.symmetry.groups import PointGroup

POINT_GROUPS = [
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

def so3_character(ell: int, theta: float, eps: float = 1e-10) -> float:
    if abs(theta) < eps:
        return float(2 * ell + 1)
    if abs(theta - math.pi) < eps:
        return float((-1) ** ell)
    return math.sin((ell + 0.5) * theta) / math.sin(theta / 2.0)

def o3_character(ell: int, parity: int, matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    det = int(round(np.linalg.det(matrix)))
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

def parity_char_to_int(parity: str) -> int:
    if parity == "e":
        return 1
    if parity == "o":
        return -1
    raise ValueError(f"Unsupported parity label: {parity}")

def invariant_multiplicity(point_group: str, ell: int, parity: str) -> int:
    ops = list(PointGroup(point_group).symmetry_ops)
    chars = [
        o3_character(ell, parity_char_to_int(parity), op.rotation_matrix)
        for op in ops
    ]
    return int(round(sum(chars) / len(chars)))
