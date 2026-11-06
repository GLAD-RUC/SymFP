"""Core CSoM utilities for point-cloud evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from .groups import canonical_label_pair, csom_operation_matrices, group_order


@dataclass(frozen=True)
class LabeledPointCloud:
    """Raw point cloud with labels used for permutation-aware matching."""

    sample_id: str
    species: tuple[str, ...]
    positions: np.ndarray
    true_group: str | None = None
    edge_index: np.ndarray | None = None


@dataclass(frozen=True)
class PreparedPointCloud:
    """Preprocessed point cloud ready for CSoM evaluation."""

    sample_id: str
    species: tuple[str, ...]
    positions: np.ndarray
    true_group: str | None
    origin: np.ndarray
    noise_scale: float
    num_nodes_before: int
    num_nodes_after: int
    edge_index: np.ndarray | None = None


@dataclass(frozen=True)
class CandidateEvaluation:
    """Target-conditioned CSoM result for one candidate point group."""

    label_input: str
    label_schoenflies: str
    label_hm: str | None
    group_order: int
    mean_deviation: float
    max_deviation: float
    per_operation_deviations: tuple[float, ...]
    rotation_matrix: np.ndarray


def stable_int_hash(text: str) -> int:
    """Return a deterministic integer hash."""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31)


def median_nearest_neighbor_distance(points: np.ndarray) -> float:
    """Return a robust structure scale based on nearest neighbors."""

    if points.shape[0] <= 1:
        return 1.0
    distances = cdist(points, points, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return 1.0
    return float(max(np.median(finite), 1e-6))


def add_relative_gaussian_noise(
    points: np.ndarray,
    *,
    sample_id: str,
    split_tag: str,
    noise_alpha: float,
    noise_seed: int,
) -> tuple[np.ndarray, float]:
    """Add deterministic Gaussian noise scaled by the median NN distance."""

    if noise_alpha <= 0:
        return np.asarray(points, dtype=float), 0.0

    scale = median_nearest_neighbor_distance(points)
    seed = stable_int_hash(f"{sample_id}|{split_tag}|{noise_alpha:.6f}|{noise_seed}")
    rng = np.random.default_rng(seed)
    noisy = np.asarray(points, dtype=float) + (noise_alpha * scale) * rng.standard_normal(points.shape)
    return noisy, scale


def center_points(
    points: np.ndarray,
    *,
    mode: str,
    manual_center: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Center a point cloud using the requested convention."""

    points = np.asarray(points, dtype=float)
    if mode == "centroid":
        origin = np.mean(points, axis=0)
    elif mode == "first_atom":
        origin = points[0]
    elif mode == "manual":
        if manual_center is None:
            raise ValueError("manual_center is required when center mode is 'manual'.")
        origin = np.asarray(manual_center, dtype=float)
        if origin.shape != (3,):
            raise ValueError(f"manual_center must have shape (3,), got {origin.shape}.")
    elif mode == "none":
        origin = np.zeros(3, dtype=float)
    else:
        raise ValueError(f"Unsupported center mode: {mode}")
    return points - origin[None, :], origin


def apply_cutoff(
    species: Sequence[str],
    points: np.ndarray,
    *,
    cutoff: float | None,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Optionally keep only points inside a radius cutoff."""

    species_tuple = tuple(str(value) for value in species)
    points = np.asarray(points, dtype=float)
    if cutoff is None:
        return species_tuple, points, np.arange(points.shape[0], dtype=int)
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}.")

    radii = np.linalg.norm(points, axis=1)
    mask = radii <= float(cutoff)
    if not np.any(mask):
        raise ValueError("All nodes were removed by the cutoff.")
    kept_indices = np.flatnonzero(mask).astype(int)
    return tuple(species_tuple[index] for index in kept_indices), points[mask], kept_indices


def remap_edge_index(
    edge_index: np.ndarray | None,
    kept_indices: np.ndarray,
) -> np.ndarray | None:
    """Remap an edge list after node filtering."""

    if edge_index is None:
        return None

    edge_index_np = np.asarray(edge_index, dtype=int)
    if edge_index_np.shape[0] != 2:
        raise ValueError(f"Expected edge_index with shape (2, E), got {edge_index_np.shape}.")

    kept_indices = np.asarray(kept_indices, dtype=int).reshape(-1)
    old_to_new = {int(old_index): new_index for new_index, old_index in enumerate(kept_indices.tolist())}
    src = edge_index_np[0]
    dst = edge_index_np[1]
    keep_mask = np.isin(src, kept_indices) & np.isin(dst, kept_indices)
    if not np.any(keep_mask):
        return np.empty((2, 0), dtype=int)

    filtered = edge_index_np[:, keep_mask]
    remapped_src = np.asarray([old_to_new[int(value)] for value in filtered[0]], dtype=int)
    remapped_dst = np.asarray([old_to_new[int(value)] for value in filtered[1]], dtype=int)
    return np.stack([remapped_src, remapped_dst], axis=0)


def prepare_point_cloud(
    sample: LabeledPointCloud,
    *,
    split_tag: str,
    center_mode: str,
    manual_center: Sequence[float] | None,
    cutoff: float | None,
    noise_alpha: float,
    noise_seed: int,
) -> PreparedPointCloud:
    """Apply noise, centering, and cutoff to a raw point cloud."""

    points = np.asarray(sample.positions, dtype=float)
    noisy_points, noise_scale = add_relative_gaussian_noise(
        points,
        sample_id=sample.sample_id,
        split_tag=split_tag,
        noise_alpha=noise_alpha,
        noise_seed=noise_seed,
    )
    centered, origin = center_points(
        noisy_points,
        mode=center_mode,
        manual_center=manual_center,
    )
    kept_species, kept_points, kept_indices = apply_cutoff(
        sample.species,
        centered,
        cutoff=cutoff,
    )
    if kept_points.shape[0] == 0:
        raise ValueError("No points remain after preprocessing.")
    remapped_edge_index = remap_edge_index(sample.edge_index, kept_indices)
    return PreparedPointCloud(
        sample_id=sample.sample_id,
        species=kept_species,
        positions=np.asarray(kept_points, dtype=float),
        true_group=sample.true_group,
        origin=np.asarray(origin, dtype=float),
        noise_scale=float(noise_scale),
        num_nodes_before=int(points.shape[0]),
        num_nodes_after=int(kept_points.shape[0]),
        edge_index=remapped_edge_index,
    )


def apply_rotation(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply a right-acting rotation to row-vector coordinates."""

    return np.asarray(points, dtype=float) @ np.asarray(rotation, dtype=float).T


def symmetry_deviation(
    reference_points: np.ndarray,
    operated_points: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Continuous symmetry deviation used by the CSoM baseline."""

    reference_points = np.asarray(reference_points, dtype=float)
    operated_points = np.asarray(operated_points, dtype=float)
    diff = reference_points - operated_points
    numerator = np.sum(diff * diff, axis=1)
    centered = reference_points - np.mean(reference_points, axis=0, keepdims=True)
    denominator = np.sum(centered * centered, axis=1)
    denominator = np.where(denominator > eps, denominator, 1.0)
    return float(100.0 * np.mean(numerator / denominator))


def _stable_unique_labels(labels: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for label in labels:
        label_text = str(label)
        if label_text in seen:
            continue
        seen.add(label_text)
        ordered.append(label_text)
    return ordered


def match_operated_points(
    reference_points: np.ndarray,
    species: Sequence[str],
    operated_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match an operated point cloud back to the reference up to same-species permutations."""

    reference_points = np.asarray(reference_points, dtype=float)
    operated_points = np.asarray(operated_points, dtype=float)
    species_array = np.asarray([str(value) for value in species], dtype=object)

    permutation = np.empty(reference_points.shape[0], dtype=int)
    for label in _stable_unique_labels(species_array.tolist()):
        label_mask = species_array == label
        label_indices = np.flatnonzero(label_mask)
        cost = cdist(
            reference_points[label_indices],
            operated_points[label_indices],
            metric="euclidean",
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        permutation[label_indices[row_ind]] = label_indices[col_ind]

    return operated_points[permutation], permutation


def operation_deviations(
    reference_points: np.ndarray,
    species: Sequence[str],
    operations: Sequence[np.ndarray],
) -> tuple[float, ...]:
    """Compute per-operation deviations for a fixed orientation."""

    deviations: list[float] = []
    for operation in operations:
        operated = apply_rotation(reference_points, operation)
        matched, _ = match_operated_points(reference_points, species, operated)
        deviations.append(symmetry_deviation(reference_points, matched))
    return tuple(float(value) for value in deviations)


def _axis_to_rotation(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.eye(3, dtype=float)
    axis = axis / axis_norm
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(z_axis, axis), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=float)
    if dot < -1.0 + 1e-10:
        return Rotation.from_rotvec(np.array([math.pi, 0.0, 0.0], dtype=float)).as_matrix()
    cross = np.cross(z_axis, axis)
    cross_norm = np.linalg.norm(cross)
    cross = cross / cross_norm
    angle = math.acos(dot)
    return Rotation.from_rotvec(cross * angle).as_matrix()


def fibonacci_sphere(samples: int) -> np.ndarray:
    """Generate approximately uniform points on the unit sphere."""

    if samples <= 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=float)

    points = np.empty((samples, 3), dtype=float)
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    indices = np.arange(samples, dtype=float)
    y = 1.0 - (indices / float(samples - 1)) * 2.0
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * indices
    points[:, 0] = np.cos(theta) * radius
    points[:, 1] = y
    points[:, 2] = np.sin(theta) * radius
    return points


def seed_rotations(
    *,
    num_axis_samples: int,
    num_angle_samples: int,
) -> tuple[np.ndarray, ...]:
    """Build deterministic orientation seeds."""

    base_axes = [
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, -1.0], dtype=float),
        np.array([-1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, -1.0, 0.0], dtype=float),
    ]
    if num_axis_samples > len(base_axes):
        base_axes.extend(list(fibonacci_sphere(num_axis_samples)))
    axes = base_axes[: max(1, num_axis_samples)]

    if num_angle_samples <= 1:
        angles = [0.0]
    else:
        angles = np.linspace(0.0, 2.0 * math.pi, num_angle_samples, endpoint=False).tolist()

    rotations: list[np.ndarray] = []
    for axis in axes:
        align = _axis_to_rotation(axis)
        axis_unit = axis / max(np.linalg.norm(axis), 1e-12)
        for angle in angles:
            twist = Rotation.from_rotvec(axis_unit * angle).as_matrix()
            rotations.append(twist @ align)

    unique: list[np.ndarray] = []
    for rotation in rotations:
        if any(np.allclose(rotation, existing, atol=1e-8) for existing in unique):
            continue
        unique.append(rotation)
    return tuple(unique)


def optimize_orientation(
    points: np.ndarray,
    species: Sequence[str],
    operations: Sequence[np.ndarray],
    *,
    num_axis_samples: int,
    num_angle_samples: int,
    local_opt_maxiter: int,
) -> tuple[np.ndarray, float]:
    """Search for an orientation minimizing the mean deviation."""

    if len(operations) == 0:
        return np.eye(3, dtype=float), 0.0

    points = np.asarray(points, dtype=float)
    seeds = seed_rotations(
        num_axis_samples=max(1, num_axis_samples),
        num_angle_samples=max(1, num_angle_samples),
    )

    best_rotation = np.eye(3, dtype=float)
    best_score = float("inf")

    def objective_from_matrix(rotation_matrix: np.ndarray) -> float:
        rotated = apply_rotation(points, rotation_matrix)
        deviations = operation_deviations(rotated, species, operations)
        return float(sum(deviations) / len(deviations))

    for rotation_matrix in seeds:
        score = objective_from_matrix(rotation_matrix)
        if score < best_score:
            best_score = score
            best_rotation = rotation_matrix

    if local_opt_maxiter > 0:
        x0 = Rotation.from_matrix(best_rotation).as_rotvec()

        def objective_from_rotvec(rotvec: np.ndarray) -> float:
            matrix = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
            return objective_from_matrix(matrix)

        result = minimize(
            objective_from_rotvec,
            x0,
            method="Powell",
            options={"maxiter": int(local_opt_maxiter), "xtol": 1e-4, "ftol": 1e-6},
        )
        if result.success and float(result.fun) <= best_score:
            best_rotation = Rotation.from_rotvec(np.asarray(result.x, dtype=float)).as_matrix()
            best_score = float(result.fun)

    return best_rotation, float(best_score)


def evaluate_candidate(
    sample: PreparedPointCloud,
    candidate_label: str,
    *,
    orientation_mode: str,
    num_axis_samples: int,
    num_angle_samples: int,
    local_opt_maxiter: int,
    exclude_identity: bool,
) -> CandidateEvaluation:
    """Evaluate one candidate point group on one prepared point cloud."""

    label_schoenflies, label_hm = canonical_label_pair(candidate_label)
    operations = csom_operation_matrices(
        candidate_label,
        exclude_identity=exclude_identity,
    )
    orientation_free = all(
        np.allclose(matrix, np.eye(3)) or np.allclose(matrix, -np.eye(3))
        for matrix in operations
    )
    if orientation_mode == "none" or orientation_free:
        rotation_matrix = np.eye(3, dtype=float)
    elif orientation_mode == "search":
        rotation_matrix, _ = optimize_orientation(
            sample.positions,
            sample.species,
            operations,
            num_axis_samples=num_axis_samples,
            num_angle_samples=num_angle_samples,
            local_opt_maxiter=local_opt_maxiter,
        )
    else:
        raise ValueError(f"Unsupported orientation mode: {orientation_mode}")

    oriented = apply_rotation(sample.positions, rotation_matrix)
    per_operation = operation_deviations(oriented, sample.species, operations)
    mean_deviation = float(sum(per_operation) / len(per_operation))
    return CandidateEvaluation(
        label_input=str(candidate_label),
        label_schoenflies=label_schoenflies,
        label_hm=label_hm,
        group_order=int(group_order(candidate_label)),
        mean_deviation=mean_deviation,
        max_deviation=float(max(per_operation)),
        per_operation_deviations=tuple(per_operation),
        rotation_matrix=np.asarray(rotation_matrix, dtype=float),
    )


def select_prediction(
    candidate_results: Sequence[CandidateEvaluation],
    *,
    hard_tol: float,
) -> tuple[CandidateEvaluation, str, int]:
    """Turn target-conditioned CSoM scores into a hard class prediction."""

    if not candidate_results:
        raise ValueError("candidate_results must be non-empty.")

    satisfied = [
        result
        for result in candidate_results
        if result.mean_deviation <= hard_tol
    ]
    if satisfied:
        predicted = min(
            satisfied,
            key=lambda item: (-item.group_order, item.mean_deviation, item.label_schoenflies),
        )
        return predicted, "max_order_below_tol", len(satisfied)

    predicted = min(
        candidate_results,
        key=lambda item: (item.mean_deviation, -item.group_order, item.label_schoenflies),
    )
    return predicted, "min_deviation_fallback", 0


__all__ = [
    "CandidateEvaluation",
    "LabeledPointCloud",
    "PreparedPointCloud",
    "evaluate_candidate",
    "prepare_point_cloud",
    "select_prediction",
]
