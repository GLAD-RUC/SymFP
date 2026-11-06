"""Callable Engel-score baseline model."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import torch
from e3nn import o3

from .common import parse_candidate_groups, parse_manual_center, record_to_sample
from .engel_orientation import (
    ProjectionObjective,
    angles_from_matrix,
    matrix_irrep,
    optimize_projection,
    projector,
    target_operations,
)
from .csom_modules.core import (
    LabeledPointCloud,
    PreparedPointCloud,
    optimize_orientation,
    prepare_point_cloud,
)
from .csom_modules.groups import (
    CandidateTemplate,
    canonical_label_pair,
    group_order,
    normalize_group_label,
)


_ROTATION_EPS = 1e-12


@dataclass(frozen=True)
class EngelGapConfig:
    """Runtime configuration for the Engel-score model."""

    candidate_groups: tuple[str, ...]
    center_mode: str
    manual_center: tuple[float, float, float] | None
    cutoff: float | None
    noise_alpha: float
    noise_seed: int
    max_ell: int
    harmonic_normalization: str
    descriptor_mode: str
    axis_mode: str
    num_axis_samples: int
    num_angle_samples: int
    local_opt_maxiter: int
    hard_tol: float
    exclude_identity: bool
    zero_radius_eps: float
    use_csom_start: bool
    orientation_seed: int
    num_orientation_samples: int
    num_local_starts: int
    orientation_maxiter: int


@dataclass(frozen=True)
class HarmonicDescriptor:
    """Harmonic embedding used by the Engel baseline."""

    blocks: tuple[np.ndarray, ...]
    raw_total_energy: float
    qeq: float
    omega: float
    eee: float
    total_order_parameter: float
    per_degree_energy: tuple[float, ...]
    num_directions_used: int


@dataclass(frozen=True)
class EngelGapCandidateEvaluation:
    """Target-conditioned Engel-score result for one candidate point group."""

    label_input: str
    label_schoenflies: str
    label_hm: str | None
    group_order: int
    engel_score: float
    engel_gap: float
    qdq: float
    qeq: float
    ede: float
    eee: float
    total_order_parameter: float
    alpha: float
    beta: float
    per_degree_alpha: tuple[float, ...]
    per_degree_beta: tuple[float, ...]
    per_degree_energy: tuple[float, ...]
    per_degree_multiplicity: tuple[int, ...]
    rotation_matrix: np.ndarray
    orientation_objective: float | None
    orientation_mode: str = "none"
    search_diagnostics: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class EngelGapSampleResult:
    """Prediction bundle for one point cloud."""

    sample_id: str
    prepared: PreparedPointCloud
    descriptor: HarmonicDescriptor
    candidate_results: tuple[EngelGapCandidateEvaluation, ...]
    predicted: EngelGapCandidateEvaluation
    prediction_mode: str
    satisfied_group_count: int
    true_group_input: str | None
    true_group_schoenflies: str | None
    true_result: EngelGapCandidateEvaluation | None
    correct: int | None
    elapsed_sec: float


@dataclass(frozen=True)
class EngelGapRunResult:
    """Dataset-level Engel-score evaluation output."""

    sample_results: tuple[EngelGapSampleResult, ...]
    prediction_mode_counts: dict[str, int]
    elapsed_sec: float

    @property
    def num_samples(self) -> int:
        return len(self.sample_results)

    @property
    def labeled_results(self) -> tuple[EngelGapSampleResult, ...]:
        return tuple(result for result in self.sample_results if result.correct is not None)

    @property
    def accuracy(self) -> float | None:
        labeled = self.labeled_results
        if not labeled:
            return None
        return float(sum(int(result.correct) for result in labeled) / len(labeled))


@dataclass(frozen=True)
class _ProjectorInfo:
    matrix: np.ndarray
    trace_value: float
    multiplicity: int


def _natural_irrep(ell: int) -> o3.Irrep:
    return o3.Irrep(int(ell), 1 if int(ell) % 2 == 0 else -1)


@lru_cache(maxsize=None)
def _harmonic_block_slices(max_ell: int) -> tuple[slice, ...]:
    offset = 0
    slices: list[slice] = []
    for ell in range(int(max_ell) + 1):
        dim = 2 * ell + 1
        slices.append(slice(offset, offset + dim))
        offset += dim
    return tuple(slices)


@lru_cache(maxsize=None)
def _projector_info(label: str, ell: int) -> _ProjectorInfo:
    normalized_label, _ = canonical_label_pair(label)
    projector_np = projector(normalized_label, ell)
    trace_value = float(np.trace(projector_np))
    multiplicity = int(round(trace_value))
    return _ProjectorInfo(
        matrix=projector_np,
        trace_value=trace_value,
        multiplicity=multiplicity,
    )


def _build_harmonic_descriptor(
    points: np.ndarray,
    *,
    max_ell: int,
    harmonic_normalization: str,
    descriptor_mode: str,
    zero_radius_eps: float,
    edge_index: np.ndarray | None = None,
) -> HarmonicDescriptor:
    points_np = np.asarray(points, dtype=float)
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points_np.shape}.")

    if descriptor_mode == "point_directions":
        direction_chunks = (points_np,)
    elif descriptor_mode == "pair_directions":
        def pair_chunks():
            # Keep every directed pair while bounding temporary harmonic arrays.
            n = len(points_np)
            for start in range(0, n * n, 65536):
                flat = np.arange(start, min(n * n, start + 65536))
                raw = points_np[flat % n] - points_np[flat // n]
                yield raw[~np.all(np.isclose(raw, 0.0, atol=float(zero_radius_eps)), axis=1)]
        direction_chunks = pair_chunks()
    elif descriptor_mode == "knn_edge_directions":
        if edge_index is None:
            raise ValueError("descriptor_mode='knn_edge_directions' requires edge_index.")
        edge_index_np = np.asarray(edge_index, dtype=int)
        if edge_index_np.ndim != 2 or edge_index_np.shape[0] != 2:
            raise ValueError(
                f"Expected edge_index with shape (2, E) for knn_edge_directions, got {edge_index_np.shape}."
            )
        if edge_index_np.shape[1] == 0:
            raise ValueError("descriptor_mode='knn_edge_directions' requires at least one edge.")
        src = edge_index_np[0]
        dst = edge_index_np[1]
        if np.any(src < 0) or np.any(dst < 0):
            raise ValueError("edge_index must be non-negative.")
        num_nodes = int(points_np.shape[0])
        if np.any(src >= num_nodes) or np.any(dst >= num_nodes):
            raise ValueError(
                f"edge_index contains node ids outside [0, {num_nodes - 1}] for points with shape {points_np.shape}."
            )
        direction_chunks = (points_np[dst] - points_np[src],)
    else:
        raise ValueError(f"Unsupported descriptor_mode: {descriptor_mode}")

    coefficient_sum = torch.zeros((int(max_ell) + 1) ** 2, dtype=torch.float64)
    num_directions_used = 0
    for directions_raw in direction_chunks:
        radii = np.linalg.norm(directions_raw, axis=1)
        valid_mask = radii > float(zero_radius_eps)
        count = int(np.count_nonzero(valid_mask))
        if not count:
            continue
        directions = directions_raw[valid_mask] / radii[valid_mask, None]
        sh_values = o3.spherical_harmonics(
            list(range(int(max_ell) + 1)),
            torch.as_tensor(directions, dtype=torch.float64),
            normalize=True,
            normalization=harmonic_normalization,
        )
        coefficient_sum += sh_values.sum(dim=0)
        num_directions_used += count
    if not num_directions_used:
        raise ValueError("Engel baseline requires at least one non-zero direction.")
    coefficients = coefficient_sum / num_directions_used

    blocks: list[np.ndarray] = []
    raw_energies: list[float] = []
    for block_slice in _harmonic_block_slices(int(max_ell)):
        block = coefficients[block_slice]
        raw_energy = float(block.square().sum().item())
        blocks.append(block.detach().cpu().numpy())
        raw_energies.append(raw_energy)

    qeq = float(sum(raw_energies[1:]))
    if qeq <= float(zero_radius_eps):
        raise ValueError("Engel harmonic embedding collapsed to zero energy for ell >= 1.")

    omega = float(1.0 / num_directions_used)
    eee = float(omega * int(max_ell) * (int(max_ell) + 2))
    total_order_parameter = float(qeq / eee - 1.0)

    per_degree_energy: list[float] = [0.0]
    for raw_energy in raw_energies[1:]:
        per_degree_energy.append(float(raw_energy / qeq))
    return HarmonicDescriptor(
        blocks=tuple(blocks),
        raw_total_energy=qeq,
        qeq=qeq,
        omega=omega,
        eee=eee,
        total_order_parameter=total_order_parameter,
        per_degree_energy=tuple(per_degree_energy),
        num_directions_used=num_directions_used,
    )


def _rotate_descriptor(
    descriptor: HarmonicDescriptor,
    rotation_matrix: np.ndarray,
) -> tuple[np.ndarray, ...]:
    angles = angles_from_matrix(rotation_matrix)
    rotated_blocks: list[np.ndarray] = []
    for ell, block in enumerate(descriptor.blocks):
        d_matrix = matrix_irrep(ell, angles)
        rotated_block = d_matrix @ np.asarray(block, dtype=float)
        rotated_blocks.append(np.asarray(rotated_block, dtype=float))
    return tuple(rotated_blocks)


def _uniform_species(count: int) -> tuple[str, ...]:
    return tuple("_" for _ in range(int(count)))


def evaluate_candidate(
    sample: PreparedPointCloud,
    descriptor: HarmonicDescriptor,
    candidate_label: str,
    *,
    axis_mode: str,
    num_axis_samples: int,
    num_angle_samples: int,
    local_opt_maxiter: int,
    exclude_identity: bool,
    use_csom_start: bool = True,
    orientation_seed: int = 20260920,
    num_orientation_samples: int = 128,
    num_local_starts: int = 12,
    orientation_maxiter: int = 200,
) -> EngelGapCandidateEvaluation:
    """Evaluate one target; `engel_search` maximizes retained energy.

    `none` uses the given frame. `csom_search` only fits coordinate deviation
    and is retained as an explicit diagnostic, not as Engel maximization.
    `orientation_objective` is discarded-energy fraction for `engel_search`,
    coordinate deviation for `csom_search`, and None for the given frame.
    """

    label_schoenflies, label_hm = canonical_label_pair(candidate_label)
    if axis_mode not in {"none", "csom_search", "engel_search"}:
        raise ValueError(f"Unsupported axis_mode: {axis_mode}")
    rotation_matrix = np.eye(3, dtype=float)
    orientation_objective = None
    diagnostics = ()
    objective = ProjectionObjective(descriptor.blocks, label_schoenflies) if axis_mode == "engel_search" else None
    needs_coordinate_fit = axis_mode == "csom_search" or (
        axis_mode == "engel_search" and use_csom_start and bool(objective.items)
    )
    if needs_coordinate_fit:
        operations = target_operations(label_schoenflies)
        if exclude_identity and len(operations) > 1:
            operations = tuple(op for op in operations if not np.allclose(op, np.eye(3)))
        # A single non-identity operation still carries an unknown orientation:
        # Cs has one mirror and C2 has one half-turn, whose plane/axis must be
        # fitted. Identity and inversion alone commute with every rotation.
        orientation_free = all(
            np.allclose(op, np.eye(3)) or np.allclose(op, -np.eye(3))
            for op in operations
        )
        if orientation_free:
            rotation_matrix = np.eye(3, dtype=float)
            orientation_objective = 0.0
        else:
            rotation_matrix, orientation_objective = optimize_orientation(
                sample.positions,
                _uniform_species(sample.positions.shape[0]),
                operations,
                num_axis_samples=num_axis_samples,
                num_angle_samples=num_angle_samples,
                local_opt_maxiter=local_opt_maxiter,
            )
    if axis_mode == "engel_search":
        rotation_matrix, orientation_objective, diagnostics = optimize_projection(
            objective,
            warm=rotation_matrix if needs_coordinate_fit else None,
            seed=orientation_seed,
            count=num_orientation_samples,
            local_starts=num_local_starts,
            maxiter=orientation_maxiter,
        )

    rotated_blocks = _rotate_descriptor(descriptor, rotation_matrix)

    qdq = 0.0
    discarded_energy = 0.0
    ede_trace_sum = 0.0
    per_degree_alpha: list[float] = [0.0]
    per_degree_beta: list[float] = [0.0]
    per_degree_multiplicity: list[int] = [0]
    for ell, block in enumerate(rotated_blocks):
        if ell == 0:
            continue
        projector_info = _projector_info(label_schoenflies, ell)
        projected = projector_info.matrix @ block
        residual = block - projected
        discarded_energy += float(residual @ residual)
        retained = float(projected @ projected)
        total_energy = float(block @ block)
        retained = min(max(retained, 0.0), total_energy + 1e-10)
        multiplicity = int(projector_info.multiplicity)
        dim = 2 * ell + 1
        per_degree_alpha.append(float(retained / descriptor.qeq))
        per_degree_beta.append(float(multiplicity / dim))
        per_degree_multiplicity.append(multiplicity)
        qdq += retained
        ede_trace_sum += float(projector_info.trace_value)

    ede = float(descriptor.omega * ede_trace_sum)
    alpha = float(qdq / descriptor.qeq)
    beta = float(ede / descriptor.eee)
    if abs(1.0 - beta) <= _ROTATION_EPS:
        engel_score = 1.0 if abs(alpha - beta) <= 1e-8 else float("nan")
        engel_gap = 0.0 if np.isfinite(engel_score) else float("nan")
    else:
        engel_gap = float(discarded_energy / descriptor.qeq / (1.0 - beta))
        engel_score = float(1.0 - engel_gap)
    if axis_mode == "engel_search":
        orientation_objective = float(discarded_energy / descriptor.qeq)

    return EngelGapCandidateEvaluation(
        label_input=str(candidate_label),
        label_schoenflies=label_schoenflies,
        label_hm=label_hm,
        group_order=int(group_order(candidate_label)),
        engel_score=engel_score,
        engel_gap=engel_gap,
        qdq=float(qdq),
        qeq=float(descriptor.qeq),
        ede=ede,
        eee=float(descriptor.eee),
        total_order_parameter=float(descriptor.total_order_parameter),
        alpha=float(alpha),
        beta=beta,
        per_degree_alpha=tuple(float(value) for value in per_degree_alpha),
        per_degree_beta=tuple(float(value) for value in per_degree_beta),
        per_degree_energy=tuple(float(value) for value in descriptor.per_degree_energy),
        per_degree_multiplicity=tuple(int(value) for value in per_degree_multiplicity),
        rotation_matrix=np.asarray(rotation_matrix, dtype=float),
        orientation_objective=None if orientation_objective is None else float(orientation_objective),
        orientation_mode=axis_mode,
        search_diagnostics=diagnostics,
    )


def select_prediction(
    candidate_results: Sequence[EngelGapCandidateEvaluation],
    *,
    hard_tol: float,
) -> tuple[EngelGapCandidateEvaluation, str, int]:
    """Turn target-conditioned Engel scores into a hard class prediction."""

    if not candidate_results:
        raise ValueError("candidate_results must be non-empty.")

    satisfied = [
        result
        for result in candidate_results
        if np.isfinite(result.engel_gap) and result.engel_gap <= hard_tol
    ]
    if satisfied:
        predicted = min(
            satisfied,
            key=lambda item: (-item.group_order, item.engel_gap, item.label_schoenflies),
        )
        return predicted, "max_order_below_tol", len(satisfied)

    predicted = min(
        candidate_results,
        key=lambda item: (
            float("inf") if not np.isfinite(item.engel_gap) else item.engel_gap,
            -item.group_order,
            item.label_schoenflies,
        ),
    )
    return predicted, "min_gap_fallback", 0


class EngelGap:
    """Engel score with shared-orientation retained-energy search by default.

    The default reproduces the enhanced manuscript protocol: identity + 128
    Haar rotations, nested BFGS refinement and a CSoM-fitted initial frame.
    Set `use_csom_start=False` to omit that extra initial frame, or
    `axis_mode='none'` to evaluate a supplied orientation without searching.
    """

    def __init__(
        self,
        *,
        candidate_groups: str | Sequence[str] = "toy32",
        center_mode: str = "centroid",
        manual_center: str | Sequence[float] | None = None,
        cutoff: float | None = None,
        noise_alpha: float = 0.0,
        noise_seed: int = 42,
        max_ell: int = 6,
        harmonic_normalization: str = "integral",
        descriptor_mode: str = "pair_directions",
        axis_mode: str = "engel_search",
        num_axis_samples: int = 10,
        num_angle_samples: int = 2,
        local_opt_maxiter: int = 100,
        hard_tol: float = 1e-4,
        exclude_identity: bool = True,
        zero_radius_eps: float = 1e-12,
        use_csom_start: bool = True,
        orientation_seed: int = 20260920,
        num_orientation_samples: int = 128,
        num_local_starts: int = 12,
        orientation_maxiter: int = 200,
    ) -> None:
        center_mode = str(center_mode)
        if center_mode not in {"centroid", "first_atom", "manual", "none"}:
            raise ValueError(f"Unsupported center_mode: {center_mode}")
        axis_mode = str(axis_mode)
        if axis_mode not in {"none", "csom_search", "engel_search"}:
            raise ValueError(f"Unsupported axis_mode: {axis_mode}")
        harmonic_normalization = str(harmonic_normalization)
        if harmonic_normalization not in {"integral", "component", "norm"}:
            raise ValueError(f"Unsupported harmonic_normalization: {harmonic_normalization}")
        descriptor_mode = str(descriptor_mode)
        if descriptor_mode not in {"pair_directions", "point_directions", "knn_edge_directions"}:
            raise ValueError(f"Unsupported descriptor_mode: {descriptor_mode}")
        if int(max_ell) < 1:
            raise ValueError(f"max_ell must be positive, got {max_ell}.")
        if int(num_orientation_samples) < 32 or int(num_local_starts) < 1 or int(orientation_maxiter) < 1:
            raise ValueError("Search requires at least 32 samples, one local start and one iteration.")

        parsed_groups = tuple(parse_candidate_groups(candidate_groups))
        parsed_center = parse_manual_center(manual_center)
        self.config = EngelGapConfig(
            candidate_groups=parsed_groups,
            center_mode=center_mode,
            manual_center=parsed_center,
            cutoff=None if cutoff is None else float(cutoff),
            noise_alpha=float(noise_alpha),
            noise_seed=int(noise_seed),
            max_ell=int(max_ell),
            harmonic_normalization=harmonic_normalization,
            descriptor_mode=descriptor_mode,
            axis_mode=axis_mode,
            num_axis_samples=int(num_axis_samples),
            num_angle_samples=int(num_angle_samples),
            local_opt_maxiter=int(local_opt_maxiter),
            hard_tol=float(hard_tol),
            exclude_identity=bool(exclude_identity),
            zero_radius_eps=float(zero_radius_eps),
            use_csom_start=bool(use_csom_start),
            orientation_seed=int(orientation_seed),
            num_orientation_samples=int(num_orientation_samples),
            num_local_starts=int(num_local_starts),
            orientation_maxiter=int(orientation_maxiter),
        )
        self.candidate_templates = tuple(
            CandidateTemplate.from_label(label)
            for label in self.config.candidate_groups
        )

    def _coerce_sample(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        index: int | None = None,
    ) -> LabeledPointCloud:
        if isinstance(sample, LabeledPointCloud):
            return sample
        if isinstance(sample, dict):
            return record_to_sample(sample, index=index)
        raise TypeError(
            "EngelGap expects each sample to be a LabeledPointCloud or a toy-style record dict."
        )

    def prepare_sample(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> PreparedPointCloud:
        labeled = self._coerce_sample(sample, index=index)
        return prepare_point_cloud(
            labeled,
            split_tag=split_tag,
            center_mode=self.config.center_mode,
            manual_center=self.config.manual_center,
            cutoff=self.config.cutoff,
            noise_alpha=self.config.noise_alpha,
            noise_seed=self.config.noise_seed,
        )

    def evaluate_sample(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> EngelGapSampleResult:
        sample_start = time.perf_counter()
        prepared = self.prepare_sample(sample, split_tag=split_tag, index=index)
        descriptor = _build_harmonic_descriptor(
            prepared.positions,
            max_ell=self.config.max_ell,
            harmonic_normalization=self.config.harmonic_normalization,
            descriptor_mode=self.config.descriptor_mode,
            zero_radius_eps=self.config.zero_radius_eps,
            edge_index=prepared.edge_index,
        )

        candidate_results = tuple(
            evaluate_candidate(
                prepared,
                descriptor,
                candidate_label,
                axis_mode=self.config.axis_mode,
                num_axis_samples=self.config.num_axis_samples,
                num_angle_samples=self.config.num_angle_samples,
                local_opt_maxiter=self.config.local_opt_maxiter,
                exclude_identity=self.config.exclude_identity,
                use_csom_start=self.config.use_csom_start,
                orientation_seed=self.config.orientation_seed,
                num_orientation_samples=self.config.num_orientation_samples,
                num_local_starts=self.config.num_local_starts,
                orientation_maxiter=self.config.orientation_maxiter,
            )
            for candidate_label in self.config.candidate_groups
        )
        predicted, prediction_mode, satisfied_count = select_prediction(
            candidate_results,
            hard_tol=self.config.hard_tol,
        )

        true_group_input = prepared.true_group if prepared.true_group else None
        true_group_schoenflies = None
        true_result = None
        if true_group_input is not None:
            true_group_schoenflies = normalize_group_label(true_group_input)
            true_result = next(
                (
                    result
                    for result in candidate_results
                    if result.label_schoenflies == true_group_schoenflies
                ),
                None,
            )

        correct = None
        if true_group_schoenflies is not None:
            correct = int(predicted.label_schoenflies == true_group_schoenflies)

        elapsed_sec = float(time.perf_counter() - sample_start)
        return EngelGapSampleResult(
            sample_id=prepared.sample_id,
            prepared=prepared,
            descriptor=descriptor,
            candidate_results=candidate_results,
            predicted=predicted,
            prediction_mode=prediction_mode,
            satisfied_group_count=int(satisfied_count),
            true_group_input=true_group_input,
            true_group_schoenflies=true_group_schoenflies,
            true_result=true_result,
            correct=correct,
            elapsed_sec=elapsed_sec,
        )

    def __call__(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> EngelGapSampleResult:
        return self.evaluate_sample(sample, split_tag=split_tag, index=index)

    def predict_label(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
        label_space: str = "input",
    ) -> str | None:
        result = self.evaluate_sample(sample, split_tag=split_tag, index=index)
        label_space = str(label_space)
        if label_space == "input":
            return result.predicted.label_input
        if label_space == "hm":
            return result.predicted.label_hm
        if label_space == "schoenflies":
            return result.predicted.label_schoenflies
        raise ValueError(f"Unsupported label_space: {label_space}")

    def evaluate_samples(
        self,
        samples: Sequence[LabeledPointCloud | dict[str, Any]],
        *,
        split_tag: str = "eval",
        report_every: int = 0,
    ) -> EngelGapRunResult:
        sample_list = list(samples)
        prediction_mode_counts: Counter[str] = Counter()
        sample_results: list[EngelGapSampleResult] = []

        run_start = time.perf_counter()
        for sample_index, sample in enumerate(sample_list):
            result = self.evaluate_sample(
                sample,
                split_tag=split_tag,
                index=sample_index,
            )
            sample_results.append(result)
            prediction_mode_counts[result.prediction_mode] += 1

            if report_every > 0 and (
                (sample_index + 1) % report_every == 0
                or (sample_index + 1) == len(sample_list)
            ):
                print(
                    f"[engel-score] processed {sample_index + 1}/{len(sample_list)} samples "
                    f"(last_id={result.sample_id}, pred={result.predicted.label_input}, "
                    f"mode={result.prediction_mode})",
                    flush=True,
                )

        elapsed_sec = float(time.perf_counter() - run_start)
        return EngelGapRunResult(
            sample_results=tuple(sample_results),
            prediction_mode_counts=dict(prediction_mode_counts),
            elapsed_sec=elapsed_sec,
        )


__all__ = [
    "EngelGap",
    "EngelGapCandidateEvaluation",
    "EngelGapConfig",
    "EngelGapRunResult",
    "EngelGapSampleResult",
    "HarmonicDescriptor",
    "evaluate_candidate",
    "select_prediction",
]
