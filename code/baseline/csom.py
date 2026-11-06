"""Thin wrapper around the vendored official CSoM implementation."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np

from .common import (
    ExternalToolExecutionError,
    ExternalToolUnavailableError,
    parse_candidate_groups,
    parse_manual_center,
    prepare_external_point_cloud,
    record_to_sample,
)
from .csom_modules.core import LabeledPointCloud, PreparedPointCloud, evaluate_candidate
from .csom_modules.groups import (
    CandidateTemplate,
    canonical_label_pair,
    group_order,
    normalize_group_label,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OFFICIAL_ROOT = (
    _PROJECT_ROOT
    / ".vendor"
    / "Continous-Symmetry-Operation-Measure-Program-main"
    / "Continous-Symmetry-Operation-Measure"
)


@dataclass(frozen=True)
class OfficialCSoMConfig:
    """Runtime configuration for the official CSoM wrapper."""

    candidate_groups: tuple[str, ...]
    center_mode: str
    manual_center: tuple[float, float, float] | None
    cutoff: float | None
    noise_alpha: float
    noise_seed: int
    orientation_mode: str
    num_axis_samples: int
    num_angle_samples: int
    hard_tol: float
    official_root: str | None
    suppress_official_stdout: bool


@dataclass(frozen=True)
class OfficialCSoMCandidateEvaluation:
    """Target-conditioned official CSoM result for one candidate point group."""

    label_input: str
    label_schoenflies: str
    label_hm: str | None
    official_group_name: str
    group_order: int
    mean_deviation: float
    max_deviation: float
    per_operation_deviations: tuple[float, ...]
    per_operation_names: tuple[str, ...]
    rotation_matrix: np.ndarray
    orientation_axis: np.ndarray
    orientation_angle_deg: float


@dataclass(frozen=True)
class OfficialCSoMSampleResult:
    """Prediction bundle for one point cloud."""

    sample_id: str
    prepared: PreparedPointCloud
    candidate_results: tuple[OfficialCSoMCandidateEvaluation, ...]
    predicted: OfficialCSoMCandidateEvaluation
    prediction_mode: str
    satisfied_group_count: int
    true_group_input: str | None
    true_group_schoenflies: str | None
    true_result: OfficialCSoMCandidateEvaluation | None
    correct: int | None
    elapsed_sec: float


@dataclass(frozen=True)
class OfficialCSoMRunResult:
    """Dataset-level official CSoM evaluation output."""

    sample_results: tuple[OfficialCSoMSampleResult, ...]
    prediction_mode_counts: dict[str, int]
    elapsed_sec: float

    @property
    def num_samples(self) -> int:
        return len(self.sample_results)

    @property
    def labeled_results(self) -> tuple[OfficialCSoMSampleResult, ...]:
        return tuple(result for result in self.sample_results if result.correct is not None)

    @property
    def accuracy(self) -> float | None:
        labeled = self.labeled_results
        if not labeled:
            return None
        return float(sum(int(result.correct) for result in labeled) / len(labeled))


def _resolve_official_root(official_root: str | Path | None) -> Path:
    resolved = _DEFAULT_OFFICIAL_ROOT if official_root is None else Path(official_root).resolve()
    if not resolved.is_dir():
        raise ExternalToolUnavailableError(
            "Vendored official CSoM program not found. "
            f"Expected directory: {resolved}"
        )
    module_path = resolved / "CSoM_functions.py"
    operations_dir = resolved / "Operations"
    if not module_path.is_file() or not operations_dir.is_dir():
        raise ExternalToolUnavailableError(
            "Vendored official CSoM program is incomplete. "
            f"Expected files under: {resolved}"
        )
    return resolved


@lru_cache(maxsize=None)
def _supported_official_group_map(official_root_text: str) -> dict[str, str]:
    official_root = _resolve_official_root(official_root_text)
    mapping: dict[str, str] = {}
    for csv_path in sorted((official_root / "Operations").glob("pointgroup_*.csv")):
        stem = csv_path.stem.removeprefix("pointgroup_")
        canonical = "C1" if stem == "E" else normalize_group_label(stem)
        mapping[canonical] = csv_path.stem
    return mapping


def official_csom_available(official_root: str | Path | None = None) -> bool:
    """Whether the authors' CSoM program is present and complete."""

    try:
        _resolve_official_root(official_root)
    except ExternalToolUnavailableError:
        return False
    return True


_bundled_fallback_announced = False


def _warn_bundled_fallback_once() -> None:
    global _bundled_fallback_announced
    if _bundled_fallback_announced:
        return
    _bundled_fallback_announced = True
    print(
        "[csom] The authors' CSoM program was not found under .vendor/, so the bundled "
        "implementation in baseline/csom_modules/ is used instead.",
        file=sys.stderr,
        flush=True,
    )


_BUNDLED_ORIENTATION_MODES = {"none": "none", "official_search": "search"}

# Bundled-search settings are independent of the official-search settings.
_BUNDLED_MIN_AXIS_SAMPLES = 16
_BUNDLED_MIN_ANGLE_SAMPLES = 4
_BUNDLED_LOCAL_OPT_MAXITER = 200


def _rotation_axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    """Axis and angle in degrees of a proper rotation matrix."""

    trace = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(trace)))
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0]), angle
    return axis / norm, angle


def evaluate_bundled_csom_candidate(
    prepared: PreparedPointCloud,
    candidate_label: str,
    *,
    orientation_mode: str,
    num_axis_samples: int,
    num_angle_samples: int,
    local_opt_maxiter: int = _BUNDLED_LOCAL_OPT_MAXITER,
    exclude_identity: bool = True,
) -> OfficialCSoMCandidateEvaluation:
    """Run the bundled CSoM implementation and report it in the wrapper's result type."""

    if orientation_mode not in _BUNDLED_ORIENTATION_MODES:
        raise ValueError(f"Unsupported orientation_mode: {orientation_mode}")

    if orientation_mode == "official_search":
        num_axis_samples = max(int(num_axis_samples), _BUNDLED_MIN_AXIS_SAMPLES)
        num_angle_samples = max(int(num_angle_samples), _BUNDLED_MIN_ANGLE_SAMPLES)

    evaluation = evaluate_candidate(
        prepared,
        candidate_label,
        orientation_mode=_BUNDLED_ORIENTATION_MODES[orientation_mode],
        num_axis_samples=int(num_axis_samples),
        num_angle_samples=int(num_angle_samples),
        local_opt_maxiter=int(local_opt_maxiter),
        exclude_identity=bool(exclude_identity),
    )
    axis, angle_deg = _rotation_axis_angle(evaluation.rotation_matrix)
    operation_count = len(evaluation.per_operation_deviations)
    return OfficialCSoMCandidateEvaluation(
        label_input=evaluation.label_input,
        label_schoenflies=evaluation.label_schoenflies,
        label_hm=evaluation.label_hm,
        official_group_name=evaluation.label_schoenflies,
        group_order=evaluation.group_order,
        mean_deviation=evaluation.mean_deviation,
        max_deviation=evaluation.max_deviation,
        per_operation_deviations=evaluation.per_operation_deviations,
        per_operation_names=tuple(f"op{i + 1}" for i in range(operation_count)),
        rotation_matrix=evaluation.rotation_matrix,
        orientation_axis=axis,
        orientation_angle_deg=angle_deg,
    )


def supported_official_csom_groups(
    official_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return Schoenflies labels supported by the vendored official CSoM code."""

    mapping = _supported_official_group_map(str(_resolve_official_root(official_root)))
    return tuple(mapping.keys())


def parse_official_csom_candidate_groups(
    candidate_groups: str | Sequence[str],
    *,
    official_root: str | Path | None = None,
) -> list[str]:
    """Parse a candidate-group preset or explicit label list."""

    if isinstance(candidate_groups, str) and candidate_groups.strip().lower() in {
        "official_all",
        "official47",
        "all",
    }:
        return list(supported_official_csom_groups(official_root))

    labels = parse_candidate_groups(candidate_groups)
    supported = _supported_official_group_map(str(_resolve_official_root(official_root)))
    unsupported = [
        label
        for label in labels
        if normalize_group_label(label) not in supported
    ]
    if unsupported:
        raise ValueError(
            "Official CSoM does not ship the following point groups: "
            + ", ".join(str(label) for label in unsupported)
        )
    return labels


@lru_cache(maxsize=None)
def _load_official_csom_module(official_root_text: str) -> ModuleType:
    official_root = _resolve_official_root(official_root_text)
    module_path = official_root / "CSoM_functions.py"
    spec = importlib.util.spec_from_file_location(
        f"_symfp_official_csom_{abs(hash(str(module_path)))}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ExternalToolUnavailableError(
            f"Unable to load vendored official CSoM module from: {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ExternalToolUnavailableError(
            "Failed to import vendored official CSoM dependencies. "
            f"Module path: {module_path}"
        ) from exc
    return module


@contextlib.contextmanager
def _maybe_silence(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield


def _select_prediction(
    candidate_results: Sequence[OfficialCSoMCandidateEvaluation],
    *,
    hard_tol: float,
) -> tuple[OfficialCSoMCandidateEvaluation, str, int]:
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


def evaluate_official_csom_candidate(
    module: ModuleType,
    sample: PreparedPointCloud,
    candidate_label: str,
    *,
    orientation_mode: str,
    num_axis_samples: int,
    num_angle_samples: int,
    suppress_official_stdout: bool,
    official_root: str | Path | None = None,
) -> OfficialCSoMCandidateEvaluation:
    """Evaluate one candidate point group with the vendored official CSoM core."""

    label_schoenflies, label_hm = canonical_label_pair(candidate_label)
    supported = _supported_official_group_map(str(_resolve_official_root(official_root)))
    if label_schoenflies not in supported:
        raise ValueError(f"Unsupported official CSoM point group: {candidate_label}")
    official_group_name = supported[label_schoenflies]

    try:
        with _maybe_silence(suppress_official_stdout):
            operation_names, operations = module.ReadSymmetryFile(
                official_group_name,
                module.Path_To_Symmetry_Operations,
            )

            oriented = np.asarray(sample.positions, dtype=float)
            rotation_matrix = np.eye(3, dtype=float)
            orientation_axis = np.array([0.0, 0.0, 1.0], dtype=float)
            orientation_angle_deg = 0.0

            if orientation_mode == "official_search" and official_group_name != "pointgroup_E":
                (
                    oriented,
                    rotation_matrix,
                    orientation_axis,
                    orientation_angle_deg,
                    _,
                ) = module.find_zaxis(
                    oriented,
                    np.asarray(sample.species, dtype=object),
                    operations,
                    N_input_zaxis=max(1, int(num_axis_samples)),
                    N_input_angles=max(1, int(num_angle_samples)),
                )
            elif orientation_mode != "none":
                raise ValueError(f"Unsupported orientation_mode: {orientation_mode}")

            per_operation_deviations = module.point_group_deviation(
                np.asarray(oriented, dtype=float),
                np.asarray(sample.species, dtype=object),
                operations,
                official_group_name,
                f"{sample.sample_id}.xyz",
                operation_names,
                False,
            )
    except Exception as exc:
        raise ExternalToolExecutionError(
            "Vendored official CSoM evaluation failed for "
            f"sample={sample.sample_id!r}, candidate={candidate_label!r}."
        ) from exc

    per_operation_tuple = tuple(float(value) for value in per_operation_deviations)
    return OfficialCSoMCandidateEvaluation(
        label_input=str(candidate_label),
        label_schoenflies=label_schoenflies,
        label_hm=label_hm,
        official_group_name=official_group_name,
        group_order=int(group_order(candidate_label)),
        mean_deviation=float(sum(per_operation_tuple) / len(per_operation_tuple)),
        max_deviation=float(max(per_operation_tuple)),
        per_operation_deviations=per_operation_tuple,
        per_operation_names=tuple(str(name) for name in operation_names),
        rotation_matrix=np.asarray(rotation_matrix, dtype=float),
        orientation_axis=np.asarray(orientation_axis, dtype=float),
        orientation_angle_deg=float(orientation_angle_deg),
    )


class OfficialCSoM:
    """Convenient callable wrapper for vendored official CSoM evaluation."""

    def __init__(
        self,
        *,
        candidate_groups: str | Sequence[str] = "toy32",
        center_mode: str = "first_atom",
        manual_center: str | Sequence[float] | None = None,
        cutoff: float | None = None,
        noise_alpha: float = 0.0,
        noise_seed: int = 42,
        orientation_mode: str = "official_search",
        num_axis_samples: int = 10,
        num_angle_samples: int = 2,
        hard_tol: float = 1e-4,
        official_root: str | Path | None = None,
        suppress_official_stdout: bool = True,
    ) -> None:
        center_mode = str(center_mode)
        if center_mode not in {"centroid", "first_atom", "manual", "none"}:
            raise ValueError(f"Unsupported center_mode: {center_mode}")

        orientation_mode = str(orientation_mode)
        if orientation_mode not in {"none", "official_search"}:
            raise ValueError(f"Unsupported orientation_mode: {orientation_mode}")

        self.uses_official = official_csom_available(official_root)
        if self.uses_official:
            parsed_groups = tuple(
                parse_official_csom_candidate_groups(
                    candidate_groups,
                    official_root=official_root,
                )
            )
        else:
            parsed_groups = tuple(parse_candidate_groups(candidate_groups))
            _warn_bundled_fallback_once()
        parsed_center = parse_manual_center(manual_center)

        self.config = OfficialCSoMConfig(
            candidate_groups=parsed_groups,
            center_mode=center_mode,
            manual_center=parsed_center,
            cutoff=None if cutoff is None else float(cutoff),
            noise_alpha=float(noise_alpha),
            noise_seed=int(noise_seed),
            orientation_mode=orientation_mode,
            num_axis_samples=int(num_axis_samples),
            num_angle_samples=int(num_angle_samples),
            hard_tol=float(hard_tol),
            official_root=None if official_root is None else str(Path(official_root).resolve()),
            suppress_official_stdout=bool(suppress_official_stdout),
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
            "OfficialCSoM expects each sample to be a LabeledPointCloud or a toy-style record dict."
        )

    @property
    def official_module(self) -> ModuleType:
        official_root = _resolve_official_root(self.config.official_root)
        return _load_official_csom_module(str(official_root))

    def prepare_sample(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> PreparedPointCloud:
        labeled = self._coerce_sample(sample, index=index)
        return prepare_external_point_cloud(
            labeled,
            split_tag=split_tag,
            center_mode=self.config.center_mode,
            manual_center=self.config.manual_center,
            cutoff=self.config.cutoff,
            noise_alpha=self.config.noise_alpha,
            noise_seed=self.config.noise_seed,
            cutoff_inclusive=False,
        )

    def evaluate_prepared_candidate(
        self,
        prepared: PreparedPointCloud,
        candidate_label: str,
    ) -> OfficialCSoMCandidateEvaluation:
        """Evaluate one candidate, through the authors' program when it is available."""

        if self.uses_official:
            return evaluate_official_csom_candidate(
                self.official_module,
                prepared,
                candidate_label,
                orientation_mode=self.config.orientation_mode,
                num_axis_samples=self.config.num_axis_samples,
                num_angle_samples=self.config.num_angle_samples,
                suppress_official_stdout=self.config.suppress_official_stdout,
                official_root=self.config.official_root,
            )
        return evaluate_bundled_csom_candidate(
            prepared,
            candidate_label,
            orientation_mode=self.config.orientation_mode,
            num_axis_samples=self.config.num_axis_samples,
            num_angle_samples=self.config.num_angle_samples,
        )

    def evaluate_sample(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> OfficialCSoMSampleResult:
        sample_start = time.perf_counter()
        prepared = self.prepare_sample(sample, split_tag=split_tag, index=index)

        candidate_results = tuple(
            self.evaluate_prepared_candidate(prepared, candidate_label)
            for candidate_label in self.config.candidate_groups
        )
        predicted, prediction_mode, satisfied_count = _select_prediction(
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
        return OfficialCSoMSampleResult(
            sample_id=prepared.sample_id,
            prepared=prepared,
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
    ) -> OfficialCSoMSampleResult:
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
    ) -> OfficialCSoMRunResult:
        sample_list = list(samples)
        prediction_mode_counts: Counter[str] = Counter()
        sample_results: list[OfficialCSoMSampleResult] = []

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
                    f"[official_csom] processed {sample_index + 1}/{len(sample_list)} samples "
                    f"(last_id={result.sample_id}, pred={result.predicted.label_input}, "
                    f"mode={result.prediction_mode})",
                    flush=True,
                )

        elapsed_sec = float(time.perf_counter() - run_start)
        return OfficialCSoMRunResult(
            sample_results=tuple(sample_results),
            prediction_mode_counts=dict(prediction_mode_counts),
            elapsed_sec=elapsed_sec,
        )


CSoM = OfficialCSoM
CSoMCandidateEvaluation = OfficialCSoMCandidateEvaluation
CSoMConfig = OfficialCSoMConfig
CSoMRunResult = OfficialCSoMRunResult
CSoMSampleResult = OfficialCSoMSampleResult
evaluate_csom_candidate = evaluate_official_csom_candidate
parse_csom_candidate_groups = parse_official_csom_candidate_groups
supported_csom_groups = supported_official_csom_groups


__all__ = [
    "CSoM",
    "CSoMCandidateEvaluation",
    "CSoMConfig",
    "CSoMRunResult",
    "CSoMSampleResult",
    "OfficialCSoM",
    "OfficialCSoMCandidateEvaluation",
    "OfficialCSoMConfig",
    "OfficialCSoMRunResult",
    "OfficialCSoMSampleResult",
    "evaluate_csom_candidate",
    "evaluate_official_csom_candidate",
    "parse_csom_candidate_groups",
    "parse_official_csom_candidate_groups",
    "supported_csom_groups",
    "supported_official_csom_groups",
]
