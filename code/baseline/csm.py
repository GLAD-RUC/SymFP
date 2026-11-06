"""Thin wrapper around the official Open University CSM exact implementation."""

from __future__ import annotations

import contextlib
import importlib
import io
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import openbabel.openbabel as ob

from .common import (
    ExternalToolExecutionError,
    ExternalToolUnavailableError,
    build_work_dir,
    parse_label_list,
    parse_manual_center,
    prepare_external_point_cloud,
    record_to_sample,
    write_xyz,
)
from .csom_modules.core import LabeledPointCloud, PreparedPointCloud
from .csom_modules.groups import canonical_label_pair, normalize_group_label


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OFFICIAL_SOURCE_ROOT = _PROJECT_ROOT / ".vendor" / "official_csm_repo" / "src"
_CN_RE = re.compile(r"^C(\d+)$")
_SN_RE = re.compile(r"^S(\d+)$")


@dataclass(frozen=True)
class OfficialExactCSMConfig:
    """Runtime configuration for the official exact CSM wrapper."""

    candidate_groups: tuple[str, ...]
    center_mode: str
    manual_center: tuple[float, float, float] | None
    cutoff: float | None
    noise_alpha: float
    noise_seed: int
    keep_structure: bool
    babel_bond: bool
    use_mass: bool
    ignore_symmetry_labels: bool
    remove_hydrogen: bool
    timeout_sec: int
    hard_tol: float
    chirality_sn_max: int
    official_source_root: str | None
    work_root: str | None
    suppress_official_stdout: bool


@dataclass(frozen=True)
class OfficialExactCSMCandidateEvaluation:
    """Target-conditioned exact CSM result for one candidate operation."""

    label_input: str
    label_schoenflies: str
    label_hm: str | None
    operation_code: str
    operation_type: str
    operation_order: int
    csm_value: float
    formula_csm: float
    permutation: tuple[int, ...]
    direction: tuple[float, ...]
    num_invalid_cycles: int
    structure_preservation: float | str | None
    command: tuple[str, ...]
    result_path: str | None


@dataclass(frozen=True)
class OfficialExactCSMSampleResult:
    """Prediction bundle for one point cloud."""

    sample_id: str
    prepared: PreparedPointCloud
    candidate_results: tuple[OfficialExactCSMCandidateEvaluation, ...]
    predicted: OfficialExactCSMCandidateEvaluation
    prediction_mode: str
    satisfied_group_count: int
    true_label_input: str | None
    true_result: OfficialExactCSMCandidateEvaluation | None
    correct: int | None
    elapsed_sec: float


@dataclass(frozen=True)
class OfficialExactCSMRunResult:
    """Dataset-level exact CSM evaluation output."""

    sample_results: tuple[OfficialExactCSMSampleResult, ...]
    prediction_mode_counts: dict[str, int]
    elapsed_sec: float

    @property
    def num_samples(self) -> int:
        return len(self.sample_results)

    @property
    def labeled_results(self) -> tuple[OfficialExactCSMSampleResult, ...]:
        return tuple(result for result in self.sample_results if result.correct is not None)

    @property
    def accuracy(self) -> float | None:
        labeled = self.labeled_results
        if not labeled:
            return None
        return float(sum(int(result.correct) for result in labeled) / len(labeled))


def _official_install_hint() -> str:
    return (
        "Official CSM was not found. Place the source tree under "
        f"{_DEFAULT_OFFICIAL_SOURCE_ROOT}, or install the `csm` package and make sure "
        "its compiled `csm.fast` extension is available."
    )


def _maybe_source_root(source_root: str | Path | None) -> Path | None:
    if source_root is None:
        if _DEFAULT_OFFICIAL_SOURCE_ROOT.is_dir():
            return _DEFAULT_OFFICIAL_SOURCE_ROOT
        return None
    resolved = Path(source_root).resolve()
    if not resolved.is_dir():
        raise ExternalToolUnavailableError(
            f"Official CSM source root not found: {resolved}. {_official_install_hint()}"
        )
    return resolved


def _ensure_official_csm_on_path(source_root: str | Path | None) -> None:
    resolved = _maybe_source_root(source_root)
    if resolved is None:
        return
    source_text = str(resolved)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)


def _load_official_csm_runner(source_root: str | Path | None) -> tuple[Any, Any]:
    _ensure_official_csm_on_path(source_root)
    try:
        module = importlib.import_module("csm.main.csm_run")
    except Exception as exc:
        raise ExternalToolUnavailableError(_official_install_hint()) from exc
    if not hasattr(module, "get_parsed_args") or not hasattr(module, "calc"):
        raise ExternalToolUnavailableError(
            "The imported `csm.main.csm_run` module does not expose the expected "
            "`get_parsed_args` and `calc` entry points."
        )
    return module.get_parsed_args, module.calc


@contextlib.contextmanager
def _maybe_silence(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield


def _official_symbol(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Atomic label must be non-empty.")
    if text.isdigit():
        atomic_number = int(text)
        if atomic_number <= 0:
            raise ValueError(f"Invalid atomic number label: {text!r}.")
        return str(ob.GetSymbol(atomic_number))
    return text


def _official_symbols(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_official_symbol(value) for value in values)


def _normalize_operation_label(label: str, *, chirality_sn_max: int) -> tuple[str, str, str | None]:
    lowered = str(label).strip().lower()
    if lowered in {"ch", "chirality"}:
        return "CH", "ch", None

    canonical = normalize_group_label(label)
    if canonical in {"Cs", "Ci"}:
        _, hm_label = canonical_label_pair(canonical)
        return canonical, canonical.lower(), hm_label

    match = _CN_RE.fullmatch(canonical)
    if match is not None:
        _, hm_label = canonical_label_pair(canonical)
        return canonical, canonical.lower(), hm_label

    match = _SN_RE.fullmatch(canonical)
    if match is not None:
        order = int(match.group(1))
        if order % 2 != 0:
            raise ValueError(
                f"Official exact CSM only supports even-order Sn operations, got {label!r}."
            )
        return canonical, canonical.lower(), None

    raise ValueError(
        "Official exact CSM only supports Cs, Ci, Cn, Sn, and CH. "
        f"Unsupported label: {label!r}."
    )


def parse_official_exact_csm_candidate_groups(
    candidate_groups: str | Sequence[str],
    *,
    chirality_sn_max: int = 8,
) -> list[str]:
    labels = parse_label_list(candidate_groups)
    normalized: list[str] = []
    seen: set[str] = set()
    for label in labels:
        canonical, _, _ = _normalize_operation_label(
            label,
            chirality_sn_max=chirality_sn_max,
        )
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(label)
    return normalized


def supports_official_exact_csm_group(
    label: str,
    *,
    chirality_sn_max: int = 8,
) -> bool:
    """Return whether a label is supported by the official exact CSM backend."""

    try:
        _normalize_operation_label(
            label,
            chirality_sn_max=chirality_sn_max,
        )
    except (KeyError, ValueError):
        return False
    return True


def filter_supported_official_exact_csm_groups(
    candidate_groups: str | Sequence[str],
    *,
    chirality_sn_max: int = 8,
) -> list[str]:
    """Filter a candidate-group list to the subset supported by official exact CSM."""

    labels = parse_label_list(candidate_groups)
    filtered: list[str] = []
    seen: set[str] = set()
    for label in labels:
        try:
            canonical, _, _ = _normalize_operation_label(
                label,
                chirality_sn_max=chirality_sn_max,
            )
        except (KeyError, ValueError):
            continue
        key = canonical.lower()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(str(label))
    return filtered


def resolve_official_exact_csm_label(
    label: str,
    *,
    chirality_sn_max: int = 8,
) -> tuple[str, str | None, str]:
    """Return (Schoenflies, HM-or-None, operation-code) for a supported label."""

    label_schoenflies, operation_code, label_hm = _normalize_operation_label(
        label,
        chirality_sn_max=chirality_sn_max,
    )
    return label_schoenflies, label_hm, operation_code


def _select_prediction(
    candidate_results: Sequence[OfficialExactCSMCandidateEvaluation],
    *,
    hard_tol: float,
) -> tuple[OfficialExactCSMCandidateEvaluation, str, int]:
    if not candidate_results:
        raise ValueError("candidate_results must be non-empty.")

    satisfied = [
        result
        for result in candidate_results
        if result.csm_value <= hard_tol
    ]
    if satisfied:
        predicted = min(
            satisfied,
            key=lambda item: (-item.operation_order, item.csm_value, item.label_schoenflies),
        )
        return predicted, "max_order_below_tol", len(satisfied)

    predicted = min(
        candidate_results,
        key=lambda item: (item.csm_value, -item.operation_order, item.label_schoenflies),
    )
    return predicted, "min_csm_fallback", 0


def evaluate_official_exact_csm_candidate(
    get_parsed_args: Any,
    calc: Any,
    sample: PreparedPointCloud,
    candidate_label: str,
    *,
    keep_structure: bool,
    babel_bond: bool,
    use_mass: bool,
    ignore_symmetry_labels: bool,
    remove_hydrogen: bool,
    timeout_sec: int,
    chirality_sn_max: int,
    suppress_official_stdout: bool,
    work_root: str | Path | None,
) -> OfficialExactCSMCandidateEvaluation:
    label_schoenflies, op_code, label_hm = _normalize_operation_label(
        candidate_label,
        chirality_sn_max=chirality_sn_max,
    )
    symbols = _official_symbols(sample.species)

    with build_work_dir(work_root, prefix="official_exact_csm_") as run_dir:
        input_path = write_xyz(
            run_dir / "input.xyz",
            sample.positions,
            symbols=symbols,
            comment=sample.sample_id,
        )
        output_dir = run_dir / f"{op_code}_results"

        args = [
            "exact",
            op_code,
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
            "--overwrite",
            "--timeout",
            str(int(timeout_sec)),
        ]
        if keep_structure:
            args.append("--keep-structure")
        if babel_bond:
            args.append("--babel-bond")
        if use_mass:
            args.append("--use-mass")
        if ignore_symmetry_labels:
            args.append("--ignore-symm")
        if remove_hydrogen:
            args.append("--remove-hy")
        if op_code == "ch":
            args.extend(["--sn-max", str(int(chirality_sn_max))])

        dictionary_args = get_parsed_args(args)
        dictionary_args["argument_string"] = " ".join(args) + "\n"
        try:
            with _maybe_silence(suppress_official_stdout):
                results = calc(dictionary_args)
        except Exception as exc:
            raise ExternalToolExecutionError(
                "Official exact CSM evaluation failed for "
                f"{candidate_label!r} on sample {sample.sample_id!r}."
            ) from exc

        try:
            result = results[0][0]
        except Exception as exc:
            raise ExternalToolExecutionError(
                "Official exact CSM returned an unexpected result container."
            ) from exc

        try:
            csm_value = float(result.csm)
            formula_csm = float(result.formula_csm)
        except Exception as exc:
            raise ExternalToolExecutionError(
                "Official exact CSM result is missing the expected scalar fields."
            ) from exc

        result_path = output_dir / "csm.txt"
        structure_preservation = result.overall_statistics.get("% structure")
        if isinstance(structure_preservation, (int, float)):
            structure_preservation = float(structure_preservation)

        direction = tuple(float(value) for value in np.asarray(result.dir, dtype=float).reshape(-1).tolist())
        permutation = tuple(int(value) for value in result.perm)

        return OfficialExactCSMCandidateEvaluation(
            label_input=str(candidate_label),
            label_schoenflies=label_schoenflies,
            label_hm=label_hm,
            operation_code=op_code,
            operation_type=str(result.op_type),
            operation_order=int(result.op_order),
            csm_value=csm_value,
            formula_csm=formula_csm,
            permutation=permutation,
            direction=direction,
            num_invalid_cycles=int(result.overall_statistics.get("# bad cycles", 0)),
            structure_preservation=structure_preservation,
            command=tuple(str(value) for value in args),
            result_path=str(result_path) if result_path.is_file() else None,
        )


class OfficialExactCSM:
    """Convenient callable wrapper for the official exact CSM implementation."""

    def __init__(
        self,
        *,
        candidate_groups: str | Sequence[str],
        center_mode: str = "none",
        manual_center: str | Sequence[float] | None = None,
        cutoff: float | None = None,
        noise_alpha: float = 0.0,
        noise_seed: int = 42,
        keep_structure: bool = True,
        babel_bond: bool = True,
        use_mass: bool = False,
        ignore_symmetry_labels: bool = False,
        remove_hydrogen: bool = False,
        timeout_sec: int = 300,
        hard_tol: float = 1e-4,
        chirality_sn_max: int = 8,
        official_source_root: str | Path | None = None,
        work_root: str | Path | None = None,
        suppress_official_stdout: bool = True,
    ) -> None:
        center_mode = str(center_mode)
        if center_mode not in {"centroid", "first_atom", "manual", "none"}:
            raise ValueError(f"Unsupported center_mode: {center_mode}")
        if keep_structure and not babel_bond:
            raise ValueError(
                "Official exact CSM requires bond information when keep_structure=True; "
                "set babel_bond=True or disable keep_structure."
            )

        parsed_groups = tuple(
            parse_official_exact_csm_candidate_groups(
                candidate_groups,
                chirality_sn_max=chirality_sn_max,
            )
        )
        self.config = OfficialExactCSMConfig(
            candidate_groups=parsed_groups,
            center_mode=center_mode,
            manual_center=parse_manual_center(manual_center),
            cutoff=None if cutoff is None else float(cutoff),
            noise_alpha=float(noise_alpha),
            noise_seed=int(noise_seed),
            keep_structure=bool(keep_structure),
            babel_bond=bool(babel_bond),
            use_mass=bool(use_mass),
            ignore_symmetry_labels=bool(ignore_symmetry_labels),
            remove_hydrogen=bool(remove_hydrogen),
            timeout_sec=int(timeout_sec),
            hard_tol=float(hard_tol),
            chirality_sn_max=int(chirality_sn_max),
            official_source_root=(
                None if official_source_root is None else str(Path(official_source_root).resolve())
            ),
            work_root=None if work_root is None else str(Path(work_root).resolve()),
            suppress_official_stdout=bool(suppress_official_stdout),
        )

    @staticmethod
    def _coerce_sample(
        sample: LabeledPointCloud | dict[str, Any],
        *,
        index: int | None,
    ) -> LabeledPointCloud:
        if isinstance(sample, LabeledPointCloud):
            return sample
        if isinstance(sample, dict):
            return record_to_sample(sample, index=index)
        raise TypeError(
            "OfficialExactCSM expects each sample to be a LabeledPointCloud or a "
            "toy-style record dict."
        )

    def _prepare_sample(
        self,
        sample: LabeledPointCloud,
        *,
        split_tag: str,
    ) -> PreparedPointCloud:
        return prepare_external_point_cloud(
            sample,
            split_tag=split_tag,
            center_mode=self.config.center_mode,
            manual_center=self.config.manual_center,
            cutoff=self.config.cutoff,
            noise_alpha=self.config.noise_alpha,
            noise_seed=self.config.noise_seed,
            cutoff_inclusive=True,
        )

    def evaluate_sample(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> OfficialExactCSMSampleResult:
        raw_sample = self._coerce_sample(sample, index=index)
        prepared = self._prepare_sample(raw_sample, split_tag=split_tag)
        get_parsed_args, calc = _load_official_csm_runner(self.config.official_source_root)

        start_time = time.perf_counter()
        candidate_results = tuple(
            evaluate_official_exact_csm_candidate(
                get_parsed_args,
                calc,
                prepared,
                candidate_label,
                keep_structure=self.config.keep_structure,
                babel_bond=self.config.babel_bond,
                use_mass=self.config.use_mass,
                ignore_symmetry_labels=self.config.ignore_symmetry_labels,
                remove_hydrogen=self.config.remove_hydrogen,
                timeout_sec=self.config.timeout_sec,
                chirality_sn_max=self.config.chirality_sn_max,
                suppress_official_stdout=self.config.suppress_official_stdout,
                work_root=self.config.work_root,
            )
            for candidate_label in self.config.candidate_groups
        )
        predicted, prediction_mode, satisfied_group_count = _select_prediction(
            candidate_results,
            hard_tol=self.config.hard_tol,
        )

        true_label_input = None if raw_sample.true_group is None else str(raw_sample.true_group)
        true_result = None
        correct = None
        if true_label_input is not None:
            try:
                true_schoenflies, _, _ = _normalize_operation_label(
                    true_label_input,
                    chirality_sn_max=self.config.chirality_sn_max,
                )
            except Exception:
                true_schoenflies = None
            if true_schoenflies is not None:
                for candidate_result in candidate_results:
                    if candidate_result.label_schoenflies == true_schoenflies:
                        true_result = candidate_result
                        break
                correct = int(predicted.label_schoenflies == true_schoenflies)

        elapsed_sec = float(time.perf_counter() - start_time)
        return OfficialExactCSMSampleResult(
            sample_id=prepared.sample_id,
            prepared=prepared,
            candidate_results=candidate_results,
            predicted=predicted,
            prediction_mode=prediction_mode,
            satisfied_group_count=satisfied_group_count,
            true_label_input=true_label_input,
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
    ) -> OfficialExactCSMSampleResult:
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
    ) -> OfficialExactCSMRunResult:
        sample_list = list(samples)
        prediction_mode_counts: Counter[str] = Counter()
        sample_results: list[OfficialExactCSMSampleResult] = []

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
                    f"[official_exact_csm] processed {sample_index + 1}/{len(sample_list)} samples "
                    f"(last_id={result.sample_id}, pred={result.predicted.label_input}, "
                    f"mode={result.prediction_mode})",
                    flush=True,
                )

        elapsed_sec = float(time.perf_counter() - run_start)
        return OfficialExactCSMRunResult(
            sample_results=tuple(sample_results),
            prediction_mode_counts=dict(prediction_mode_counts),
            elapsed_sec=elapsed_sec,
        )


CSM = OfficialExactCSM
CSMCandidateEvaluation = OfficialExactCSMCandidateEvaluation
CSMConfig = OfficialExactCSMConfig
CSMRunResult = OfficialExactCSMRunResult
CSMSampleResult = OfficialExactCSMSampleResult
evaluate_csm_candidate = evaluate_official_exact_csm_candidate
filter_supported_csm_groups = filter_supported_official_exact_csm_groups
parse_csm_candidate_groups = parse_official_exact_csm_candidate_groups
resolve_csm_label = resolve_official_exact_csm_label
supports_csm_group = supports_official_exact_csm_group


__all__ = [
    "CSM",
    "CSMCandidateEvaluation",
    "CSMConfig",
    "CSMRunResult",
    "CSMSampleResult",
    "OfficialExactCSM",
    "OfficialExactCSMCandidateEvaluation",
    "OfficialExactCSMConfig",
    "OfficialExactCSMRunResult",
    "OfficialExactCSMSampleResult",
    "evaluate_csm_candidate",
    "evaluate_official_exact_csm_candidate",
    "filter_supported_csm_groups",
    "filter_supported_official_exact_csm_groups",
    "parse_csm_candidate_groups",
    "parse_official_exact_csm_candidate_groups",
    "resolve_csm_label",
    "resolve_official_exact_csm_label",
    "supports_csm_group",
    "supports_official_exact_csm_group",
]
