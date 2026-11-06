"""Thin wrappers around official SHAPE/cosymlib CShM backends."""

from __future__ import annotations

import ctypes
import importlib
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import (
    ExternalToolExecutionError,
    ExternalToolUnavailableError,
    build_work_dir,
    command_as_text,
    default_symbols,
    extract_table_value_after_header,
    find_newest_json,
    find_numeric_value,
    parse_label_list,
    parse_manual_center,
    prepare_external_point_cloud,
    record_to_sample,
    resolve_executable,
    write_xyz,
)
from .csom_modules.core import LabeledPointCloud, PreparedPointCloud
from .csom_modules.groups import normalize_group_label

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_VENDOR_DIR = _PROJECT_ROOT / ".vendor"
_LOCAL_VENDOR_BIN_DIR = _LOCAL_VENDOR_DIR / "bin"
_LOCAL_VENDOR_LIB_DIRS = (
    _LOCAL_VENDOR_DIR / "cosymlib.libs",
    _LOCAL_VENDOR_DIR / "symgroupy.libs",
)
_PRELOADED_VENDOR_LIBS = False


def _ensure_local_vendor_on_path() -> None:
    if not _LOCAL_VENDOR_DIR.is_dir():
        return
    vendor_text = str(_LOCAL_VENDOR_DIR)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)


def _prepend_env_path(env: dict[str, str], key: str, values: Sequence[Path]) -> None:
    parts = [str(value.resolve()) for value in values if value.is_dir()]
    if not parts:
        return
    existing = [chunk for chunk in env.get(key, "").split(os.pathsep) if chunk]
    merged: list[str] = []
    seen: set[str] = set()
    for part in [*parts, *existing]:
        if part in seen:
            continue
        seen.add(part)
        merged.append(part)
    env[key] = os.pathsep.join(merged)


def _vendor_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    if not _LOCAL_VENDOR_DIR.is_dir():
        return env
    _prepend_env_path(env, "PYTHONPATH", (_LOCAL_VENDOR_DIR,))
    _prepend_env_path(env, "PATH", (_LOCAL_VENDOR_BIN_DIR,))
    _prepend_env_path(env, "LD_LIBRARY_PATH", _LOCAL_VENDOR_LIB_DIRS)
    return env


def _preload_local_vendor_shared_objects() -> None:
    global _PRELOADED_VENDOR_LIBS
    if _PRELOADED_VENDOR_LIBS:
        return

    ordered_prefixes = ("libquadmath", "libgfortran")
    loaded_paths: set[Path] = set()
    for lib_dir in _LOCAL_VENDOR_LIB_DIRS:
        if not lib_dir.is_dir():
            continue
        for prefix in ordered_prefixes:
            for shared_lib in sorted(lib_dir.glob(f"{prefix}*.so*")):
                if shared_lib in loaded_paths:
                    continue
                ctypes.CDLL(str(shared_lib), mode=ctypes.RTLD_GLOBAL)
                loaded_paths.add(shared_lib)
        for shared_lib in sorted(lib_dir.glob("*.so*")):
            if shared_lib in loaded_paths:
                continue
            ctypes.CDLL(str(shared_lib), mode=ctypes.RTLD_GLOBAL)
            loaded_paths.add(shared_lib)

    _PRELOADED_VENDOR_LIBS = True


@dataclass(frozen=True)
class OfficialShapeReference:
    """One official SHAPE reference polyhedron."""

    label: str
    code: int
    symmetry_label: str
    description: str
    coordination_number: int


@dataclass(frozen=True)
class OfficialCShMConfig:
    """Runtime configuration for the official CShM wrapper."""

    candidate_shapes: tuple[str, ...]
    center_mode: str
    manual_center: tuple[float, float, float] | None
    cutoff: float | None
    noise_alpha: float
    noise_seed: int
    backend: str
    shape_executable: str | None
    extra_cli_args: tuple[str, ...]
    work_root: str | None
    central_atom_index: int | None
    fix_permutation: bool
    hard_tol: float | None
    symbols_mode: str
    uniform_symbol: str
    true_label_key: str | None


@dataclass(frozen=True)
class OfficialCShMCandidateEvaluation:
    """Target-conditioned official CShM result for one candidate shape label."""

    label_input: str
    shape_measure: float
    backend: str
    command: tuple[str, ...] | None
    result_path: str | None


@dataclass(frozen=True)
class OfficialCShMSampleResult:
    """Prediction bundle for one point cloud."""

    sample_id: str
    prepared: PreparedPointCloud
    candidate_results: tuple[OfficialCShMCandidateEvaluation, ...]
    predicted: OfficialCShMCandidateEvaluation
    prediction_mode: str
    satisfied_shape_count: int
    true_label_input: str | None
    true_result: OfficialCShMCandidateEvaluation | None
    correct: int | None
    elapsed_sec: float
    backend: str


@dataclass(frozen=True)
class OfficialCShMRunResult:
    """Dataset-level official CShM evaluation output."""

    sample_results: tuple[OfficialCShMSampleResult, ...]
    prediction_mode_counts: dict[str, int]
    elapsed_sec: float

    @property
    def num_samples(self) -> int:
        return len(self.sample_results)

    @property
    def labeled_results(self) -> tuple[OfficialCShMSampleResult, ...]:
        return tuple(result for result in self.sample_results if result.correct is not None)

    @property
    def accuracy(self) -> float | None:
        labeled = self.labeled_results
        if not labeled:
            return None
        return float(sum(int(result.correct) for result in labeled) / len(labeled))


def _shape_install_hint() -> str:
    return (
        "Official SHAPE/cosymlib tools were not found. Install the official "
        "`shape` executable or the `cosymlib` Python package, or keep the "
        "vendored build under `.vendor/`, or pass `shape_executable=` explicitly."
    )


def _resolve_shape_executable(override: str | None) -> str:
    if override is None:
        local_shape = _LOCAL_VENDOR_BIN_DIR / "shape"
        if local_shape.is_file():
            return str(local_shape)
    return resolve_executable(
        override=override,
        candidates=("shape",),
        install_hint=_shape_install_hint(),
    )


def _has_shape_cli(override: str | None) -> bool:
    try:
        _resolve_shape_executable(override)
    except ExternalToolUnavailableError:
        return False
    return True


def _load_cosymlib_geometry_class() -> Any:
    _ensure_local_vendor_on_path()
    try:
        _preload_local_vendor_shared_objects()
        module = importlib.import_module("cosymlib.molecule.geometry")
    except Exception as exc:
        raise ExternalToolUnavailableError(_shape_install_hint()) from exc
    if not hasattr(module, "Geometry"):
        raise ExternalToolUnavailableError(
            "cosymlib.molecule.geometry.Geometry was not found in the installed cosymlib package."
        )
    return module.Geometry


def _load_cosymlib_shape_tools_module() -> Any:
    _ensure_local_vendor_on_path()
    try:
        _preload_local_vendor_shared_objects()
        module = importlib.import_module("cosymlib.shape.tools")
    except Exception as exc:
        raise ExternalToolUnavailableError(_shape_install_hint()) from exc
    if not hasattr(module, "shape_structure_references"):
        raise ExternalToolUnavailableError(
            "cosymlib.shape.tools.shape_structure_references was not found in the installed cosymlib package."
        )
    return module


def _canonical_shape_symmetry(label: str) -> str:
    text = str(label).strip()
    if not text:
        raise ValueError("Shape symmetry label must be non-empty.")
    try:
        return normalize_group_label(text)
    except Exception:
        return text


@lru_cache(maxsize=None)
def official_shape_references(coordination_number: int) -> tuple[OfficialShapeReference, ...]:
    """Return official SHAPE reference polyhedra for a coordination number."""

    resolved_number = int(coordination_number)
    if resolved_number <= 0:
        raise ValueError(f"coordination_number must be positive, got {coordination_number!r}.")

    tools_module = _load_cosymlib_shape_tools_module()
    lookup_key = f"{resolved_number} Vertices"
    raw_entries = getattr(tools_module, "shape_structure_references", {}).get(lookup_key)
    if raw_entries is None:
        raise ValueError(
            "Official SHAPE references are unavailable for "
            f"coordination_number={resolved_number}."
        )

    return tuple(
        OfficialShapeReference(
            label=str(entry[0]),
            code=int(entry[1]),
            symmetry_label=str(entry[2]),
            description=str(entry[3]),
            coordination_number=resolved_number,
        )
        for entry in raw_entries
    )


def supported_official_shape_labels(coordination_number: int) -> tuple[str, ...]:
    """Return official SHAPE labels for a coordination number."""

    return tuple(reference.label for reference in official_shape_references(coordination_number))


def resolve_official_shape_candidate_shapes(
    *,
    coordination_number: int,
    candidate_groups: str | Sequence[str] | None = None,
    candidate_shapes: str | Sequence[str] | None = None,
) -> list[str]:
    """Resolve official SHAPE labels explicitly or by matching ideal point-group symmetry."""

    if candidate_groups is not None and candidate_shapes is not None:
        raise ValueError("Specify at most one of candidate_groups and candidate_shapes.")

    references = official_shape_references(coordination_number)
    by_label = {reference.label.lower(): reference for reference in references}

    if candidate_shapes is not None:
        if isinstance(candidate_shapes, str) and candidate_shapes.strip().lower() in {
            "all",
            "official_all",
        }:
            return [reference.label for reference in references]

        labels = parse_label_list(candidate_shapes)
        resolved: list[str] = []
        seen: set[str] = set()
        unsupported: list[str] = []
        for label in labels:
            key = str(label).strip().lower()
            reference = by_label.get(key)
            if reference is None:
                unsupported.append(str(label))
                continue
            if key in seen:
                continue
            seen.add(key)
            resolved.append(reference.label)
        if unsupported:
            raise ValueError(
                "Official SHAPE does not define the following labels for "
                f"coordination_number={int(coordination_number)}: {', '.join(unsupported)}"
            )
        return resolved

    if candidate_groups is None:
        return [reference.label for reference in references]

    target_groups = {
        _canonical_shape_symmetry(label)
        for label in parse_label_list(candidate_groups)
    }
    resolved = [
        reference.label
        for reference in references
        if _canonical_shape_symmetry(reference.symmetry_label) in target_groups
    ]
    if not resolved:
        raise ValueError(
            "No official SHAPE references matched the requested candidate_groups="
            f"{tuple(parse_label_list(candidate_groups))!r} for coordination_number={int(coordination_number)}."
        )
    return resolved


def _build_geometry_object(
    geometry_cls: Any,
    *,
    name: str,
    positions: np.ndarray,
    symbols: Sequence[str],
) -> Any:
    attempts = [
        lambda: geometry_cls(positions=positions, symbols=tuple(symbols), name=name),
        lambda: geometry_cls(np.asarray(positions, dtype=float), symbols=tuple(symbols), name=name),
        lambda: geometry_cls(symbols=tuple(symbols), positions=positions, name=name),
        lambda: geometry_cls(np.asarray(positions, dtype=float), tuple(symbols), name),
    ]
    last_error: Exception | None = None
    for build in attempts:
        try:
            return build()
        except Exception as exc:
            last_error = exc
    raise ExternalToolExecutionError(
        "Failed to construct a cosymlib Geometry object from the provided coordinates."
    ) from last_error


def _coerce_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    array = np.asarray(value, dtype=float)
    if array.size == 1:
        return float(array.reshape(-1)[0])
    raise ExternalToolExecutionError(
        f"Expected a scalar CShM value, got an array with shape {array.shape}."
    )


def _central_atom_candidates(central_atom_index: int | None) -> tuple[int, ...]:
    if central_atom_index is None:
        return (0,)

    index = int(central_atom_index)
    candidates = []
    if index >= 0:
        candidates.append(index + 1)
    candidates.append(index)
    deduplicated: list[int] = []
    seen: set[int] = set()
    for value in candidates:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return tuple(deduplicated)


def _shape_symbols(
    sample: PreparedPointCloud,
    *,
    symbols_mode: str,
    uniform_symbol: str,
) -> tuple[str, ...]:
    if symbols_mode == "uniform":
        return default_symbols(sample.positions.shape[0], symbol=uniform_symbol)
    if symbols_mode == "species":
        return tuple(str(symbol) for symbol in sample.species)
    raise ValueError(f"Unsupported symbols_mode: {symbols_mode}")


def _extract_true_label(
    sample: LabeledPointCloud | dict[str, Any],
    *,
    true_label_key: str | None,
) -> str | None:
    if true_label_key is None:
        return None
    if isinstance(sample, dict):
        value = sample.get(true_label_key)
        return None if value is None else str(value)
    if isinstance(sample, LabeledPointCloud) and true_label_key == "true_group":
        return sample.true_group
    return None


def _parse_shape_cli_result(
    *,
    work_dir: Path,
    candidate_label: str,
    stdout: str,
    stderr: str,
) -> tuple[float | None, Path | None]:
    json_path, payload = find_newest_json(work_dir)
    if payload is not None:
        value = find_numeric_value(
            payload,
            preferred_keys=("measure", "shape_measure", "cshm", "value"),
        )
        if value is not None:
            return float(value), json_path

    merged = "\n".join(chunk for chunk in (stdout, stderr) if chunk)
    value = extract_table_value_after_header(merged, header_token=candidate_label)
    if value is not None:
        return float(value), None
    return None, None


def evaluate_shape_cli_candidate(
    sample: PreparedPointCloud,
    candidate_label: str,
    *,
    shape_executable: str | None,
    work_root: str | Path | None,
    extra_cli_args: Sequence[str],
    symbols_mode: str,
    uniform_symbol: str,
    central_atom_index: int | None,
    fix_permutation: bool,
) -> OfficialCShMCandidateEvaluation:
    """Evaluate one candidate shape label with the official `shape` CLI."""

    executable = _resolve_shape_executable(shape_executable)

    with build_work_dir(work_root, prefix="official_cshm_") as work_dir:
        xyz_path = work_dir / f"{sample.sample_id}.xyz"
        write_xyz(
            xyz_path,
            sample.positions,
            symbols=_shape_symbols(
                sample,
                symbols_mode=symbols_mode,
                uniform_symbol=uniform_symbol,
            ),
            comment=sample.sample_id,
        )
        cmd = [executable, str(xyz_path), "-m", str(candidate_label)]
        if central_atom_index is not None:
            cmd.extend(["-c", str(int(central_atom_index) + 1)])
        if fix_permutation:
            cmd.append("--fix_permutation")
        cmd.extend(str(arg) for arg in extra_cli_args)
        process = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            check=False,
            env=_vendor_subprocess_env(),
        )
        stdout = process.stdout or ""
        stderr = process.stderr or ""
        value, result_path = _parse_shape_cli_result(
            work_dir=work_dir,
            candidate_label=str(candidate_label),
            stdout=stdout,
            stderr=stderr,
        )

        if process.returncode != 0 or value is None:
            raise ExternalToolExecutionError(
                "Official shape CLI execution failed. "
                f"cmd={command_as_text(cmd)!r}, returncode={process.returncode}, "
                f"stdout={stdout.strip()!r}, stderr={stderr.strip()!r}"
            )

        return OfficialCShMCandidateEvaluation(
            label_input=str(candidate_label),
            shape_measure=float(value),
            backend="shape_cli",
            command=tuple(str(part) for part in cmd),
            result_path=None if result_path is None else str(result_path),
        )


def evaluate_cosymlib_candidate(
    sample: PreparedPointCloud,
    candidate_label: str,
    *,
    symbols_mode: str,
    uniform_symbol: str,
    central_atom_index: int | None,
    fix_permutation: bool,
) -> OfficialCShMCandidateEvaluation:
    """Evaluate one candidate shape label with the official cosymlib Python API."""

    geometry_cls = _load_cosymlib_geometry_class()
    geometry = _build_geometry_object(
        geometry_cls,
        name=sample.sample_id,
        positions=np.asarray(sample.positions, dtype=float),
        symbols=_shape_symbols(
            sample,
            symbols_mode=symbols_mode,
            uniform_symbol=uniform_symbol,
        ),
    )

    last_error: Exception | None = None
    value: Any | None = None
    for central_atom in _central_atom_candidates(central_atom_index):
        attempts = [
            lambda: geometry.get_shape_measure(
                str(candidate_label),
                central_atom=central_atom,
                fix_permutation=bool(fix_permutation),
            ),
            lambda: geometry.get_shape_measure(
                str(candidate_label),
                central_atom,
                bool(fix_permutation),
            ),
            lambda: geometry.get_shape_measure(
                str(candidate_label),
                central_atom=central_atom,
            ),
            lambda: geometry.get_shape_measure(str(candidate_label)),
        ]
        for attempt in attempts:
            try:
                value = attempt()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is None:
            break

    if last_error is not None:
        raise ExternalToolExecutionError(
            "cosymlib get_shape_measure failed for "
            f"sample={sample.sample_id!r}, candidate={candidate_label!r}."
        ) from last_error

    return OfficialCShMCandidateEvaluation(
        label_input=str(candidate_label),
        shape_measure=_coerce_scalar(value),
        backend="cosymlib",
        command=None,
        result_path=None,
    )


class OfficialCShM:
    """Convenient callable wrapper for official SHAPE/cosymlib evaluation."""

    def __init__(
        self,
        *,
        candidate_shapes: str | Sequence[str],
        center_mode: str = "none",
        manual_center: str | Sequence[float] | None = None,
        cutoff: float | None = None,
        noise_alpha: float = 0.0,
        noise_seed: int = 42,
        backend: str = "auto",
        shape_executable: str | None = None,
        extra_cli_args: Sequence[str] = (),
        work_root: str | Path | None = None,
        central_atom_index: int | None = None,
        fix_permutation: bool = False,
        hard_tol: float | None = None,
        symbols_mode: str = "uniform",
        uniform_symbol: str = "H",
        true_label_key: str | None = None,
    ) -> None:
        center_mode = str(center_mode)
        if center_mode not in {"centroid", "first_atom", "manual", "none"}:
            raise ValueError(f"Unsupported center_mode: {center_mode}")

        backend = str(backend)
        if backend not in {"auto", "shape_cli", "cosymlib"}:
            raise ValueError(f"Unsupported backend: {backend}")

        symbols_mode = str(symbols_mode)
        if symbols_mode not in {"uniform", "species"}:
            raise ValueError(f"Unsupported symbols_mode: {symbols_mode}")

        parsed_shapes = tuple(parse_label_list(candidate_shapes))
        parsed_center = parse_manual_center(manual_center)

        self.config = OfficialCShMConfig(
            candidate_shapes=parsed_shapes,
            center_mode=center_mode,
            manual_center=parsed_center,
            cutoff=None if cutoff is None else float(cutoff),
            noise_alpha=float(noise_alpha),
            noise_seed=int(noise_seed),
            backend=backend,
            shape_executable=shape_executable,
            extra_cli_args=tuple(str(arg) for arg in extra_cli_args),
            work_root=None if work_root is None else str(Path(work_root).resolve()),
            central_atom_index=None if central_atom_index is None else int(central_atom_index),
            fix_permutation=bool(fix_permutation),
            hard_tol=None if hard_tol is None else float(hard_tol),
            symbols_mode=symbols_mode,
            uniform_symbol=str(uniform_symbol),
            true_label_key=None if true_label_key is None else str(true_label_key),
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
            "OfficialCShM expects each sample to be a LabeledPointCloud or a toy-style record dict."
        )

    def _resolve_backend(self) -> str:
        if self.config.backend != "auto":
            return self.config.backend

        try:
            _load_cosymlib_geometry_class()
            return "cosymlib"
        except ExternalToolUnavailableError:
            if _has_shape_cli(self.config.shape_executable):
                return "shape_cli"
            raise

    def prepare_sample(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> PreparedPointCloud:
        labeled = self._coerce_sample(sample, index=index)
        prepared = prepare_external_point_cloud(
            labeled,
            split_tag=split_tag,
            center_mode=self.config.center_mode,
            manual_center=self.config.manual_center,
            cutoff=self.config.cutoff,
            noise_alpha=self.config.noise_alpha,
            noise_seed=self.config.noise_seed,
            cutoff_inclusive=False,
        )
        if (
            self.config.central_atom_index is not None
            and not (0 <= self.config.central_atom_index < prepared.positions.shape[0])
        ):
            raise ValueError(
                "central_atom_index is out of bounds after preprocessing. "
                f"Got index={self.config.central_atom_index}, num_points={prepared.positions.shape[0]}."
            )
        return prepared

    def evaluate_sample(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> OfficialCShMSampleResult:
        sample_start = time.perf_counter()
        prepared = self.prepare_sample(sample, split_tag=split_tag, index=index)
        backend = self._resolve_backend()

        if backend == "shape_cli":
            evaluator = lambda label: evaluate_shape_cli_candidate(
                prepared,
                label,
                shape_executable=self.config.shape_executable,
                work_root=self.config.work_root,
                extra_cli_args=self.config.extra_cli_args,
                symbols_mode=self.config.symbols_mode,
                uniform_symbol=self.config.uniform_symbol,
                central_atom_index=self.config.central_atom_index,
                fix_permutation=self.config.fix_permutation,
            )
        elif backend == "cosymlib":
            evaluator = lambda label: evaluate_cosymlib_candidate(
                prepared,
                label,
                symbols_mode=self.config.symbols_mode,
                uniform_symbol=self.config.uniform_symbol,
                central_atom_index=self.config.central_atom_index,
                fix_permutation=self.config.fix_permutation,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        candidate_results = tuple(
            evaluator(candidate_label)
            for candidate_label in self.config.candidate_shapes
        )
        predicted = min(
            candidate_results,
            key=lambda item: (item.shape_measure, item.label_input),
        )

        true_label_input = _extract_true_label(sample, true_label_key=self.config.true_label_key)
        true_result = None
        if true_label_input is not None:
            true_result = next(
                (
                    result
                    for result in candidate_results
                    if result.label_input == true_label_input
                ),
                None,
            )

        correct = None
        if true_label_input is not None:
            correct = int(predicted.label_input == true_label_input)

        satisfied_shape_count = 0
        if self.config.hard_tol is not None:
            satisfied_shape_count = int(
                sum(
                    result.shape_measure <= self.config.hard_tol
                    for result in candidate_results
                )
            )

        elapsed_sec = float(time.perf_counter() - sample_start)
        return OfficialCShMSampleResult(
            sample_id=prepared.sample_id,
            prepared=prepared,
            candidate_results=candidate_results,
            predicted=predicted,
            prediction_mode="min_measure",
            satisfied_shape_count=satisfied_shape_count,
            true_label_input=true_label_input,
            true_result=true_result,
            correct=correct,
            elapsed_sec=elapsed_sec,
            backend=backend,
        )

    def __call__(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> OfficialCShMSampleResult:
        return self.evaluate_sample(sample, split_tag=split_tag, index=index)

    def predict_label(
        self,
        sample: LabeledPointCloud | dict[str, Any],
        *,
        split_tag: str = "eval",
        index: int | None = None,
    ) -> str:
        result = self.evaluate_sample(sample, split_tag=split_tag, index=index)
        return result.predicted.label_input

    def evaluate_samples(
        self,
        samples: Sequence[LabeledPointCloud | dict[str, Any]],
        *,
        split_tag: str = "eval",
        report_every: int = 0,
    ) -> OfficialCShMRunResult:
        sample_list = list(samples)
        prediction_mode_counts: Counter[str] = Counter()
        sample_results: list[OfficialCShMSampleResult] = []

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
                    f"[official_cshm] processed {sample_index + 1}/{len(sample_list)} samples "
                    f"(last_id={result.sample_id}, pred={result.predicted.label_input}, "
                    f"backend={result.backend})",
                    flush=True,
                )

        elapsed_sec = float(time.perf_counter() - run_start)
        return OfficialCShMRunResult(
            sample_results=tuple(sample_results),
            prediction_mode_counts=dict(prediction_mode_counts),
            elapsed_sec=elapsed_sec,
        )


CShM = OfficialCShM
CShMCandidateEvaluation = OfficialCShMCandidateEvaluation
CShMConfig = OfficialCShMConfig
CShMRunResult = OfficialCShMRunResult
CShMSampleResult = OfficialCShMSampleResult
shape_references = official_shape_references
resolve_shape_candidate_shapes = resolve_official_shape_candidate_shapes
supported_shape_labels = supported_official_shape_labels


__all__ = [
    "CShM",
    "CShMCandidateEvaluation",
    "CShMConfig",
    "CShMRunResult",
    "CShMSampleResult",
    "OfficialCShM",
    "OfficialCShMCandidateEvaluation",
    "OfficialCShMConfig",
    "OfficialCShMRunResult",
    "OfficialCShMSampleResult",
    "OfficialShapeReference",
    "evaluate_cosymlib_candidate",
    "evaluate_shape_cli_candidate",
    "official_shape_references",
    "resolve_official_shape_candidate_shapes",
    "resolve_shape_candidate_shapes",
    "shape_references",
    "supported_official_shape_labels",
    "supported_shape_labels",
]
