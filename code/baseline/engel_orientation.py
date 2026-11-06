"""Shared-orientation retained-energy search for the Engel score.

All degrees use the same spatial rotation. The finite multistart budget estimates
the orientation maximum; it does not certify a global optimum.
"""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from e3nn.o3._wigner import so3_generators
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .csom_modules.groups import group_matrices, normalize_group_label


@lru_cache(maxsize=1)
def _reference_groups():
    # These matrices fix the benchmark frame independently of optional CSoM
    # installations. Three-/six-fold operations use double-precision sqrt(3).
    return json.loads(Path(__file__).with_name("engel_point_groups.json").read_text())


@lru_cache(maxsize=None)
def target_operations(label):
    label = normalize_group_label(label)
    reference = _reference_groups().get(label)
    matrices = group_matrices(label) if reference is None else reference
    return tuple(np.asarray(matrix, dtype=np.float64) for matrix in matrices)


@lru_cache(maxsize=None)
def generators(ell):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        return so3_generators(ell).numpy().copy()
    finally:
        torch.set_default_dtype(previous)


@lru_cache(maxsize=None)
def axis_eigh(ell, axis):
    return np.linalg.eigh(1j * generators(ell)[axis])


def axis_matrix(ell, axis, angle):
    values, vectors = axis_eigh(ell, axis)
    return ((vectors * np.exp(-1j * angle * values)) @ vectors.conj().T).real


def angles_from_matrix(matrix):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        return Rotation.from_matrix(matrix).as_euler("YXY")


def matrix_irrep(ell, angles, derivatives=False):
    a, b, g = angles
    ya = axis_matrix(ell, 1, a)
    xb = axis_matrix(ell, 0, b)
    yg = axis_matrix(ell, 1, g)
    result = ya @ xb @ yg
    if not derivatives:
        return result
    gx, gy, _ = generators(ell)
    return result, (gy @ result, ya @ gx @ xb @ yg, result @ gy)


@lru_cache(maxsize=None)
def projector(label, ell):
    matrices = target_operations(label)
    p = np.zeros((2 * ell + 1, 2 * ell + 1))
    for matrix in matrices:
        parity = 1 if np.linalg.det(matrix) > 0 else -1
        p += parity ** ell * matrix_irrep(ell, angles_from_matrix(parity * matrix))
    p /= len(matrices)
    values, vectors = np.linalg.eigh((p + p.T) / 2)
    if np.max(np.minimum(abs(values), abs(values - 1))) > 1e-9:
        raise ValueError(f"Inaccurate point-group projector for {label}, degree {ell}.")
    fixed = vectors[:, values > 0.5]
    return fixed @ fixed.T


class ProjectionObjective:
    """Normalized discarded energy and its Euler-angle gradient."""

    def __init__(self, blocks, label):
        self.label = label
        total = sum(float(np.sum(v * v)) for ell, v in enumerate(blocks) if ell > 0)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Engel orientation search requires non-zero finite harmonic energy.")
        self.covariances = {
            ell: np.outer(v, v) / total for ell, v in enumerate(blocks) if ell > 0
        }
        self.items = [
            (ell, c, projector(label, ell))
            for ell, c in self.covariances.items()
            if np.trace(c) > 1e-25 and 0 < round(np.trace(projector(label, ell))) < len(c)
        ]
        self.constant_loss = sum(
            float(np.trace(c)) for ell, c in self.covariances.items()
            if round(np.trace(projector(label, ell))) == 0
        )

    def value_gradient(self, angles):
        loss = self.constant_loss
        grad = np.zeros(3)
        for ell, c, p in self.items:
            d, derivatives = matrix_irrep(ell, angles, True)
            pdc = p @ d @ c
            loss += float(np.trace(c) - np.sum(pdc * d))
            for i, dd in enumerate(derivatives):
                grad[i] -= 2 * float(np.sum(pdc * dd))
        return loss, grad


def seed_bank(seed=20260920, count=128):
    random = Rotation.random(count, random_state=np.random.default_rng(seed)).as_euler("YXY")
    return np.concatenate([np.zeros((1, 3)), random])


def optimize_projection(objective, *, warm=None, seed=20260920, count=128,
                        local_starts=12, maxiter=200):
    """Retain the base search, then extend it to the enhanced search budget.

    The base uses identity + the first 32 Haar rotations and four local runs;
    the enhanced stage uses all sampled rotations and `local_starts` local runs.
    An optional coordinate-fitted start is refined against this same objective.
    """
    if count < 32 or local_starts < 1 or maxiter < 1:
        raise ValueError("Search requires count >= 32, local_starts >= 1 and maxiter >= 1.")
    if not objective.items:
        return np.eye(3), float(objective.constant_loss), ()
    starts = seed_bank(seed, count)
    coarse = np.array([objective.value_gradient(a)[0] for a in starts])
    best_index = int(np.argmin(coarse[:33]))
    best_angles = starts[best_index].copy()
    best_loss = float(coarse[best_index])
    diagnostics = []

    def improve(angles, stage):
        nonlocal best_angles, best_loss
        before = objective.value_gradient(angles)[0]
        if before < best_loss:
            best_loss, best_angles = float(before), angles.copy()
        result = minimize(objective.value_gradient, angles, jac=True, method="BFGS",
                          options={"gtol": 1e-10, "maxiter": maxiter})
        if np.isfinite(result.fun) and result.fun < best_loss:
            best_loss, best_angles = float(result.fun), result.x.copy()
        diagnostics.append(dict(stage=stage, success=bool(result.success),
                                nit=int(result.nit), nfev=int(result.nfev),
                                loss=float(result.fun), status=int(result.status)))

    if warm is not None:
        improve(angles_from_matrix(warm), "coordinate_start")
    for index in np.argsort(coarse[:33], kind="stable")[:4]:
        improve(starts[index], "base")
    index = int(np.argmin(coarse))
    if coarse[index] < best_loss:
        best_loss, best_angles = float(coarse[index]), starts[index].copy()
    for index in np.argsort(coarse, kind="stable")[:local_starts]:
        improve(starts[index], "enhanced")
    return Rotation.from_euler("YXY", best_angles).as_matrix(), best_loss, tuple(diagnostics)
