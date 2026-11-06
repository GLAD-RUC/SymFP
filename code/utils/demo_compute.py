"""NumPy/SciPy implementation of the PF hybrid representation and symmetry scores for small CPU demos.

The radial weights, node/edge channels, centre mask and leakage pooling follow
ParaFreeModel and SymFPWrapper. Real spherical harmonics use a different
orthonormal basis, which leaves each block's singular values unchanged. This
implementation uses float64 throughout and does not import the training stack.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import sph_harm_y

MAX_ATOMS = 128
_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba "
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb "
    "Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs "
    "Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_ATOMIC_NUMBERS = {symbol: z for z, symbol in enumerate(_ELEMENTS, 1)}

# Character averages from models.symmetry_groups.invariant_multiplicity for
# natural-parity degrees 0..6. These are fixed-subspace dimensions, not cached
# structure scores. Keep the same 32-group order as SymFPWrapper.
_RANK_BOUNDS = {
    "1": (1, 3, 5, 7, 9, 11, 13),
    "-1": (1, 0, 5, 0, 9, 0, 13),
    "2": (1, 1, 3, 3, 5, 5, 7),
    "m": (1, 2, 3, 4, 5, 6, 7),
    "2/m": (1, 0, 3, 0, 5, 0, 7),
    "222": (1, 0, 2, 1, 3, 2, 4),
    "mm2": (1, 1, 2, 2, 3, 3, 4),
    "mmm": (1, 0, 2, 0, 3, 0, 4),
    "4": (1, 1, 1, 1, 3, 3, 3),
    "-4": (1, 0, 1, 2, 3, 2, 3),
    "4/m": (1, 0, 1, 0, 3, 0, 3),
    "422": (1, 0, 1, 0, 2, 1, 2),
    "4mm": (1, 1, 1, 1, 2, 2, 2),
    "-42m": (1, 0, 1, 1, 2, 1, 2),
    "4/mmm": (1, 0, 1, 0, 2, 0, 2),
    "3": (1, 1, 1, 3, 3, 3, 5),
    "-3": (1, 0, 1, 0, 3, 0, 5),
    "32": (1, 0, 1, 1, 2, 1, 3),
    "3m": (1, 1, 1, 2, 2, 2, 3),
    "-3m": (1, 0, 1, 0, 2, 0, 3),
    "6": (1, 1, 1, 1, 1, 1, 3),
    "-6": (1, 0, 1, 2, 1, 2, 3),
    "6/m": (1, 0, 1, 0, 1, 0, 3),
    "622": (1, 0, 1, 0, 1, 0, 2),
    "6mm": (1, 1, 1, 1, 1, 1, 2),
    "-6m2": (1, 0, 1, 1, 1, 1, 2),
    "6/mmm": (1, 0, 1, 0, 1, 0, 2),
    "23": (1, 0, 0, 1, 1, 0, 2),
    "m-3": (1, 0, 0, 0, 1, 0, 2),
    "432": (1, 0, 0, 0, 1, 0, 1),
    "-43m": (1, 0, 0, 1, 1, 0, 1),
    "m-3m": (1, 0, 0, 0, 1, 0, 1),
}


def validate_structure(symbols, positions):
    """Validate a finite, nonperiodic structure before building a complete graph."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.shape != (len(symbols), 3) or not 2 <= len(symbols) <= MAX_ATOMS:
        raise ValueError(f"Expected 2–{MAX_ATOMS} atoms with an N × 3 coordinate array.")
    if not np.isfinite(pos).all():
        raise ValueError("Coordinates must be finite.")
    try:
        atomic = np.array([_ATOMIC_NUMBERS[s] for s in symbols], dtype=np.int64)
    except (TypeError, KeyError) as exc:
        raise ValueError("Use standard element symbols, such as C, H, or Fe.") from exc
    pos = pos - pos.mean(axis=0)
    distances = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    if distances.min() < 1e-6:
        raise ValueError("Duplicate or nearly coincident atoms (< 1e-6 coordinate units).")
    return atomic, pos


def read_xyz(path):
    """Read exactly one standard XYZ frame; reject ambiguous trajectories."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ needs an atom count, a comment line, and atom rows.")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("The first XYZ line must be an integer atom count.") from exc
    rows = lines[2:]
    while rows and not rows[-1].strip():
        rows.pop()
    if len(rows) != count:
        raise ValueError("Atom count mismatch: provide exactly one XYZ frame.")
    symbols, positions = [], []
    for row in rows:
        fields = row.split()
        if len(fields) != 4:
            raise ValueError("Each XYZ atom row must contain: element x y z.")
        symbols.append(fields[0])
        positions.append([float(x) for x in fields[1:]])
    _, pos = validate_structure(symbols, positions)
    return symbols, pos


def deform_structure(positions, *, stretch=0.0, twist=0.0, noise=0.0, seed=0):
    """Apply z strain, a z-dependent twist (radians), then seeded Cartesian noise."""
    params = np.asarray([stretch, twist, noise], dtype=float)
    if not np.isfinite(params).all() or stretch <= -1 or noise < 0:
        raise ValueError("Require finite parameters, stretch > -1, and noise >= 0.")
    pos = np.asarray(positions, dtype=float).copy()
    pos -= pos.mean(axis=0)
    zscale = max(float(np.abs(pos[:, 2]).max()), 1e-12)
    angles = twist * pos[:, 2] / zscale
    x, y = pos[:, 0].copy(), pos[:, 1].copy()
    pos[:, 0] = np.cos(angles) * x - np.sin(angles) * y
    pos[:, 1] = np.sin(angles) * x + np.cos(angles) * y
    pos[:, 2] *= 1 + stretch
    pos += np.random.default_rng(seed).normal(scale=noise, size=pos.shape)
    return pos - pos.mean(axis=0)


def _real_spherical_harmonics(vectors, ell):
    """Real degree-ell harmonics with sum_m Y_lm**2 = 1 for nonzero vectors.

    SciPy uses (degree, order, polar angle, azimuth). Rescaling its integral
    normalization by sqrt(4*pi/(2*l+1)) matches e3nn's 'norm' normalization.
    """
    theta = np.arctan2(np.hypot(vectors[:, 0], vectors[:, 1]), vectors[:, 2])
    phi = np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * np.pi)
    harmonics = sph_harm_y(ell, np.arange(ell + 1)[None, :], theta[:, None], phi[:, None])
    real = np.concatenate((harmonics[:, :1].real,
                           np.sqrt(2) * harmonics[:, 1:].real,
                           np.sqrt(2) * harmonics[:, 1:].imag), axis=1)
    return real * np.sqrt(4 * np.pi / (2 * ell + 1))


class LiveSymFP:
    """Compute PF hybrid spectra and target compatibility using NumPy/SciPy."""

    def __init__(self, *, max_ell=6, channels=16, rank_tol=1e-4, ls_eps=1e-10):
        if (not isinstance(max_ell, (int, np.integer)) or not 1 <= max_ell <= 6
                or not isinstance(channels, (int, np.integer)) or not 1 <= channels <= 32):
            raise ValueError("This small CPU demo supports l_max=1..6 and channels=1..32.")
        if not np.isfinite([rank_tol, ls_eps]).all() or rank_tol <= 0 or ls_eps <= 0:
            raise ValueError("Rank tolerance and LS epsilon must be finite and positive.")
        start = perf_counter()
        self.max_ell, self.channels = int(max_ell), int(channels)
        self.rank_tol, self.ls_eps = float(rank_tol), float(ls_eps)
        self.radial_centers = np.array([0.5]) if channels == 1 else np.linspace(0, 1, channels)
        self.point_groups = list(_RANK_BOUNDS)
        self.signature_names = [f"{ell}{'e' if ell % 2 == 0 else 'o'}" for ell in range(max_ell + 1)]
        # One node attribute and three edge attributes give 4 * channels rows.
        self.observable_rank_matrix = np.minimum(
            np.array(list(_RANK_BOUNDS.values()), dtype=int)[:, :max_ell + 1], 4 * channels)
        self.settings = dict(max_ell=max_ell, channels=channels, rank_tol=rank_tol,
                             ls_eps=ls_eps, device="cpu", branch="hybrid",
                             implementation="NumPy / SciPy", dtype="float64",
                             graph="complete directed graph without self edges",
                             node_features="atomic number (scalar)")
        self.initialization_seconds = perf_counter() - start

    def _radial_basis(self, radii):
        normalized = radii / (radii.mean() + 1e-10)
        return np.exp(-0.5 * ((normalized[:, None] - self.radial_centers) * self.channels) ** 2)

    def _feature_blocks(self, atomic, centered):
        """PF node and directed-edge channels, concatenated within each degree."""
        radii = np.linalg.norm(centered, axis=1)
        node_weights = atomic[:, None] * self._radial_basis(radii)
        at_center = radii <= 1e-5 * (radii.mean() + 1e-10)

        row, col = np.where(~np.eye(len(centered), dtype=bool))
        vectors = centered[col] - centered[row]
        distances = np.linalg.norm(vectors, axis=1)
        edge_attributes = np.column_stack((atomic[row], atomic[col], distances))
        edge_weights = (edge_attributes[:, :, None]
                        * self._radial_basis(distances)[:, None, :]).reshape(len(row), -1)

        blocks = []
        for ell in range(self.max_ell + 1):
            node_harmonics = _real_spherical_harmonics(centered, ell)
            if ell > 0:
                node_harmonics[at_center] = 0
            edge_harmonics = _real_spherical_harmonics(vectors, ell)
            blocks.append(np.concatenate((node_weights.T @ node_harmonics,
                                          edge_weights.T @ edge_harmonics), axis=0))
        return blocks

    def compute(self, symbols, positions):
        atomic, centered = validate_structure(symbols, positions)
        start = perf_counter()
        spectra = [np.linalg.svd(block, compute_uv=False)
                   for block in self._feature_blocks(atomic, centered)]
        ranks = np.array([np.count_nonzero(values > self.rank_tol) for values in spectra])
        degree_leakage = []
        for ell, values in enumerate(spectra):
            energy = values ** 2
            # Entry m is the tail after the m allowed singular values.
            tails = np.append(np.cumsum(energy[::-1])[::-1], 0.0)
            degree_leakage.append(tails[self.observable_rank_matrix[:, ell]]
                                  / (energy.sum() + self.ls_eps))
        leakage = np.mean(degree_leakage, axis=0)
        scores = pd.DataFrame({
            "candidate (Hermann–Mauguin)": self.point_groups,
            "LS (lower = more compatible)": leakage,
            "pLS": -np.log10(leakage + self.ls_eps),
        })
        rank_table = pd.DataFrame(self.observable_rank_matrix.copy(),
                                  index=self.point_groups, columns=self.signature_names)
        rank_table.index.name = "candidate (Hermann–Mauguin)"
        return dict(spectra=spectra, ranks=ranks, scores=scores, bounds=rank_table,
                    channels=list(self.signature_names), seconds=perf_counter() - start)
