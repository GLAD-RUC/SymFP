"""Shared configuration and read-only data access for the interactive demos."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

PROJECT_ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "demo" / "data" / "manifest.json").is_file()
)
DEMO_DATA = PROJECT_ROOT / "demo" / "data"

PALETTE = {
    "teal": "#96CCCB",
    "pink": "#F6CAE5",
    "purple": "#C4A5DE",
    "blue": "#A1A9D0",
    "coral": "#F0988C",
    "deep_purple": "#B883D3",
    "grey": "#9E9E9E",
    "ink": "#1A1A1A",
}

RANK_ACTIVATION_CASES = {
    "B1_octahedron_tetragonal": {
        "label": "Oh → D4h: strain along a four-fold axis",
        "parent": "Oh",
        "child": "D4h",
        "degrees": {2: (0, 1), 4: (1, 2)},
    },
    "B2_octahedron_trigonal": {
        "label": "Oh → D3d: strain along a three-fold axis",
        "parent": "Oh",
        "child": "D3d",
        "degrees": {2: (0, 1), 4: (1, 2)},
    },
    "B3_prism_axial_chirality": {
        "label": "D3h → D3: twist about the three-fold axis",
        "parent": "D3h",
        "child": "D3",
        "degrees": {4: (1, 2), 6: (2, 3)},
    },
}

ANGULAR_MODES = {
    "AX2_spiral": {"label": "axial spiral", "child": "C1", "angular_order": 1},
    "IP1_ellipse": {"label": "in-plane ellipse", "child": "C2", "angular_order": 2},
    "IP2_radial": {"label": "in-plane radial", "child": "C3", "angular_order": 3},
    "AX1_pucker": {"label": "axial pucker", "child": "C3", "angular_order": 3},
}

SUBGROUP_TARGETS = ("Oh", "D4h", "D2h", "C2h", "Ci")
SUBGROUP_BOUNDARIES = (1, 16, 31, 46)
SUBGROUP_ENTERED = ("D4h", "D2h", "C2h", "Ci")
SUBGROUP_EXPECTED_ONSETS = {"Oh": 1, "D4h": 16, "D2h": 31, "C2h": 46, "Ci": None}
SUBGROUP_COLUMNS = {
    target: f"leak_{index}" for index, target in enumerate(SUBGROUP_TARGETS)
}

DEGREE_STYLES = {
    1: (PALETTE["coral"], "-"),
    2: (PALETTE["teal"], "-"),
    3: (PALETTE["purple"], "-"),
    4: (PALETTE["blue"], "--"),
    5: (PALETTE["deep_purple"], ":"),
    6: (PALETTE["pink"], "-."),
}


def environment_versions() -> pd.DataFrame:
    """Versions relevant to reproducing the notebooks."""
    return pd.DataFrame(
        [
            ("Python", platform.python_version()),
            ("NumPy", np.__version__),
            ("pandas", pd.__version__),
            ("Matplotlib", matplotlib.__version__),
            ("SciPy", scipy.__version__),
        ],
        columns=["Component", "Version"],
    )


def runtime_summary() -> dict[str, str]:
    """Short runtime summary for display at notebook startup."""
    return {
        "project_root": str(PROJECT_ROOT),
        "device_available": "CPU (NumPy / SciPy)",
    }


def load_rank_activation(case: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Structures and cached singular values for one single-step case."""
    if case not in RANK_ACTIVATION_CASES:
        raise KeyError(
            f"Unknown case {case!r}; choose from {tuple(RANK_ACTIVATION_CASES)}"
        )
    geometry = np.load(
        DEMO_DATA / "dataset/symmetry_breaking" / case / "positions.npz",
        allow_pickle=False,
    )
    positions = geometry["positions"]
    frame_groups = geometry["frame_group_hm"]
    spectra = pd.read_csv(
        DEMO_DATA / "results/symmetry_breaking/records" / case / "symfp_singvals.csv"
    )
    if len(positions) != spectra["frame"].nunique():
        raise ValueError("Structure and singular-value frame counts do not match.")
    return positions, frame_groups, spectra


def load_subgroup_chain() -> pd.DataFrame:
    """Cached target-conditioned leakage along the published octahedral chain."""
    table = pd.read_csv(
        DEMO_DATA
        / "results/symmetry_breaking/records/C1_octahedron_descent/symfp_group_leakage.csv"
    )
    expected = {"frame", *SUBGROUP_COLUMNS.values()}
    missing = expected.difference(table.columns)
    if missing:
        raise ValueError(f"Subgroup-chain table is missing columns: {sorted(missing)}")
    return table


def load_angular_mode(mode: str) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    """Cached scores and geometry for one matched-amplitude displacement field."""
    if mode not in ANGULAR_MODES:
        raise KeyError(f"Unknown mode {mode!r}; choose from {tuple(ANGULAR_MODES)}")
    frames = pd.read_csv(
        DEMO_DATA / "results/mode_separation/mode_separation_frames.csv"
    )
    mode_frames = (
        frames[frames["case_id"] == mode].sort_values("frame").reset_index(drop=True)
    )
    with open(DEMO_DATA / "results/mode_separation/verdict.json") as handle:
        verdict = json.load(handle)
    geometry = np.load(
        DEMO_DATA / "dataset/mode_separation" / mode / "positions.npz",
        allow_pickle=False,
    )
    positions = geometry["positions"]
    subunit_index = geometry["subunit_index"]
    if len(positions) != len(mode_frames):
        raise ValueError("Geometry and cached score frame counts do not match.")
    return mode_frames, verdict, positions, subunit_index
