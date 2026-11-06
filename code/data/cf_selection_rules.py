"""Crystal-field selection rules read off the invariant-subspace dimensions.

The f-electron crystal-field Hamiltonian is a sum of Stevens operators over even
ranks k in {2, 4, 6}, so the number of independent Wybourne parameters allowed by a
site symmetry H is the total dimension of the H-invariant subspace of those ranks.
That is the same quantity the symmetry signature already reports, which is why the
counts below reproduce the crystal-field literature without any fitting.

Point groups are named in Hermann-Mauguin notation, matching
``models.symmetry_groups`` and ``SymFPWrapper``. Do not pass Schoenflies labels
here; ``data.schoenflies`` is the Schoenflies-named entry point.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

from models.symmetry_groups import POINT_GROUPS, invariant_multiplicity

CF_RANKS: Tuple[int, ...] = (2, 4, 6)

HM_TO_SCHOENFLIES: Dict[str, str] = {
    "1": "C1",
    "-1": "Ci",
    "2": "C2",
    "m": "Cs",
    "2/m": "C2h",
    "222": "D2",
    "mm2": "C2v",
    "mmm": "D2h",
    "4": "C4",
    "-4": "S4",
    "4/m": "C4h",
    "422": "D4",
    "4mm": "C4v",
    "-42m": "D2d",
    "4/mmm": "D4h",
    "3": "C3",
    "-3": "S6",
    "32": "D3",
    "3m": "C3v",
    "-3m": "D3d",
    "6": "C6",
    "-6": "C3h",
    "6/m": "C6h",
    "622": "D6",
    "6mm": "C6v",
    "-6m2": "D3h",
    "6/mmm": "D6h",
    "23": "T",
    "m-3": "Th",
    "432": "O",
    "-43m": "Td",
    "m-3m": "Oh",
}


@lru_cache(maxsize=None)
def rank_multiplicity(point_group: str, rank: int) -> int:
    """Number of independent crystal-field parameters of the given rank."""
    return invariant_multiplicity(point_group, ell=int(rank), parity="e")


@lru_cache(maxsize=None)
def cf_parameter_count(point_group: str) -> int:
    """Total number of independent crystal-field parameters allowed by the group."""
    return sum(rank_multiplicity(point_group, rank) for rank in CF_RANKS)


def rank_multiplicities(point_group: str) -> Tuple[int, ...]:
    """Per-rank counts ordered as ``CF_RANKS``."""
    return tuple(rank_multiplicity(point_group, rank) for rank in CF_RANKS)


def selection_rule_table() -> Dict[str, Dict[str, object]]:
    """Per-rank and total crystal-field parameter counts for all 32 point groups."""
    table: Dict[str, Dict[str, object]] = {}
    for hm in POINT_GROUPS:
        multiplicities = rank_multiplicities(hm)
        table[hm] = {
            "schoenflies": HM_TO_SCHOENFLIES[hm],
            "multiplicities": multiplicities,
            "total": sum(multiplicities),
        }
    return table


__all__ = [
    "CF_RANKS",
    "HM_TO_SCHOENFLIES",
    "cf_parameter_count",
    "rank_multiplicities",
    "rank_multiplicity",
    "selection_rule_table",
]
