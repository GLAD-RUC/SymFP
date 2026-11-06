"""Two counted spectroscopic observables of Eu(III), derived from the rank template.

Both are standard lanthanide photophysics, and both turn out to be functions of a
single signature entry, which is what makes the signature legible against a measured
spectrum without any calibrated cut-off:

``5D0 -> 7F0`` appears only for a polar site. Written in terms of the signature, the
line is observed exactly when ``m_1(H) >= 1`` at natural (odd) parity, which selects
the ten polar crystallographic point groups.

``5D0 -> 7F1`` splits into as many Stark components as the rank-2 crystal field
resolves. Since ``J = 1`` carries the degree-one representation and only ``k = 2``
terms act within it, the count is fixed by ``m_2(H)``: no rank-2 term leaves the level
threefold degenerate, a single term (``B_2^0``) splits it in two, and two or more give
three distinct levels.

The predictions do not require assigning a point group. By the fixed-subspace theorem
the rank of a structure's degree-``l`` block equals ``m_l`` of its exact symmetry, so
the measured ranks feed the rules directly.
"""

from __future__ import annotations

from typing import Dict, Tuple

from models.symmetry_groups import POINT_GROUPS, invariant_multiplicity

# The ten polar crystallographic point groups, in Schoenflies notation.
POLAR_GROUPS: Tuple[str, ...] = (
    "C1", "Cs", "C2", "C2v", "C3", "C3v", "C4", "C4v", "C6", "C6v",
)
CUBIC_GROUPS: Tuple[str, ...] = ("T", "Th", "O", "Td", "Oh")
ORTHORHOMBIC_AND_BELOW: Tuple[str, ...] = (
    "C1", "Ci", "C2", "Cs", "C2h", "D2", "C2v", "D2h",
)


def f0_line_observed(rank_l1: int) -> bool:
    """Whether ``5D0 -> 7F0`` is observed, from the degree-one rank."""
    return int(rank_l1) >= 1


def f1_component_count(rank_l2: int) -> int:
    """Number of ``5D0 -> 7F1`` Stark components, from the degree-two rank."""
    rank_l2 = int(rank_l2)
    if rank_l2 == 0:
        return 1
    if rank_l2 == 1:
        return 2
    return 3


def predict_from_signature(rank_l1: int, rank_l2: int) -> Dict[str, object]:
    return {
        "f0_observed": f0_line_observed(rank_l1),
        "f1_components": f1_component_count(rank_l2),
    }


def textbook_expectation(schoenflies: str) -> Dict[str, object]:
    """The same two observables as the spectroscopy literature states them per group."""
    if schoenflies in CUBIC_GROUPS:
        components = 1
    elif schoenflies in ORTHORHOMBIC_AND_BELOW:
        components = 3
    else:
        components = 2
    return {
        "f0_observed": schoenflies in POLAR_GROUPS,
        "f1_components": components,
    }


def verify_rules() -> Dict[str, Dict[str, object]]:
    """Check both rules against the literature statement for all 32 point groups."""
    from data.cf_selection_rules import HM_TO_SCHOENFLIES

    report: Dict[str, Dict[str, object]] = {}
    for hm in POINT_GROUPS:
        schoenflies = HM_TO_SCHOENFLIES[hm]
        predicted = predict_from_signature(
            invariant_multiplicity(hm, ell=1, parity="o"),
            invariant_multiplicity(hm, ell=2, parity="e"),
        )
        expected = textbook_expectation(schoenflies)
        if predicted != expected:
            raise AssertionError(
                f"{schoenflies} ({hm}): signature gives {predicted}, "
                f"spectroscopy literature expects {expected}."
            )
        report[schoenflies] = {
            "point_group_hm": hm,
            "m_1_odd": invariant_multiplicity(hm, ell=1, parity="o"),
            "m_2_even": invariant_multiplicity(hm, ell=2, parity="e"),
            **predicted,
        }
    return report


__all__ = [
    "CUBIC_GROUPS",
    "ORTHORHOMBIC_AND_BELOW",
    "POLAR_GROUPS",
    "f0_line_observed",
    "f1_component_count",
    "predict_from_signature",
    "textbook_expectation",
    "verify_rules",
]
