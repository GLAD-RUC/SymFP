from typing import Callable, Dict, Optional, Type

import torch

from .blocks import (
    AgnosticNonlinearInteractionBlock,
    AgnosticResidualNonlinearInteractionBlock,
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    ResidualElementDependentInteractionBlock,
    ScaleShiftBlock,
)
from .radial import BesselBasis, PolynomialCutoff
from .symmetric_contraction import SymmetricContraction


interaction_classes: Dict[str, Type[InteractionBlock]] = {
    "AgnosticNonlinearInteractionBlock": AgnosticNonlinearInteractionBlock,
    "ResidualElementDependentInteractionBlock": ResidualElementDependentInteractionBlock,
    "AgnosticResidualNonlinearInteractionBlock": AgnosticResidualNonlinearInteractionBlock,
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
}

gate_dict: Dict[str, Optional[Callable]] = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}

__all__ = [
    "AgnosticNonlinearInteractionBlock",
    "AgnosticResidualNonlinearInteractionBlock",
    "AtomicEnergiesBlock",
    "BesselBasis",
    "EquivariantProductBasisBlock",
    "InteractionBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "RadialEmbeddingBlock",
    "RealAgnosticInteractionBlock",
    "RealAgnosticResidualInteractionBlock",
    "ResidualElementDependentInteractionBlock",
    "ScaleShiftBlock",
    "SymmetricContraction",
    "gate_dict",
    "interaction_classes",
]
