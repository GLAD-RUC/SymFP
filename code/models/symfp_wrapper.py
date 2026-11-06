from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .symmetry_groups import POINT_GROUPS as TOY_POINT_GROUPS
from .symmetry_groups import invariant_multiplicity
from .symfp_hard import (
    SignatureKey,
    block_singular_values,
    hard_rank_from_block,
    merge_irrep_blocks,
)


def format_signature_key(key: SignatureKey) -> str:
    ell, parity = key
    return f"{ell}{parity}"


def _signature_metadata(hidden_irreps) -> tuple[List[SignatureKey], Dict[SignatureKey, int], Dict[SignatureKey, int]]:
    signature_keys: List[SignatureKey] = []
    channels_by_key: Dict[SignatureKey, int] = defaultdict(int)
    max_rank_by_key: Dict[SignatureKey, int] = {}

    for mul, ir in hidden_irreps:
        key = (ir.l, "e" if ir.p == 1 else "o")
        if key not in channels_by_key:
            signature_keys.append(key)
        channels_by_key[key] += int(mul)
        max_rank_by_key[key] = min(channels_by_key[key], int(ir.dim))

    return signature_keys, dict(channels_by_key), max_rank_by_key


class SymFPWrapper(nn.Module):
    """Wrap an equivariant encoder and expose singular-value SymFP diagnostics."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        hidden_irreps=None,
        rank_tol: float = 1e-4,
        ls_eps: float = 1e-10,
        point_groups: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_irreps = hidden_irreps if hidden_irreps is not None else getattr(encoder, "hidden_irreps", None)
        if self.hidden_irreps is None:
            raise ValueError(
                "SymFPWrapper requires hidden_irreps. Pass it explicitly or provide encoder.hidden_irreps."
            )

        self.rank_tol = float(rank_tol)
        self.ls_eps = float(ls_eps)
        self.point_groups = list(point_groups or TOY_POINT_GROUPS)
        self.point_group_to_idx = {
            point_group: idx
            for idx, point_group in enumerate(self.point_groups)
        }

        signature_keys, channels_by_key, max_rank_by_key = _signature_metadata(self.hidden_irreps)
        self.signature_keys = signature_keys
        self.signature_names = [format_signature_key(key) for key in self.signature_keys]
        self.channels_by_key = channels_by_key
        self.max_rank_by_key = max_rank_by_key
        self.output_dim = len(self.signature_keys)

        raw_rows = []
        observable_rows = []
        for point_group in self.point_groups:
            raw_row = []
            observable_row = []
            for key in self.signature_keys:
                ell, parity = key
                multiplicity = int(invariant_multiplicity(point_group, ell, parity))
                raw_row.append(float(multiplicity))
                observable_row.append(float(min(multiplicity, self.max_rank_by_key[key])))
            raw_rows.append(raw_row)
            observable_rows.append(observable_row)

        self.register_buffer(
            "theoretical_rank_matrix",
            torch.tensor(raw_rows, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "observable_rank_matrix",
            torch.tensor(observable_rows, dtype=torch.float32),
            persistent=False,
        )

    def _resolve_graph_sh(self, encoder_outputs: Dict[str, Tensor]) -> Tensor:
        graph_sh = encoder_outputs.get("graph_sh")
        graph_irreps = encoder_outputs.get("graph_irreps")
        expected_dim = self.hidden_irreps.dim

        if graph_irreps is not None:
            if graph_irreps.dim() != 2:
                raise ValueError(
                    f"Expected graph_irreps to have shape [B, dim], got {tuple(graph_irreps.shape)}."
                )
            if graph_irreps.size(-1) != expected_dim:
                raise ValueError(
                    f"graph_irreps width mismatch: expected {expected_dim}, got {graph_irreps.size(-1)}."
                )
            return graph_irreps

        if graph_sh is None:
            raise ValueError("Encoder output must contain graph_sh or graph_irreps.")
        if graph_sh.dim() != 2:
            raise ValueError(f"Expected graph_sh to have shape [B, dim], got {tuple(graph_sh.shape)}.")
        if graph_sh.size(-1) != expected_dim:
            raise ValueError(
                f"graph_sh width mismatch: expected {expected_dim}, got {graph_sh.size(-1)}."
            )
        return graph_sh

    def _compute_soft_scores(self, block_singular_values: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        per_key_scores = []
        num_groups = len(self.point_groups)

        for key_idx, singular_values in enumerate(block_singular_values):
            squared = singular_values.square()
            total_energy = squared.sum(dim=-1)
            group_scores = []

            for group_idx in range(num_groups):
                multiplicity = int(self.observable_rank_matrix[group_idx, key_idx].item())
                if multiplicity >= singular_values.size(-1):
                    tail_energy = torch.zeros_like(total_energy)
                else:
                    tail_energy = squared[:, multiplicity:].sum(dim=-1)
                group_scores.append(tail_energy / (total_energy + self.ls_eps))

            per_key_scores.append(torch.stack(group_scores, dim=-1))

        ls_matrix = torch.stack(per_key_scores, dim=-1).mean(dim=-1)
        pls_matrix = -torch.log10(ls_matrix + self.ls_eps)
        return ls_matrix, pls_matrix

    def forward(self, data: Data, *, compute_soft_scores: bool = True) -> Dict[str, object]:
        encoder_outputs = self.encoder(data)
        graph_sh = self._resolve_graph_sh(encoder_outputs)
        signature_keys, merged_blocks, _ = merge_irrep_blocks(graph_sh, self.hidden_irreps)
        if signature_keys != self.signature_keys:
            raise ValueError("Merged signature keys do not match initialization-time hidden_irreps.")

        singular_value_blocks: List[Tensor] = []
        rank_columns = []
        for key in self.signature_keys:
            block = merged_blocks[key].to(dtype=torch.float32)
            singular_values = block_singular_values(block)
            singular_value_blocks.append(singular_values)
            rank_columns.append(hard_rank_from_block(block, rank_tol=self.rank_tol).to(dtype=torch.long))

        rank_signature = torch.stack(rank_columns, dim=-1)
        outputs: Dict[str, object] = {
            "graph_sh": graph_sh,
            "encoder_outputs": encoder_outputs,
            "signature_keys": list(self.signature_keys),
            "block_singular_values": singular_value_blocks,
            "rank_signature": rank_signature,
        }
        if compute_soft_scores:
            ls_matrix, pls_matrix = self._compute_soft_scores(singular_value_blocks)
            outputs["ls_matrix"] = ls_matrix
            outputs["pls_matrix"] = pls_matrix
        return outputs


__all__ = [
    "SignatureKey",
    "SymFPWrapper",
    "format_signature_key",
]
