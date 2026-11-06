from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .utils import build_model_output


SignatureKey = Tuple[int, str]


def split_flat_irreps(tensor: Tensor, hidden_irreps) -> List[Tuple[SignatureKey, Tensor]]:
    if tensor.dim() != 2:
        raise ValueError(f"Expected a rank-2 tensor [B, dim], got shape={tuple(tensor.shape)}.")

    batch_size = tensor.size(0)
    offset = 0
    blocks: List[Tuple[SignatureKey, Tensor]] = []
    for mul, ir in hidden_irreps:
        width = mul * ir.dim
        block = tensor[:, offset : offset + width].view(batch_size, mul, ir.dim)
        key = (ir.l, "e" if ir.p == 1 else "o")
        blocks.append((key, block))
        offset += width

    if offset != tensor.size(1):
        raise ValueError(
            f"hidden_irreps expects width={offset}, but tensor width={tensor.size(1)}."
        )
    return blocks


def merge_irrep_blocks(
    tensor: Tensor,
    hidden_irreps,
) -> Tuple[List[SignatureKey], Dict[SignatureKey, Tensor], Dict[SignatureKey, int]]:
    merged: Dict[SignatureKey, List[Tensor]] = defaultdict(list)
    signature_keys: List[SignatureKey] = []
    channels: Dict[SignatureKey, int] = defaultdict(int)

    for key, block in split_flat_irreps(tensor, hidden_irreps):
        if key not in merged:
            signature_keys.append(key)
        merged[key].append(block)
        channels[key] += block.size(1)

    merged_blocks = {
        key: torch.cat(chunks, dim=1)
        for key, chunks in merged.items()
    }
    return signature_keys, merged_blocks, dict(channels)


def block_singular_values(block: Tensor) -> Tensor:
    if block.dim() != 3:
        raise ValueError(f"Expected a rank-3 block [B, C, D], got shape={tuple(block.shape)}.")
    return torch.linalg.svdvals(block.to(dtype=torch.float32))


def hard_rank_from_block(
    block: Tensor,
    rank_tol: float,
) -> Tensor:
    if block.dim() != 3:
        raise ValueError(f"Expected a rank-3 block [B, C, D], got shape={tuple(block.shape)}.")
    return torch.linalg.matrix_rank(block.to(dtype=torch.float32), atol=float(rank_tol))


class HardFingerprintOperator(nn.Module):
    METRIC_NAMES = ("rank", "trace", "energy_ratio", "r2", "r3", "stable_rank")

    def __init__(self, hidden_irreps, rank_tol: float = 1e-4, eps: float = 1e-8):
        super().__init__()
        self.hidden_irreps = hidden_irreps
        self.rank_tol = rank_tol
        self.eps = eps

        signature_keys: List[SignatureKey] = []
        channels_by_key: Dict[SignatureKey, int] = defaultdict(int)
        for mul, ir in hidden_irreps:
            key = (ir.l, "e" if ir.p == 1 else "o")
            if key not in channels_by_key:
                signature_keys.append(key)
            channels_by_key[key] += int(mul)

        self.signature_keys = signature_keys
        self.signature_names = [f"{ell}{parity}" for ell, parity in signature_keys]
        self.channels_by_key = dict(channels_by_key)
        self.num_signatures = len(signature_keys)
        self.metric_names = list(self.METRIC_NAMES)
        self.metric_dim = len(self.metric_names)
        self.output_dim = self.num_signatures * self.metric_dim

    def forward(self, graph_sh: Tensor) -> Dict[str, Tensor]:
        signature_keys, merged_blocks, _ = merge_irrep_blocks(graph_sh, self.hidden_irreps)
        if signature_keys != self.signature_keys:
            raise ValueError("Merged signature keys do not match initialization-time hidden_irreps.")

        traces = []
        trace2s = []
        trace3s = []
        ranks = []

        for key in self.signature_keys:
            block = merged_blocks[key]
            gram = torch.bmm(block.transpose(1, 2), block)
            trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(dim=-1)
            gram2 = torch.bmm(gram, gram)
            trace2 = torch.diagonal(gram2, dim1=-2, dim2=-1).sum(dim=-1)
            gram3 = torch.bmm(gram2, gram)
            trace3 = torch.diagonal(gram3, dim1=-2, dim2=-1).sum(dim=-1)

            traces.append(trace)
            trace2s.append(trace2)
            trace3s.append(trace3)
            ranks.append(hard_rank_from_block(block, rank_tol=self.rank_tol).to(block.dtype))

        trace_tensor = torch.stack(traces, dim=-1)
        total_trace = trace_tensor.sum(dim=-1, keepdim=True) + self.eps
        trace2_tensor = torch.stack(trace2s, dim=-1)
        trace3_tensor = torch.stack(trace3s, dim=-1)
        rank_tensor = torch.stack(ranks, dim=-1)

        energy_ratio = trace_tensor / total_trace
        r2 = trace2_tensor / (trace_tensor.square() + self.eps)
        r3 = trace3_tensor / (trace_tensor.pow(3) + self.eps)
        stable_rank = 1.0 / (r2 + self.eps)

        hard_metrics_3d = torch.stack(
            [rank_tensor, trace_tensor, energy_ratio, r2, r3, stable_rank],
            dim=-1,
        )
        hard_feat = hard_metrics_3d.reshape(graph_sh.size(0), -1)

        return {
            "hard_feat": hard_feat,
            "hard_metrics_3d": hard_metrics_3d,
            "rank_signature": rank_tensor,
            "trace_signature": trace_tensor,
            "energy_ratio_signature": energy_ratio,
            "r2_signature": r2,
            "r3_signature": r3,
            "stable_rank_signature": stable_rank,
        }


class SymFPHard(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_irreps=None,
        rank_tol: float = 1e-4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_irreps = hidden_irreps if hidden_irreps is not None else getattr(encoder, "hidden_irreps", None)
        if self.hidden_irreps is None:
            raise ValueError(
                "SymFPHard requires hidden_irreps. Pass it explicitly or provide an encoder with encoder.hidden_irreps."
            )

        self.hard_op = HardFingerprintOperator(
            hidden_irreps=self.hidden_irreps,
            rank_tol=rank_tol,
            eps=eps,
        )
        self.signature_keys = self.hard_op.signature_keys
        self.signature_names = self.hard_op.signature_names
        self.metric_names = self.hard_op.metric_names
        self.output_dim = self.hard_op.output_dim

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
            raise ValueError(
                f"Expected graph_sh to have shape [B, dim], got {tuple(graph_sh.shape)}."
            )
        if graph_sh.size(-1) != expected_dim:
            raise ValueError(
                "Encoder graph_sh width does not match hidden_irreps.dim. "
                f"Expected {expected_dim}, got {graph_sh.size(-1)}. "
                "This usually means the encoder is still exporting only non-scalar irreps."
            )
        return graph_sh

    def forward(self, data: Data):
        encoder_outputs = self.encoder(data)
        full_graph_sh = self._resolve_graph_sh(encoder_outputs)
        hard_outputs = self.hard_op(full_graph_sh)

        return build_model_output(
            node_feat=encoder_outputs.get("node_feat"),
            node_sh=encoder_outputs.get("node_sh"),
            graph_feat=hard_outputs["hard_feat"],
            graph_sh=full_graph_sh,
            node_pos=encoder_outputs.get("node_pos"),
            encoder_graph_feat=encoder_outputs.get("graph_feat"),
            encoder_graph_sh=full_graph_sh,
            hard_feat=hard_outputs["hard_feat"],
            hard_metrics_3d=hard_outputs["hard_metrics_3d"],
            rank_signature=hard_outputs["rank_signature"],
            trace_signature=hard_outputs["trace_signature"],
            energy_ratio_signature=hard_outputs["energy_ratio_signature"],
            r2_signature=hard_outputs["r2_signature"],
            r3_signature=hard_outputs["r3_signature"],
            stable_rank_signature=hard_outputs["stable_rank_signature"],
            node_irreps=encoder_outputs.get("node_irreps"),
            graph_irreps=encoder_outputs.get("graph_irreps", full_graph_sh),
        )


__all__ = [
    "SignatureKey",
    "split_flat_irreps",
    "merge_irrep_blocks",
    "block_singular_values",
    "hard_rank_from_block",
    "HardFingerprintOperator",
    "SymFPHard",
]
