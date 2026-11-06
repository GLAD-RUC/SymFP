"""Natural-parity rank templates and fixed-threshold evaluation."""
from typing import List, Tuple, Dict
import torch
from models.symfp_hard import block_singular_values, merge_irrep_blocks
from utils.toy_data_utils import POINT_GROUPS, POINT_GROUP_TO_IDX

def build_natural_parity_template(max_ell: int) -> Tuple[List[Tuple[int, str]], torch.Tensor]:
    """Character-average multiplicities over the natural-parity trunk to `max_ell`, for all 32 groups."""
    from models.symmetry_groups import invariant_multiplicity  # noqa

    signature_keys: List[Tuple[int, str]] = []
    for ell in range(max_ell + 1):
        parity = "e" if ell % 2 == 0 else "o"
        signature_keys.append((ell, parity))

    rows = []
    for pg in POINT_GROUPS:
        row = []
        for ell, parity in signature_keys:
            row.append(invariant_multiplicity(pg, ell, parity))
        rows.append(row)
    template = torch.tensor(rows, dtype=torch.long)  # [32, K]
    return signature_keys, template

def collect_sigmas(symfp_encoder, data_module, splits, device, signature_keys, hidden_irreps):
    """Return dict split -> list of (mid, true_pg, sigma_vectors_per_key)."""
    out = {s: [] for s in splits}
    for split_name, loader in splits.items():
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                enc_out = symfp_encoder(batch)
                graph_sh = enc_out["graph_sh"] if isinstance(enc_out, dict) else enc_out
            keys_here, merged, _ = merge_irrep_blocks(graph_sh, hidden_irreps)
            assert keys_here == signature_keys, (keys_here, signature_keys)
            per_key = []
            for key in signature_keys:
                block = merged[key].to(dtype=torch.float32)
                per_key.append(block_singular_values(block).cpu())
            batch_size = per_key[0].shape[0]
            true_pgs = [POINT_GROUPS[int(y)] for y in batch.y.view(-1).tolist()]
            mids = list(batch.mid) if hasattr(batch, "mid") else [f"{split_name}_{i}" for i in range(batch_size)]
            for i in range(batch_size):
                sigmas = [per_key[k][i] for k in range(len(signature_keys))]
                out[split_name].append((mids[i], true_pgs[i], sigmas))
    return out

def _effective_tol(sigmas_per_key: List[torch.Tensor], tol: float, relative: bool) -> float:
    """The threshold applied to this structure's singular values.

    In relative mode the level is tol times the leading degree-zero singular value
    sigma_1^(0), which is admissible for every point group and sets a per-structure
    scale; in absolute mode the raw tol is used.
    """
    if not relative:
        return tol
    s0 = float(sigmas_per_key[0][0]) if sigmas_per_key[0].numel() else 0.0
    return tol * s0

def classify_with_tol(sigmas_per_key: List[torch.Tensor], template: torch.Tensor, tol: float,
                      relative: bool = False) -> int:
    """Return predicted point-group index."""
    thr = _effective_tol(sigmas_per_key, tol, relative)
    ranks = torch.tensor([int((sv > thr).sum()) for sv in sigmas_per_key], dtype=torch.long)
    # L1 distance against template rows
    distances = (template - ranks.unsqueeze(0)).abs().sum(dim=-1)
    return int(distances.argmin().item())

def evaluate_at_tol(cached, template, tol, relative: bool = False) -> Dict[str, Dict[str, float]]:
    result = {}
    for split, entries in cached.items():
        correct = 0
        exact = 0
        for mid, true_pg, sigmas in entries:
            pred_idx = classify_with_tol(sigmas, template, tol, relative)
            pred_pg = POINT_GROUPS[pred_idx]
            if pred_pg == true_pg:
                correct += 1
            true_idx = POINT_GROUP_TO_IDX[true_pg]
            thr = _effective_tol(sigmas, tol, relative)
            pred_ranks = tuple(int((sv > thr).sum()) for sv in sigmas)
            true_ranks = tuple(int(x) for x in template[true_idx].tolist())
            if pred_ranks == true_ranks:
                exact += 1
        n = len(entries)
        result[split] = {"accuracy": correct / n, "exact_match_rate": exact / n, "count": n}
    return result

class SymFPMinimal(torch.nn.Module):
    """Thin wrapper that reproduces the SymFP forward interface but skips soft-score computation."""

    def __init__(self, encoder, hidden_irreps):
        super().__init__()
        self.encoder = encoder
        self.hidden_irreps = hidden_irreps

    def forward(self, data):
        return self.encoder(data)
