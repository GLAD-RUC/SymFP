"""Chain selection and mmCIF loading for the protein assemblies.

Kept apart from the baseline drivers so that structure loading and figure
composition do not depend on the optional external symmetry programs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from torch import Tensor
import torch

DEFAULT_ACCESSION = "P03437"


@dataclass
class HAStructure:
    """One HA structure with selected protein chains."""

    pdb_id: str
    title: str
    cif_path: Path
    selected_chains: tuple[str, ...]
    structure: object
    model: object
    chain_residue_counts: dict[str, int]


def as_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def polypeptide_chains(mmcif: MMCIF2Dict) -> list[str]:
    entity_types = as_list(mmcif.get("_entity_poly.type", []))
    chain_fields = as_list(mmcif.get("_entity_poly.pdbx_strand_id", []))
    selected: list[str] = []
    for entity_type, chain_field in zip(entity_types, chain_fields):
        if "polypeptide" not in str(entity_type).lower():
            continue
        for chain in str(chain_field).split(","):
            chain = chain.strip()
            if chain and chain not in selected:
                selected.append(chain)
    return selected


def accession_chains(mmcif: MMCIF2Dict, accession: str) -> list[str]:
    accessions = as_list(mmcif.get("_struct_ref_seq.pdbx_db_accession", []))
    chain_fields = as_list(mmcif.get("_struct_ref_seq.pdbx_strand_id", []))
    selected: list[str] = []
    for chain_accession, chain_field in zip(accessions, chain_fields):
        if chain_accession != accession:
            continue
        for chain in str(chain_field).split(","):
            chain = chain.strip()
            if chain and chain not in selected:
                selected.append(chain)
    return selected


def count_standard_residues(chain) -> int:
    return sum(1 for residue in chain if residue.id[0] == " ")


def load_ha_structure(path: str | Path, *, accession: str = DEFAULT_ACCESSION) -> HAStructure:
    """Load one HA mmCIF and keep only the protein chains of interest."""

    resolved = Path(path).resolve()
    mmcif = MMCIF2Dict(str(resolved))

    pdb_id = as_list(mmcif.get("_entry.id", resolved.stem))[0].strip()
    title = as_list(mmcif.get("_struct.title", ""))[0].strip()

    selected_chains = polypeptide_chains(mmcif)
    if not selected_chains:
        selected_chains = accession_chains(mmcif, accession)
    if not selected_chains:
        raise ValueError(f"No chains annotated with accession={accession} in {resolved}.")

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(resolved))
    model = next(structure.get_models())
    selected_chains = tuple(sorted(selected_chains))

    available_chains = set(model.child_dict)
    missing = [chain_id for chain_id in selected_chains if chain_id not in available_chains]
    if missing:
        raise ValueError(f"Missing selected chains in parsed structure {resolved}: {missing}")

    chain_residue_counts = {
        chain_id: count_standard_residues(model.child_dict[chain_id])
        for chain_id in selected_chains
    }
    return HAStructure(
        pdb_id=pdb_id,
        title=title,
        cif_path=resolved,
        selected_chains=selected_chains,
        structure=structure,
        model=model,
        chain_residue_counts=chain_residue_counts,
    )


def build_knn_edge_index(
    pos: Tensor,
    *,
    k: int,
    chunk_size: int = 1024,
    include_ties: bool = False,
    tie_rtol: float = 1e-6,
    tie_atol: float = 1e-7,
) -> Tensor:
    """Build directed neighbours, optionally including the complete kth-distance shell.

    ``include_ties=False`` reproduces the deposited fixed-k graphs. Use True for
    exact symmetric inputs: selecting a subset of a tied shell can depend on atom
    order. Shell inclusion does not remove discontinuities at the distance cutoff.
    """
    if pos.ndim != 2 or pos.size(-1) != 3:
        raise ValueError(f"Expected positions with shape (N, 3), got {tuple(pos.shape)}.")

    num_nodes = int(pos.size(0))
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)

    k = max(1, min(int(k), num_nodes - 1))
    chunk_size = max(1, int(chunk_size))

    edge_src_parts: list[Tensor] = []
    edge_dst_parts: list[Tensor] = []

    for start in range(0, num_nodes, chunk_size):
        end = min(num_nodes, start + chunk_size)
        chunk = pos[start:end]
        distances = torch.cdist(chunk, pos)
        row_indices = torch.arange(start, end, dtype=torch.long, device=pos.device)
        distances[torch.arange(end - start, device=pos.device), row_indices] = float("inf")
        if include_ties:
            cutoff = torch.topk(distances, k=k, largest=False, dim=1).values[:, -1:]
            mask = distances <= cutoff + tie_atol + tie_rtol * cutoff
            source, target = torch.where(mask)
            edge_src_parts.append(source + start)
            edge_dst_parts.append(target)
            continue
        knn_indices = torch.topk(distances, k=k, largest=False, dim=1).indices
        edge_src_parts.append(row_indices.unsqueeze(1).expand(-1, k).reshape(-1))
        edge_dst_parts.append(knn_indices.reshape(-1).to(dtype=torch.long))

    return torch.stack(
        [torch.cat(edge_src_parts, dim=0), torch.cat(edge_dst_parts, dim=0)],
        dim=0,
    )
