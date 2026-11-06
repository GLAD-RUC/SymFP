"""
Toy Symmetry Dataset for SymFP.

The official toy dataset stores finite point clouds whose Cartesian coordinates
in ``pos`` have the exact target Euclidean point group. The optional
``structure`` field is kept only for serialization compatibility.
"""

from typing import Any, Sequence

import torch
from torch_geometric.data import Data
from pymatgen.core import Element
from data.base_dataset import BaseMPDataset


class ToyDataset(BaseMPDataset):
    """
    Toy symmetry dataset with 32 crystallographic point groups.

    Each sample contains:
        - x: atomic numbers (n,)
        - pos: Cartesian coordinates (n, 3)
        - edge_index: fully connected graph (2, n*(n-1))
        - point_group: point group label (Hermann-Mauguin notation)

    This dataset is synthetic and has no target tensors - it's designed
    for testing symmetry-equivariant representations on finite point clouds.

    Usage:
        dataset = ToyDataset(
            root='data/processed',
            pkl_path='dataset/toydataset/toy_symmetry.pkl',
            split='train'
        )
    """

    DATASET_NAME = 'toy_symmetry'

    def __init__(self, root, pkl_path, split='train',
                 transform=None, pre_transform=None):
        super().__init__(root, pkl_path, split, transform, pre_transform)

    def structure_to_x_pos(self, structure_dict):
        """
        Override to use pre-computed x and pos from record.

        The official toy dataset already stores x and pos as tensors, so we can
        return them directly instead of re-extracting from structure dict.
        """
        # This method will be called with the record dict, not structure dict
        # We'll handle this in process() instead
        raise NotImplementedError("Use process() method directly")

    @staticmethod
    def _to_long_tensor(values: Any) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(dtype=torch.long)
        return torch.tensor(values, dtype=torch.long)

    @staticmethod
    def _to_float_tensor(values: Any) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(dtype=torch.float32)
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _species_to_x(species: Sequence[str]) -> torch.Tensor:
        atomic_numbers = [Element(symbol).Z for symbol in species]
        return torch.tensor(atomic_numbers, dtype=torch.long)

    def process(self):
        """Process raw pkl into PyG Data objects."""
        split_data = self.get_split_data()
        data_list = []

        for record in split_data:
            # Prefer explicit finite-point-cloud tensors from the official format.
            if 'x' in record:
                x = self._to_long_tensor(record['x'])
            elif 'species' in record:
                x = self._species_to_x(record['species'])
            else:
                x, _ = self.structure_to_x_pos(record['structure'])

            if 'pos' in record:
                pos = self._to_float_tensor(record['pos'])
            else:
                _, pos = self.structure_to_x_pos(record['structure'])

            # Build fully-connected edge index
            edge_index = self.build_fully_connected_edge_index(len(x))

            data = Data(
                x=x,
                pos=pos,
                edge_index=edge_index,
                num_nodes=len(x),
                mid=record['mid'],
                point_group=record['point_group'],
            )
            if 'species' in record:
                data.species = list(record['species'])
            if 'point_group_hm' in record:
                data.point_group_hm = record['point_group_hm']
            if 'pg_order' in record:
                data.pg_order = int(record['pg_order'])

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
