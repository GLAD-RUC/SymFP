"""
Base PyG Dataset for SymFP materials.
Handles loading from pkl and building fully-connected graphs.
"""

import os
import pickle
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from pymatgen.core import Structure


class BaseMPDataset(InMemoryDataset):
    """Base class for SymFP PyG datasets."""

    # Subclasses should set this to their dataset name
    DATASET_NAME = 'base'

    def __init__(self, root, pkl_path, split='train', transform=None, pre_transform=None):
        self.pkl_path = pkl_path
        self.split = split
        self._raw_data = None
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0], data_cls=Data)

    @property
    def raw_file_names(self):
        return [self.pkl_path]

    @property
    def processed_file_names(self):
        return [f'{self.DATASET_NAME}_{self.split}.pt']

    def load_pkl(self):
        """Load pkl data once and cache."""
        if self._raw_data is None:
            with open(self.pkl_path, 'rb') as f:
                self._raw_data = pickle.load(f)
        return self._raw_data

    def get_split_data(self):
        """Get data for current split."""
        data = self.load_pkl()
        return data[self.split]

    def structure_to_x_pos(self, structure_dict):
        """Convert pymatgen structure dict to atomic numbers and positions."""
        struct = Structure.from_dict(structure_dict)
        x = torch.tensor([site.specie.Z for site in struct], dtype=torch.long)
        pos = torch.tensor(struct.cart_coords, dtype=torch.float32)
        return x, pos

    def build_fully_connected_edge_index(self, n_nodes):
        """Build fully-connected edge index without self-loops."""
        if n_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long)

        # Create all pairs
        i, j = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='ij')
        i, j = i.flatten(), j.flatten()

        # Remove self-loops
        mask = i != j
        edge_index = torch.stack([i[mask], j[mask]], dim=0)
        return edge_index

    def process(self):
        """Process raw pkl into PyG Data objects."""
        split_data = self.get_split_data()
        data_list = []

        for record in split_data:
            x, pos = self.structure_to_x_pos(record['structure'])
            edge_index = self.build_fully_connected_edge_index(len(x))

            data = self.build_data(record, x, pos, edge_index)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def build_data(self, record, x, pos, edge_index):
        """Build Data object. Override in subclasses to add task-specific fields."""
        data = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            num_nodes=len(x),
            mid=record['mid'],
        )
        return data
