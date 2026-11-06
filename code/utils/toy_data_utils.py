from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from data.toy_dataset import ToyDataset
from utils.toy_runtime_utils import (
    ONE_HOT_MODELS,
    POINT_GROUPS,
    POINT_GROUP_TO_IDX,
    add_scaled_gaussian_noise,
    build_distance_edge_attr,
    build_node_features,
    build_test_metric_names,
    center_positions,
    format_noise_tag,
    parse_noise_levels,
)


DEFAULT_TEST_NOISE_LEVELS = "0.0,0.02,0.05,0.10,0.20"


def maybe_add_pos_noise(
    data,
    split: str,
    noise_alpha: float,
    noise_seed: int,
    deterministic_noise: bool,
):
    if noise_alpha <= 0:
        return data

    noisy_pos, scale = add_scaled_gaussian_noise(
        data.pos.to(dtype=torch.float32),
        sample_key=str(getattr(data, "mid", "")),
        split=split,
        noise_alpha=noise_alpha,
        noise_seed=noise_seed,
        deterministic_noise=deterministic_noise,
        center=False,
    )
    data.pos = center_positions(noisy_pos)
    data.noise_alpha = float(noise_alpha)
    data.noise_scale = float(scale.item())
    return data


def _point_group_label_index(data) -> int:
    return POINT_GROUP_TO_IDX[data.point_group]


def apply_classification_transform(
    data,
    *,
    split: str,
    noise_alpha: float,
    noise_seed: int,
    deterministic_noise: bool,
    model_type: str,
    label_getter: Callable[[object], int] = _point_group_label_index,
    store_atomic_numbers: bool = True,
):
    data = data.clone()
    atomic_numbers = data.x.clone().to(dtype=torch.long)
    data = maybe_add_pos_noise(
        data=data,
        split=split,
        noise_alpha=noise_alpha,
        noise_seed=noise_seed,
        deterministic_noise=deterministic_noise,
    )

    if store_atomic_numbers:
        data.atomic_numbers = atomic_numbers

    data.x = build_node_features(atomic_numbers, model_type)
    data.edge_attr = build_distance_edge_attr(data.pos, data.edge_index)
    data.y = torch.tensor(label_getter(data), dtype=torch.long)
    return data


def build_classification_transform(
    *,
    split: str,
    noise_alpha: float,
    noise_seed: int,
    deterministic_noise: bool,
    model_type: str,
    label_getter: Callable[[object], int] = _point_group_label_index,
    store_atomic_numbers: bool = True,
):
    return partial(
        apply_classification_transform,
        split=split,
        noise_alpha=noise_alpha,
        noise_seed=noise_seed,
        deterministic_noise=deterministic_noise,
        model_type=model_type,
        label_getter=label_getter,
        store_atomic_numbers=store_atomic_numbers,
    )


class ToySymmetryDataModule(pl.LightningDataModule):
    """ToyDataset datamodule for clean or noisy evaluation."""

    def __init__(
        self,
        pkl_path: str,
        processed_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        model_type: str = "hegnn",
        test_noise_levels: Optional[Sequence[float]] = None,
        noise_seed: int = 42,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.pkl_path = pkl_path
        self.processed_root = processed_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.noise_seed = noise_seed
        levels = list(test_noise_levels or [0.0])
        self.test_noise_levels = [alpha for alpha in levels if alpha > 0]
        self.pin_memory = pin_memory
        self.test_loader_names = build_test_metric_names(levels)

        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets: List[ToyDataset] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = ToyDataset(
                root=self.processed_root,
                pkl_path=self.pkl_path,
                split="train",
                transform=self._create_transform(
                    split="train",
                    noise_alpha=0.0,
                    deterministic_noise=False,
                ),
            )
            self.val_dataset = ToyDataset(
                root=self.processed_root,
                pkl_path=self.pkl_path,
                split="val",
                transform=self._create_transform(
                    split="val",
                    noise_alpha=0.0,
                    deterministic_noise=True,
                ),
            )

        if stage in (None, "test"):
            self.test_datasets = [
                ToyDataset(
                    root=self.processed_root,
                    pkl_path=self.pkl_path,
                    split="test",
                    transform=self._create_transform(
                        split="test",
                        noise_alpha=0.0,
                        deterministic_noise=True,
                    ),
                )
            ]
            for noise_alpha in self.test_noise_levels:
                self.test_datasets.append(
                    ToyDataset(
                        root=self.processed_root,
                        pkl_path=self.pkl_path,
                        split="test",
                        transform=self._create_transform(
                            split="test",
                            noise_alpha=noise_alpha,
                            deterministic_noise=True,
                        ),
                    )
                )

    def _create_transform(self, split: str, noise_alpha: float, deterministic_noise: bool):
        return build_classification_transform(
            split=split,
            noise_alpha=noise_alpha,
            noise_seed=self.noise_seed,
            deterministic_noise=deterministic_noise,
            model_type=self.model_type,
        )

    def _build_loader(self, dataset: ToyDataset, shuffle: bool):
        return PyGDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return [self._build_loader(dataset, shuffle=False) for dataset in self.test_datasets]


__all__ = [
    "DEFAULT_TEST_NOISE_LEVELS",
    "ONE_HOT_MODELS",
    "POINT_GROUPS",
    "POINT_GROUP_TO_IDX",
    "ToySymmetryDataModule",
    "apply_classification_transform",
    "build_classification_transform",
    "build_test_metric_names",
    "format_noise_tag",
    "maybe_add_pos_noise",
    "parse_noise_levels",
]

