"""
Point group classification training script for the toy symmetry dataset.

By default this script measures natural robustness:
    - train on clean coordinates
    - validate on clean coordinates
    - test on both clean and noisy coordinates

Noise is injected online in the DataModule before edge features are recomputed,
so both `pos` and `edge_attr` stay consistent.
"""

import argparse
import os
import sys
from functools import partial
from typing import Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch_geometric.loader import DataLoader as PyGDataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.toy_dataset import ToyDataset
from utils.checkpoint_utils import ELL_MODELS, export_normalized_toy_checkpoint
from utils.lightning_utils import (
    EpochTrackerCallback,
    build_default_run_name,
    build_logger,
    build_run_dir,
    is_global_zero_process,
    merge_test_results,
    parse_trainer_limit,
)
from utils.toy_backbone_utils import build_classification_backbone
from utils.toy_data_utils import (
    DEFAULT_TEST_NOISE_LEVELS,
    POINT_GROUPS,
    POINT_GROUP_TO_IDX,
    apply_classification_transform,
    build_test_metric_names,
    parse_noise_levels,
)
from utils.toy_training_reporting import (
    add_retention_metrics,
    build_final_logger_metrics,
    print_supervised_run_header,
    print_supervised_test_summary,
    save_supervised_summary,
)

SO3_CLASSES = ["C1", "C2", "C3", "C4", "C6", "D2", "D3", "D4", "D6", "T", "O"]
SO3_TO_IDX = {name: idx for idx, name in enumerate(SO3_CLASSES)}
PG_TO_SO3 = {
    "1": "C1",
    "-1": "C1",
    "2": "C2",
    "m": "C2",
    "2/m": "C2",
    "3": "C3",
    "-3": "C3",
    "4": "C4",
    "-4": "C4",
    "4/m": "C4",
    "4mm": "C4",
    "6": "C6",
    "-6": "C6",
    "6/m": "C6",
    "6mm": "C6",
    "222": "D2",
    "mm2": "D2",
    "mmm": "D2",
    "32": "D3",
    "3m": "D3",
    "-3m": "D3",
    "422": "D4",
    "-42m": "D4",
    "4/mmm": "D4",
    "622": "D6",
    "-6m2": "D6",
    "6/mmm": "D6",
    "23": "T",
    "m-3": "T",
    "432": "O",
    "-43m": "O",
    "m-3m": "O",
}

NUM_CLASSES = len(POINT_GROUPS)


class ToyClassificationDataModule(pl.LightningDataModule):
    """DataModule for clean-train / clean-val / multi-noise-test evaluation."""

    def __init__(
        self,
        pkl_path: str,
        processed_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        model_type: str = "hegnn",
        so3: bool = False,
        test_noise_levels: Optional[Sequence[float]] = None,
        noise_seed: int = 42,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.pkl_path = pkl_path
        self.processed_root = processed_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.so3 = so3
        self.noise_seed = noise_seed
        levels = list(test_noise_levels or [0.0])
        self.test_noise_levels = [alpha for alpha in levels if alpha > 0]
        self.pin_memory = pin_memory
        self.test_loader_names = build_test_metric_names(levels)

        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets: List[ToyDataset] = []

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = ToyDataset(
                root=self.processed_root,
                pkl_path=self.pkl_path,
                split="train",
                transform=self._create_transform(
                    split="train", noise_alpha=0.0, deterministic_noise=False
                ),
            )
            self.val_dataset = ToyDataset(
                root=self.processed_root,
                pkl_path=self.pkl_path,
                split="val",
                transform=self._create_transform(
                    split="val", noise_alpha=0.0, deterministic_noise=True
                ),
            )

        if stage in (None, "test"):
            self.test_datasets = [
                ToyDataset(
                    root=self.processed_root,
                    pkl_path=self.pkl_path,
                    split="test",
                    transform=self._create_transform(
                        split="test", noise_alpha=0.0, deterministic_noise=True
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

    def _label_index(self, data) -> int:
        if self.so3:
            return SO3_TO_IDX[PG_TO_SO3[data.point_group]]
        return POINT_GROUP_TO_IDX[data.point_group]

    def _create_transform(
        self, split: str, noise_alpha: float, deterministic_noise: bool
    ):
        return partial(
            apply_classification_transform,
            split=split,
            noise_alpha=noise_alpha,
            noise_seed=self.noise_seed,
            deterministic_noise=deterministic_noise,
            model_type=self.model_type,
            label_getter=self._label_index,
            store_atomic_numbers=False,
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
        return [
            self._build_loader(dataset, shuffle=False) for dataset in self.test_datasets
        ]


class PointGroupClassifier(pl.LightningModule):
    """Point group classifier with clean-train / noisy-test reporting."""

    def __init__(
        self,
        model_type: str = "hegnn",
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 64,
        num_layers: int = 4,
        max_ell: int = 6,
        irreps_channels: int = 16,
        mace_correlation: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        freeze_encoder: bool = False,
        so3: bool = False,
        test_metric_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["test_metric_names"])

        self.model_type = model_type
        self.so3 = so3
        self.num_classes = len(SO3_CLASSES) if so3 else num_classes
        self.test_metric_names = list(test_metric_names or ["clean"])
        self.backbone, feature_dim = build_classification_backbone(
            model_type=model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_ell=max_ell,
            irreps_channels=irreps_channels,
            mace_correlation=mace_correlation,
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_classes),
        )

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        frozen = sum(p.numel() for p in self.backbone.parameters())
        trainable = 0

        clf_params = sum(p.numel() for p in self.classifier.parameters())
        print(f"\n{'=' * 60}")
        print("Encoder FROZEN")
        print(f"  Frozen params: {frozen:,}")
        print(f"  Trainable (backbone): {trainable:,}")
        print(f"  Trainable (classifier): {clf_params:,}")
        print(f"{'=' * 60}\n")

    def forward(self, batch):
        outputs = self.backbone(batch)
        features = outputs["graph_feat"]
        logits = self.classifier(features)
        return logits

    def _compute_loss_acc(self, batch):
        logits = self(batch)
        loss = F.cross_entropy(logits, batch.y)
        acc = (logits.argmax(dim=-1) == batch.y).float().mean()
        return loss, acc, batch.num_graphs

    def training_step(self, batch, batch_idx):
        del batch_idx
        loss, acc, batch_size = self._compute_loss_acc(batch)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        loss, acc, batch_size = self._compute_loss_acc(batch)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        del batch_idx
        metric_name = self.test_metric_names[dataloader_idx]
        loss, acc, batch_size = self._compute_loss_acc(batch)
        self.log(
            f"test_{metric_name}_loss",
            loss,
            prog_bar=dataloader_idx == 0,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=batch_size,
        )
        self.log(
            f"test_{metric_name}_acc",
            acc,
            prog_bar=dataloader_idx == 0,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=batch_size,
        )
        return {
            f"test_{metric_name}_loss": loss.detach(),
            f"test_{metric_name}_acc": acc.detach(),
        }

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train toy point-group classifier with noisy test evaluation"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="hegnn",
        choices=[
            "egnn",
            "equiformer",
            "equiformer_balance",
            "hegnn",
            "hegnn_balance",
            "schnet",
            "dimenet",
            "gvp",
            "tfn",
            "tfn_balance",
            "mace",
            "mace_balance",
        ],
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument(
        "--max_ell",
        type=int,
        default=None,
        help="Default: 6 for high-order models. Non-equivariant models ignore this argument.",
    )
    parser.add_argument("--irreps_channels", type=int, default=16)
    parser.add_argument("--mace_correlation", type=int, default=2)

    parser.add_argument("--pkl_path", type=str, default="datasets/toy/toy_symmetry.pkl")
    parser.add_argument("--processed_root", type=str, default="outputs/processed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.set_defaults(pin_memory=True)
    parser.add_argument(
        "--test_noise_levels", type=str, default=DEFAULT_TEST_NOISE_LEVELS
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "ddp", "ddp_find_unused_parameters_true"],
    )
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--limit_train_batches", type=parse_trainer_limit, default=1.0)
    parser.add_argument("--limit_val_batches", type=parse_trainer_limit, default=1.0)
    parser.add_argument("--limit_test_batches", type=parse_trainer_limit, default=1.0)

    parser.add_argument(
        "--logger",
        type=str,
        default="none",
        choices=["tensorboard", "wandb", "swanlab", "none"],
    )
    parser.add_argument("--project", type=str, default="toy-supervised")
    parser.add_argument("--swanlab_project", type=str, default="toy-supervised")
    parser.add_argument("--swanlab_run_name", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default="outputs/toy_supervised")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--export_best_ckpt",
        action="store_true",
        help="Export the best checkpoint to a normalized filename under export_dest_dir.",
    )
    parser.add_argument(
        "--export_dest_dir",
        type=str,
        default="outputs/checkpoints",
        help="Destination directory used with --export_best_ckpt.",
    )
    parser.add_argument(
        "--skip_manifest_update",
        action="store_true",
        help="Do not update ckpts/toydataset manifest.csv during normalized export.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--so3", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    parser.add_argument("--disable_early_stopping", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_ell is None:
        args.max_ell = 6
    test_noise_levels = parse_noise_levels(args.test_noise_levels)
    test_metric_names = build_test_metric_names(test_noise_levels)
    run_dir = build_run_dir(
        work_dir=args.work_dir,
        run_name=args.run_name,
        fallback_name_factory=partial(
            build_default_run_name,
            model=args.model,
            task="toy-supervised",
            seed=args.seed,
        ),
    )

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed, workers=True)

    dm = ToyClassificationDataModule(
        pkl_path=args.pkl_path,
        processed_root=args.processed_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_type=args.model,
        so3=args.so3,
        test_noise_levels=test_noise_levels,
        noise_seed=args.seed,
        pin_memory=args.pin_memory,
    )

    model = PointGroupClassifier(
        model_type=args.model,
        num_classes=NUM_CLASSES,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_ell=args.max_ell,
        irreps_channels=args.irreps_channels,
        mace_correlation=args.mace_correlation,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_encoder=args.freeze_encoder,
        so3=args.so3,
        test_metric_names=test_metric_names,
    )

    logger = build_logger(args, run_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir / "ckpts"),
        monitor="val_acc",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="epoch={epoch:02d}-val_acc={val_acc:.4f}",
        auto_insert_metric_name=False,
    )
    callbacks: List[Callback] = [
        checkpoint_callback,
    ]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    epoch_tracker = EpochTrackerCallback()
    callbacks.append(epoch_tracker)
    if not args.disable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_acc",
                mode="max",
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
            )
        )

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        max_epochs=args.epochs,
        devices=args.devices,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
    )

    if is_global_zero_process():
        print_supervised_run_header(
            args,
            test_metric_names,
            run_dir,
            num_classes=len(SO3_CLASSES) if args.so3 else NUM_CLASSES,
        )

    trainer.fit(model, dm, ckpt_path=args.ckpt_path)

    if is_global_zero_process():
        print("\n" + "=" * 60)
        print("Testing best model on clean + noisy test suites...")
        print("=" * 60)
    best_ckpt = checkpoint_callback.best_model_path or None
    test_results = trainer.test(model=model, datamodule=dm, ckpt_path=best_ckpt)
    merged_test_metrics = merge_test_results(test_results)
    retention_metrics = add_retention_metrics(merged_test_metrics)
    merged_test_metrics.update(retention_metrics)
    normalized_ckpt_row = None

    if is_global_zero_process() and args.export_best_ckpt:
        if not best_ckpt:
            raise RuntimeError("No best checkpoint available for normalized export.")
        normalized_ckpt_row = export_normalized_toy_checkpoint(
            best_ckpt_path=best_ckpt,
            run_dir=run_dir,
            model=args.model,
            stage="supervised",
            max_ell=args.max_ell if args.model in ELL_MODELS else None,
            dest_dir=args.export_dest_dir,
            update_manifest=not args.skip_manifest_update,
        )

    if is_global_zero_process():
        summary = save_supervised_summary(
            args,
            run_dir,
            checkpoint_callback,
            epoch_tracker,
            merged_test_metrics,
            normalized_ckpt_row=normalized_ckpt_row,
        )
        if logger and hasattr(logger, "log_metrics"):
            final_logger_metrics = build_final_logger_metrics(summary)
            final_logger_metrics.update(retention_metrics)
            logger.log_metrics(final_logger_metrics, step=trainer.global_step)
        print_supervised_test_summary(summary)


if __name__ == "__main__":
    main()
