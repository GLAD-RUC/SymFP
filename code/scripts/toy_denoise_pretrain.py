"""
ToyDataset denoising pretraining for equivariant backbones.

This script corrupts finite point clouds with Gaussian coordinate noise and
trains a backbone to reconstruct clean Cartesian coordinates. The backbone is
expected to return a dict with at least:

    - node_pos: predicted clean coordinates
    - graph_feat / graph_sh: optional graph-level outputs

Supported backbones:
    - equiformer
    - equiformer_balance
    - hegnn
    - hegnn_balance
    - tfn
    - tfn_balance
    - mace
    - mace_balance
"""

import argparse
import os
import sys
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch_geometric.loader import DataLoader as PyGDataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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
from utils.toy_backbone_utils import build_denoise_backbone
from utils.toy_runtime_utils import (
    add_scaled_gaussian_noise,
    build_distance_edge_attr,
    build_node_features,
    build_test_metric_names,
    center_positions,
    parse_noise_levels,
)
from utils.toy_training_reporting import (
    print_denoise_run_header,
    print_denoise_test_summary,
    save_denoise_summary,
)

DEFAULT_TEST_NOISE_LEVELS = "0.0,0.05,0.10"


def choose_train_noise_alpha(min_alpha: float, max_alpha: float) -> float:
    if max_alpha < min_alpha:
        raise ValueError(
            f"Expected max_alpha >= min_alpha, got {max_alpha} < {min_alpha}."
        )
    if max_alpha == min_alpha:
        return float(min_alpha)
    return float(
        torch.empty((), dtype=torch.float32).uniform_(min_alpha, max_alpha).item()
    )


def apply_denoise_transform(
    data,
    *,
    split: str,
    deterministic_noise: bool,
    noise_seed: int,
    model_type: str,
    noise_alpha: Optional[float] = None,
    noise_alpha_sampler: Optional[Callable[[], float]] = None,
):
    resolved_noise_alpha = noise_alpha
    if resolved_noise_alpha is None:
        if noise_alpha_sampler is None:
            raise ValueError(
                "Either noise_alpha or noise_alpha_sampler must be provided."
            )
        resolved_noise_alpha = float(noise_alpha_sampler())

    data = data.clone()
    clean_pos = center_positions(data.pos)
    noisy_pos, noise_scale = add_scaled_gaussian_noise(
        clean_pos,
        sample_key=str(getattr(data, "mid", "")),
        split=split,
        noise_alpha=resolved_noise_alpha,
        noise_seed=noise_seed,
        deterministic_noise=deterministic_noise,
        hash_method="weighted_utf8_sum",
        center=True,
    )

    atomic_numbers = data.x.clone().to(dtype=torch.long)
    data.x = build_node_features(atomic_numbers, model_type)
    data.clean_pos = clean_pos
    data.target_delta = clean_pos - noisy_pos
    data.pos = noisy_pos
    data.edge_attr = build_distance_edge_attr(data.pos, data.edge_index)
    data.noise_alpha = float(resolved_noise_alpha)
    data.noise_scale = float(noise_scale.item())
    return data


class ToyDenoisePretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        pkl_path: str,
        processed_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        model_type: str = "hegnn_balance",
        train_noise_min: float = 0.02,
        train_noise_max: float = 0.10,
        val_noise_alpha: float = 0.05,
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
        self.train_noise_min = train_noise_min
        self.train_noise_max = train_noise_max
        self.val_noise_alpha = val_noise_alpha
        self.test_noise_levels = list(test_noise_levels or [0.0, val_noise_alpha])
        self.noise_seed = noise_seed
        self.pin_memory = pin_memory
        self.test_loader_names = build_test_metric_names(self.test_noise_levels)

        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets: List[ToyDataset] = []

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = ToyDataset(
                root=self.processed_root,
                pkl_path=self.pkl_path,
                split="train",
                transform=self._create_train_transform(),
            )
            self.val_dataset = ToyDataset(
                root=self.processed_root,
                pkl_path=self.pkl_path,
                split="val",
                transform=self._create_eval_transform(
                    split="val",
                    noise_alpha=self.val_noise_alpha,
                ),
            )

        if stage in (None, "test"):
            self.test_datasets = [
                ToyDataset(
                    root=self.processed_root,
                    pkl_path=self.pkl_path,
                    split="test",
                    transform=self._create_eval_transform(
                        split="test",
                        noise_alpha=noise_alpha,
                    ),
                )
                for noise_alpha in self.test_noise_levels
            ]

    def _create_train_transform(self):
        return partial(
            apply_denoise_transform,
            split="train",
            deterministic_noise=False,
            noise_seed=self.noise_seed,
            model_type=self.model_type,
            noise_alpha_sampler=partial(
                choose_train_noise_alpha,
                self.train_noise_min,
                self.train_noise_max,
            ),
        )

    def _create_eval_transform(self, split: str, noise_alpha: float):
        return partial(
            apply_denoise_transform,
            split=split,
            deterministic_noise=True,
            noise_seed=self.noise_seed,
            model_type=self.model_type,
            noise_alpha=noise_alpha,
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


class ToyDenoisePretrainer(pl.LightningModule):
    def __init__(
        self,
        model_type: str = "hegnn_balance",
        hidden_dim: int = 64,
        num_layers: int = 4,
        max_ell: int = 6,
        irreps_channels: int = 16,
        mace_correlation: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        edge_loss_weight: float = 0.1,
        test_metric_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["test_metric_names"])
        self.test_metric_names = list(test_metric_names or ["clean"])
        self.backbone = build_denoise_backbone(
            model_type=model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_ell=max_ell,
            irreps_channels=irreps_channels,
            mace_correlation=mace_correlation,
        )

    def forward(self, batch):
        return self.backbone(batch)

    def _compute_metrics(self, batch):
        outputs = self.backbone(batch)
        pred_pos = outputs["node_pos"]
        clean_pos = batch.clean_pos
        target_delta = batch.target_delta

        row, col = batch.edge_index
        pred_dist = torch.linalg.norm(pred_pos[col] - pred_pos[row], dim=-1)
        clean_dist = torch.linalg.norm(clean_pos[col] - clean_pos[row], dim=-1)
        pred_delta = pred_pos - batch.pos

        pos_loss = F.smooth_l1_loss(pred_pos, clean_pos)
        edge_loss = F.smooth_l1_loss(pred_dist, clean_dist)
        loss = pos_loss + self.hparams.edge_loss_weight * edge_loss

        pos_mae = F.l1_loss(pred_pos, clean_pos)
        delta_mae = F.l1_loss(pred_delta, target_delta)
        edge_mae = F.l1_loss(pred_dist, clean_dist)
        pos_rmse = torch.sqrt(F.mse_loss(pred_pos, clean_pos))

        return {
            "loss": loss,
            "pos_loss": pos_loss,
            "edge_loss": edge_loss,
            "pos_mae": pos_mae,
            "delta_mae": delta_mae,
            "edge_mae": edge_mae,
            "pos_rmse": pos_rmse,
            "batch_size": batch.num_graphs,
        }

    def training_step(self, batch, batch_idx):
        del batch_idx
        metrics = self._compute_metrics(batch)
        self.log(
            "train_loss",
            metrics["loss"],
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=metrics["batch_size"],
        )
        self.log(
            "train_pos_mae",
            metrics["pos_mae"],
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=metrics["batch_size"],
        )
        self.log(
            "train_edge_mae",
            metrics["edge_mae"],
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=metrics["batch_size"],
        )
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        del batch_idx
        metrics = self._compute_metrics(batch)
        self.log(
            "val_loss",
            metrics["loss"],
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=metrics["batch_size"],
        )
        self.log(
            "val_pos_mae",
            metrics["pos_mae"],
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=metrics["batch_size"],
        )
        self.log(
            "val_delta_mae",
            metrics["delta_mae"],
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=metrics["batch_size"],
        )
        self.log(
            "val_edge_mae",
            metrics["edge_mae"],
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=metrics["batch_size"],
        )
        self.log(
            "val_pos_rmse",
            metrics["pos_rmse"],
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=metrics["batch_size"],
        )
        return metrics["loss"]

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        del batch_idx
        metric_name = self.test_metric_names[dataloader_idx]
        metrics = self._compute_metrics(batch)
        self.log(
            f"test_{metric_name}_loss",
            metrics["loss"],
            prog_bar=dataloader_idx == 0,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=metrics["batch_size"],
        )
        self.log(
            f"test_{metric_name}_pos_mae",
            metrics["pos_mae"],
            prog_bar=dataloader_idx == 0,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=metrics["batch_size"],
        )
        self.log(
            f"test_{metric_name}_delta_mae",
            metrics["delta_mae"],
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=metrics["batch_size"],
        )
        self.log(
            f"test_{metric_name}_edge_mae",
            metrics["edge_mae"],
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=metrics["batch_size"],
        )
        self.log(
            f"test_{metric_name}_pos_rmse",
            metrics["pos_rmse"],
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
            batch_size=metrics["batch_size"],
        )
        return {
            f"test_{metric_name}_loss": metrics["loss"].detach(),
            f"test_{metric_name}_pos_mae": metrics["pos_mae"].detach(),
            f"test_{metric_name}_delta_mae": metrics["delta_mae"].detach(),
            f"test_{metric_name}_edge_mae": metrics["edge_mae"].detach(),
            f"test_{metric_name}_pos_rmse": metrics["pos_rmse"].detach(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
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
        description="Toy denoising pretraining with equivariant backbones"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="hegnn",
        choices=[
            "equiformer",
            "equiformer_balance",
            "hegnn",
            "hegnn_balance",
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
        help="Default: 6 for all supported backbones.",
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
    parser.add_argument("--train_noise_min", type=float, default=0.02)
    parser.add_argument("--train_noise_max", type=float, default=0.10)
    parser.add_argument("--val_noise_alpha", type=float, default=0.05)
    parser.add_argument(
        "--test_noise_levels", type=str, default=DEFAULT_TEST_NOISE_LEVELS
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--edge_loss_weight", type=float, default=0.1)

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
    parser.add_argument("--project", type=str, default="toy-denoise-pretrain")
    parser.add_argument("--swanlab_project", type=str, default="toy-denoise-pretrain")
    parser.add_argument("--swanlab_run_name", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default="outputs/toy_denoise_pretrain")
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
        help="Do not update ckpts/toydataset pretrained_manifest.csv during normalized export.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
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
            task="toy-denoise-pretrain",
            seed=args.seed,
        ),
    )

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed, workers=True)

    dm = ToyDenoisePretrainDataModule(
        pkl_path=args.pkl_path,
        processed_root=args.processed_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_type=args.model,
        train_noise_min=args.train_noise_min,
        train_noise_max=args.train_noise_max,
        val_noise_alpha=args.val_noise_alpha,
        test_noise_levels=test_noise_levels,
        noise_seed=args.seed,
        pin_memory=args.pin_memory,
    )

    model = ToyDenoisePretrainer(
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_ell=args.max_ell,
        irreps_channels=args.irreps_channels,
        mace_correlation=args.mace_correlation,
        lr=args.lr,
        weight_decay=args.weight_decay,
        edge_loss_weight=args.edge_loss_weight,
        test_metric_names=test_metric_names,
    )

    logger = build_logger(args, run_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir / "ckpts"),
        monitor="val_pos_mae",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="epoch={epoch:02d}-val_pos_mae={val_pos_mae:.6f}",
        auto_insert_metric_name=False,
    )

    callbacks: List[Callback] = [checkpoint_callback]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    epoch_tracker = EpochTrackerCallback()
    callbacks.append(epoch_tracker)
    if not args.disable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_pos_mae",
                mode="min",
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
        print_denoise_run_header(args, test_metric_names, run_dir)

    trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)

    if is_global_zero_process():
        print("\n" + "=" * 60)
        print("Testing best model on clean + noisy denoising suites...")
        print("=" * 60)
    best_ckpt = checkpoint_callback.best_model_path or None
    test_results = trainer.test(model=model, datamodule=dm, ckpt_path=best_ckpt)
    merged_test_metrics = merge_test_results(test_results)
    normalized_ckpt_row = None
    if is_global_zero_process() and args.export_best_ckpt:
        if not best_ckpt:
            raise RuntimeError("No best checkpoint available for normalized export.")
        normalized_ckpt_row = export_normalized_toy_checkpoint(
            best_ckpt_path=best_ckpt,
            run_dir=run_dir,
            model=args.model,
            stage="pretrained",
            max_ell=args.max_ell if args.model in ELL_MODELS else None,
            dest_dir=args.export_dest_dir,
            update_manifest=not args.skip_manifest_update,
        )

    if is_global_zero_process():
        summary = save_denoise_summary(
            args=args,
            run_dir=run_dir,
            checkpoint_callback=checkpoint_callback,
            epoch_tracker=epoch_tracker,
            test_metrics=merged_test_metrics,
            normalized_ckpt_row=normalized_ckpt_row,
        )
        print_denoise_test_summary(summary)


if __name__ == "__main__":
    main()
