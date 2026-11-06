from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from utils.common_io import write_json, write_key_value_csv


def add_retention_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    clean_acc = metrics.get("test_clean_acc")
    if clean_acc is None or clean_acc == 0:
        return {}

    retention = {}
    for key, value in metrics.items():
        if not key.startswith("test_noise_") or not key.endswith("_acc"):
            continue
        metric_name = key.replace("test_", "").replace("_acc", "")
        retention[f"retention_{metric_name}"] = value / clean_acc
    return retention


def build_final_logger_metrics(summary: Dict[str, object]) -> Dict[str, float]:
    metrics = {
        "max_epochs": float(summary["max_epochs"]),
        "completed_epochs": float(summary["completed_epochs"]),
        "stopped_early": float(int(summary["stopped_early"])),
    }
    if summary["best_val_acc"] is not None:
        metrics["best_val_acc"] = float(summary["best_val_acc"])
    return metrics


def save_supervised_summary(
    args,
    run_dir: Path,
    checkpoint_callback,
    epoch_tracker,
    test_metrics: Dict[str, float],
    normalized_ckpt_row: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_score = None
    if checkpoint_callback.best_model_score is not None:
        best_score = float(checkpoint_callback.best_model_score.detach().cpu().item())

    summary = {
        "model": args.model,
        "seed": args.seed,
        "max_epochs": args.epochs,
        "completed_epochs": epoch_tracker.last_completed_epoch,
        "stopped_early": epoch_tracker.last_completed_epoch < args.epochs,
        "max_ell": args.max_ell,
        "irreps_channels": args.irreps_channels,
        "mace_correlation": args.mace_correlation,
        "best_model_path": checkpoint_callback.best_model_path,
        "best_val_acc": best_score,
        "normalized_checkpoint_filename": (
            None if normalized_ckpt_row is None else normalized_ckpt_row["filename"]
        ),
        "normalized_checkpoint_source_ckpt": (
            None if normalized_ckpt_row is None else normalized_ckpt_row["source_ckpt"]
        ),
        "test_metrics": test_metrics,
    }

    write_json(results_dir / "summary.json", summary, indent=2, sort_keys=True)
    write_key_value_csv(
        results_dir / "summary.csv",
        [
            ("model", args.model),
            ("seed", args.seed),
            ("max_epochs", args.epochs),
            ("completed_epochs", epoch_tracker.last_completed_epoch),
            ("stopped_early", int(epoch_tracker.last_completed_epoch < args.epochs)),
            ("max_ell", args.max_ell),
            ("irreps_channels", args.irreps_channels),
            ("mace_correlation", args.mace_correlation),
            ("best_model_path", checkpoint_callback.best_model_path),
            ("best_val_acc", best_score if best_score is not None else ""),
            (
                "normalized_checkpoint_filename",
                "" if normalized_ckpt_row is None else normalized_ckpt_row["filename"],
            ),
            (
                "normalized_checkpoint_source_ckpt",
                "" if normalized_ckpt_row is None else normalized_ckpt_row["source_ckpt"],
            ),
            *[(key, value) for key, value in sorted(test_metrics.items())],
        ],
    )
    return summary


def print_supervised_run_header(
    args,
    test_metric_names: Sequence[str],
    run_dir: Path,
    *,
    num_classes: int,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Training {args.model.upper()} for point group classification")
    print(f"{'=' * 60}")
    print(f"Run dir: {run_dir}")
    print(f"Model: {args.model}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    if args.model in (
        "equiformer",
        "equiformer_balance",
        "hegnn",
        "hegnn_balance",
        "tfn",
        "tfn_balance",
        "mace",
        "mace_balance",
    ):
        print(f"Max ell: {args.max_ell}")
    if args.model in (
        "equiformer",
        "equiformer_balance",
        "hegnn",
        "hegnn_balance",
        "gvp",
        "tfn",
        "tfn_balance",
        "mace",
        "mace_balance",
    ):
        print(f"Irreps channels: {args.irreps_channels}")
    if args.model in ("mace", "mace_balance"):
        print(f"MACE correlation: {args.mace_correlation}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Devices: {args.devices}")
    print(f"Strategy: {args.strategy}")
    print(f"Num classes: {num_classes}")
    print(f"Test suites: {', '.join(test_metric_names)}")
    print(f"Seed: {args.seed}")
    print(f"{'=' * 60}\n")


def print_supervised_test_summary(summary: Dict[str, object]) -> None:
    print("\n" + "=" * 60)
    print("Run Summary")
    print("=" * 60)
    print(f"Model: {summary['model']}")
    print(
        f"Completed epochs: {summary['completed_epochs']} / {summary['max_epochs']} "
        f"(stopped_early={summary['stopped_early']})"
    )
    print(f"Best val acc: {summary['best_val_acc']}")
    print(f"Best checkpoint: {summary['best_model_path']}")
    if summary.get("normalized_checkpoint_filename"):
        print(f"Normalized checkpoint: {summary['normalized_checkpoint_filename']}")
    for key, value in sorted(summary["test_metrics"].items()):
        print(f"{key}: {value:.6f}")
    print("=" * 60)


def save_denoise_summary(
    args,
    run_dir: Path,
    checkpoint_callback,
    epoch_tracker,
    test_metrics: Dict[str, float],
    normalized_ckpt_row: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    best_score = None
    if checkpoint_callback.best_model_score is not None:
        best_score = float(checkpoint_callback.best_model_score.detach().cpu().item())

    summary = {
        "model": args.model,
        "seed": args.seed,
        "max_epochs": args.epochs,
        "completed_epochs": epoch_tracker.last_completed_epoch,
        "stopped_early": epoch_tracker.last_completed_epoch < args.epochs,
        "max_ell": args.max_ell,
        "irreps_channels": args.irreps_channels,
        "mace_correlation": args.mace_correlation,
        "best_model_path": checkpoint_callback.best_model_path,
        "best_val_pos_mae": best_score,
        "train_noise_min": args.train_noise_min,
        "train_noise_max": args.train_noise_max,
        "val_noise_alpha": args.val_noise_alpha,
        "test_noise_levels": args.test_noise_levels,
        "normalized_checkpoint_filename": (
            None if normalized_ckpt_row is None else normalized_ckpt_row["filename"]
        ),
        "normalized_checkpoint_source_ckpt": (
            None if normalized_ckpt_row is None else normalized_ckpt_row["source_ckpt"]
        ),
        "test_metrics": test_metrics,
    }

    write_json(results_dir / "summary.json", summary, indent=2, sort_keys=True)
    write_key_value_csv(
        results_dir / "summary.csv",
        [
            ("model", args.model),
            ("seed", args.seed),
            ("max_epochs", args.epochs),
            ("completed_epochs", epoch_tracker.last_completed_epoch),
            ("stopped_early", int(epoch_tracker.last_completed_epoch < args.epochs)),
            ("max_ell", args.max_ell),
            ("irreps_channels", args.irreps_channels),
            ("mace_correlation", args.mace_correlation),
            ("best_model_path", checkpoint_callback.best_model_path),
            ("best_val_pos_mae", best_score if best_score is not None else ""),
            ("train_noise_min", args.train_noise_min),
            ("train_noise_max", args.train_noise_max),
            ("val_noise_alpha", args.val_noise_alpha),
            ("test_noise_levels", args.test_noise_levels),
            (
                "normalized_checkpoint_filename",
                "" if normalized_ckpt_row is None else normalized_ckpt_row["filename"],
            ),
            (
                "normalized_checkpoint_source_ckpt",
                "" if normalized_ckpt_row is None else normalized_ckpt_row["source_ckpt"],
            ),
            *[(key, value) for key, value in sorted(test_metrics.items())],
        ],
    )
    return summary


def print_denoise_run_header(
    args,
    test_metric_names: Sequence[str],
    run_dir: Path,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Toy denoising pretraining: {args.model.upper()}")
    print(f"{'=' * 60}")
    print(f"Run dir: {run_dir}")
    print(f"Model: {args.model}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Num layers: {args.num_layers}")
    if args.model in {"equiformer", "equiformer_balance", "hegnn", "hegnn_balance", "tfn", "tfn_balance", "mace", "mace_balance"}:
        print(f"Max ell: {args.max_ell}")
    if args.model in {"equiformer", "equiformer_balance", "hegnn", "hegnn_balance", "tfn", "tfn_balance", "mace", "mace_balance"}:
        print(f"Irreps channels: {args.irreps_channels}")
    if args.model in {"mace", "mace_balance"}:
        print(f"MACE correlation: {args.mace_correlation}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Train noise range: [{args.train_noise_min:.3f}, {args.train_noise_max:.3f}]")
    print(f"Val noise alpha: {args.val_noise_alpha:.3f}")
    print(f"Test suites: {', '.join(test_metric_names)}")
    print(f"Seed: {args.seed}")
    print(f"{'=' * 60}\n")


def print_denoise_test_summary(summary: Dict[str, object]) -> None:
    print("\n" + "=" * 60)
    print("Run Summary")
    print("=" * 60)
    print(f"Model: {summary['model']}")
    print(
        f"Completed epochs: {summary['completed_epochs']} / {summary['max_epochs']} "
        f"(stopped_early={summary['stopped_early']})"
    )
    print(f"Best val pos mae: {summary['best_val_pos_mae']}")
    print(f"Best checkpoint: {summary['best_model_path']}")
    if summary.get("normalized_checkpoint_filename"):
        print(f"Normalized checkpoint: {summary['normalized_checkpoint_filename']}")
    for key, value in sorted(summary["test_metrics"].items()):
        print(f"{key}: {value:.6f}")
    print("=" * 60)


__all__ = [
    "add_retention_metrics",
    "build_final_logger_metrics",
    "print_denoise_run_header",
    "print_denoise_test_summary",
    "print_supervised_run_header",
    "print_supervised_test_summary",
    "save_denoise_summary",
    "save_supervised_summary",
]

