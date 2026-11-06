from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


SWANLAB_API_KEY = (
    os.environ.get("SWANLAB_API_KEY")
    or os.environ.get("SWANLAB_KEY")
    or os.environ.get("SWANLAB_TOKEN")
)


def parse_trainer_limit(text: str) -> int | float:
    value = float(text)
    if value <= 0:
        raise ValueError(f"Trainer limit must be positive, got {text}.")
    if value >= 1 and abs(value - round(value)) < 1e-8:
        return int(round(value))
    return value


def is_global_zero_process() -> bool:
    for env_name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(env_name)
        if value is not None:
            return int(value) == 0
    return True


@lru_cache(maxsize=1)
def resolve_swanlab_logger_class():
    if not is_global_zero_process():
        return None

    try:
        import swanlab

        try:
            swanlab.finish()
        except Exception:
            pass

        if SWANLAB_API_KEY:
            swanlab.login(api_key=SWANLAB_API_KEY, save=False)

        from swanlab.integration.pytorch_lightning import SwanLabLogger

        return SwanLabLogger
    except Exception:
        return None


class EpochTrackerCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.last_completed_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        del pl_module
        self.last_completed_epoch = trainer.current_epoch + 1


def build_default_run_name(*, model: str, task: str, seed: int) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{model}-{task}-seed{seed}-{timestamp}"


def build_run_dir(
    *,
    work_dir: str | Path,
    run_name: str | None,
    fallback_name_factory: Callable[[], str],
) -> Path:
    resolved_run_name = str(run_name) if run_name is not None else str(fallback_name_factory())
    run_dir = Path(work_dir) / resolved_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_logger(args, run_dir: Path):
    if not is_global_zero_process():
        return False

    run_name = getattr(args, "run_name", None) or run_dir.name
    logger_name = getattr(args, "logger", "tensorboard")

    if logger_name == "none":
        return False
    if logger_name == "swanlab":
        SwanLabLogger = resolve_swanlab_logger_class()
        if SwanLabLogger is None:
            raise RuntimeError("SwanLabLogger is not available in the current environment.")
        if not SWANLAB_API_KEY:
            raise RuntimeError("SwanLab logger requires SWANLAB_API_KEY in non-interactive environments.")
        return SwanLabLogger(
            project=getattr(args, "swanlab_project"),
            experiment_name=getattr(args, "swanlab_run_name", None) or run_name,
            config=vars(args),
        )
    if logger_name == "wandb":
        return WandbLogger(
            project=getattr(args, "project"),
            name=run_name,
            save_dir=str(run_dir),
            log_model=True,
        )
    return TensorBoardLogger(
        save_dir=str(run_dir),
        name="tensorboard",
        version=None,
    )


def merge_test_results(test_results: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for result in test_results:
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                merged[key] = float(value.detach().cpu().item())
            else:
                merged[key] = float(value)
    return merged


__all__ = [
    "EpochTrackerCallback",
    "SWANLAB_API_KEY",
    "build_default_run_name",
    "build_logger",
    "build_run_dir",
    "is_global_zero_process",
    "merge_test_results",
    "parse_trainer_limit",
    "resolve_swanlab_logger_class",
]

