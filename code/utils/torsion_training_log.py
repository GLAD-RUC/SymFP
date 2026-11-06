"""Local JSONL training metrics; no online service or credentials required."""

import json
from pathlib import Path
from types import SimpleNamespace

_log_path = None


def init_run(out, config, *, resume=False):
    global _log_path
    path = Path(out) / "training_log.jsonl"
    if path.exists() and not resume:
        raise FileExistsError(
            "Training log exists; use --resume or a new output directory"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    _log_path = path
    return SimpleNamespace(url=str(path))


def log(values, *, step):
    if _log_path is None:
        raise RuntimeError("Initialize the local log first")
    with _log_path.open("a") as handle:
        handle.write(json.dumps(dict(step=step, **values), allow_nan=False) + "\n")


def log_epoch(
    *,
    epoch,
    train_mae,
    validation_mae,
    energy_std_hartree,
    learning_rate,
    best_validation_mae
):
    log(
        dict(
            train_mae_normalized=train_mae,
            validation_mae_normalized=validation_mae,
            train_mae_hartree=train_mae * energy_std_hartree,
            validation_mae_hartree=validation_mae * energy_std_hartree,
            learning_rate=learning_rate,
            best_validation_mae=best_validation_mae,
        ),
        step=epoch,
    )


def finish():
    global _log_path
    _log_path = None
