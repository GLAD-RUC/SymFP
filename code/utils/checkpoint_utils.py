from __future__ import annotations

import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELL_MODELS = {
    "equiformer",
    "equiformer_balance",
    "hegnn",
    "hegnn_balance",
    "tfn",
    "tfn_balance",
    "mace",
    "mace_balance",
}
STAGE_TO_MANIFEST = {
    "supervised": "manifest.csv",
    "pretrained": "pretrained_manifest.csv",
}


def normalized_toy_checkpoint_name(
    model: str,
    stage: str,
    *,
    max_ell: int | None = None,
) -> str:
    model = str(model)
    stage = str(stage)
    if stage not in STAGE_TO_MANIFEST:
        raise ValueError(f"Unsupported checkpoint stage: {stage}")

    if model in ELL_MODELS:
        if max_ell is None:
            raise ValueError(f"Missing max_ell for model {model}.")
        return f"{model}_ell{int(max_ell)}_{stage}.ckpt"
    return f"{model}_{stage}.ckpt"


def make_relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def load_manifest_rows(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.is_file():
        return {}

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row for row in reader}


def write_manifest(manifest_path: Path, rows: dict[str, dict[str, str]]) -> None:
    fieldnames = ["filename", "source_run_dir", "source_ckpt"]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for filename in sorted(rows):
            writer.writerow(rows[filename])


def export_normalized_toy_checkpoint(
    *,
    best_ckpt_path: str | Path,
    run_dir: str | Path,
    model: str,
    stage: str,
    max_ell: int | None = None,
    dest_dir: str | Path,
    update_manifest: bool = True,
) -> dict[str, str]:
    best_ckpt = Path(best_ckpt_path).resolve()
    run_dir_path = Path(run_dir).resolve()
    dest_dir_path = Path(dest_dir).resolve()
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    filename = normalized_toy_checkpoint_name(
        model=model,
        stage=stage,
        max_ell=max_ell,
    )
    dest_path = dest_dir_path / filename
    shutil.copy2(best_ckpt, dest_path)

    row = {
        "filename": filename,
        "source_run_dir": make_relpath(run_dir_path),
        "source_ckpt": make_relpath(best_ckpt),
    }

    if update_manifest:
        manifest_name = STAGE_TO_MANIFEST[stage]
        manifest_path = dest_dir_path / manifest_name
        rows = load_manifest_rows(manifest_path)
        rows[filename] = row
        write_manifest(manifest_path, rows)

    return row


__all__ = [
    "ELL_MODELS",
    "PROJECT_ROOT",
    "STAGE_TO_MANIFEST",
    "export_normalized_toy_checkpoint",
    "load_manifest_rows",
    "make_relpath",
    "normalized_toy_checkpoint_name",
    "write_manifest",
]

