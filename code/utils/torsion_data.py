"""Prepare the paired pretraining split from the shared torsion dataset."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def prepare_data(data_dir, seed=42):
    data_dir = Path(data_dir)
    pairs = pd.DataFrame(json.loads((data_dir / "evaluated_pairs.json").read_text()))
    pairs["record_id"] = pairs.record_id.astype(int)
    ids = np.sort(pairs.record_id.unique())
    if len(ids) != 86 or len(pairs) != 94 or pairs.duplicated().any():
        raise ValueError("Expected 94 unique targets across the 86-molecule cohort")
    shuffled = np.random.Generator(np.random.PCG64(seed)).permutation(ids)
    n_val = int(round(len(ids) * 0.2))
    val_ids = sorted(shuffled[:n_val].tolist())
    train_ids = sorted(shuffled[n_val:].tolist())
    # Match the decimal parsing used to prepare the original training targets.
    frames = pd.read_csv(data_dir / "responses.csv")
    frames = (
        frames[frames.record_id.isin(ids)]
        .sort_values(["record_id", "frame_index"])
        .reset_index(drop=True)
    )
    if len(frames) != 2064 or frames.duplicated(["record_id", "frame_index"]).any():
        raise ValueError("Expected 2,064 distinct conformations")
    for _, group in frames.groupby("record_id"):
        if not np.array_equal(group.frame_index.to_numpy(), np.arange(24)):
            raise ValueError("Each molecule must contain frame indices 0 through 23")
        if group.trajectory_path.nunique() != 1:
            raise ValueError("Each molecule must reference a single trajectory")
        path = (data_dir / group.trajectory_path.iloc[0]).resolve()
        if not path.is_relative_to(data_dir.resolve()) or not path.is_file():
            raise ValueError("Invalid trajectory path")
    if not np.isfinite(frames.energy_hartree).all():
        raise ValueError("Nonfinite energy")
    frames["split"] = np.where(frames.record_id.isin(train_ids), "train", "validation")
    energies = frames.loc[frames.split == "train", "energy_hartree"].to_numpy(
        dtype=np.float64
    )
    mean, std = float(energies.mean()), float(energies.std(ddof=0))
    if std <= 0:
        raise ValueError("Training energies must have nonzero variance")
    frames["energy_normalized"] = (frames.energy_hartree - mean) / std
    point_groups = pd.read_csv(data_dir / "point_groups.csv")
    frames = frames.merge(
        point_groups, on=["record_id", "frame_index"], validate="one_to_one"
    )
    if len(frames) != 2064 or frames[["pg_tol0.1", "pg_tol0.3"]].isna().any().any():
        raise ValueError("Incomplete point-group labels")
    sources = [
        data_dir / name
        for name in ("responses.csv", "evaluated_pairs.json", "point_groups.csv")
    ]
    sources += sorted(data_dir / p for p in frames.trajectory_path.unique())
    split = dict(
        seed=seed,
        rng="numpy.random.Generator(PCG64)",
        algorithm="Sort numeric record IDs; permute; first round(0.2*N) are validation; remainder train",
        split_unit="molecule/record_id, all 24 conformations kept together",
        train_record_ids=train_ids,
        validation_record_ids=val_ids,
        counts=dict(
            train_molecules=69,
            validation_molecules=17,
            train_frames=1656,
            validation_frames=408,
        ),
        source_sha256={
            str(p.relative_to(data_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )
    normalization = dict(
        target="energy_hartree",
        output="energy_normalized",
        method="z-score",
        fit_subset="training conformations only",
        mean_hartree=mean,
        std_hartree=std,
        ddof=0,
        statistics_dtype="float64",
        transform="(energy_hartree - mean_hartree) / std_hartree",
        inverse="energy_normalized * std_hartree + mean_hartree",
        loss="MAE on normalized total energy",
        mae_conversion="MAE_hartree = MAE_normalized * std_hartree",
        coordinate_normalization="none; original geometry and units preserved",
    )
    return frames, pairs, split, normalization


def save_protocol(out, split, normalization):
    """Record the generated split and statistics without copying any input data."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    contents = {
        name: json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        for name, value in [
            ("split.json", split),
            ("normalization.json", normalization),
        ]
    }
    for name, text in contents.items():
        if (out / name).exists() and (out / name).read_text() != text:
            raise FileExistsError(f"Different training protocol: {out / name}")
    for name, text in contents.items():
        (out / name).write_text(text)
