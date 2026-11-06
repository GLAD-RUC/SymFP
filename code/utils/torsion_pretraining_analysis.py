"""Portable analysis of paired torsion pLS scores; no training or plotting dependencies."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from utils.torsion_data import prepare_data, save_protocol

MODELS = ("hegnn", "tfn", "mace", "equiformer")
GROUPS = ("Cs", "C2", "C2v", "D2", "D2d", "D2h", "Ci", "C2h", "S4")
HM = {
    "Cs": "m",
    "C2": "2",
    "C2v": "mm2",
    "D2": "222",
    "D2d": "-42m",
    "D2h": "mmm",
    "Ci": "-1",
    "C2h": "2/m",
    "S4": "-4",
}
CONTAINED_IN = {
    "Ci": {"Ci", "C2h", "D2h"},
    "Cs": {"Cs", "C2v", "C2h", "D2h", "D2d"},
    "C2": {"C2", "C2v", "C2h", "D2", "S4", "D2d", "D2h"},
    "C2h": {"C2h", "D2h"},
    "C2v": {"C2v", "D2h", "D2d"},
    "D2": {"D2", "D2h", "D2d"},
    "S4": {"S4", "D2d"},
    "D2d": {"D2d"},
    "D2h": {"D2h"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes repository: {relative}")
    return path


def load_manifest(bundle, verify=True):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "configs/torsion_pretraining.json").read_text())
    if manifest["format_version"] != 2:
        raise ValueError("Unsupported manifest version")
    if verify:
        for relative, expected in manifest["files"].items():
            if sha(contained_path(bundle, relative)) != expected:
                raise ValueError(f"Checksum mismatch: {relative}")
    return manifest


def labels(energy):
    left = energy - np.roll(energy, 1)
    right = energy - np.roll(energy, -1)
    maximum = (left > 0) & (right > 0)
    minimum = (left < 0) & (right < 0)
    return dict(maximum=maximum, minimum=minimum, all=maximum | minimum)


def recall_curve(score, positive):
    # Average ranks within exact ties yield expected fractional recall at every budget.
    order = np.argsort(-score, kind="stable")
    v = score[order]
    y = np.asarray(positive, float)[order]
    if not y.sum():
        return np.full(len(v) + 1, np.nan)
    expected = np.zeros(len(v))
    a = 0
    while a < len(v):
        b = a + 1
        while b < len(v) and v[b] == v[a]:
            b += 1
        expected[a:b] = y[a:b].mean()
        a = b
    return np.r_[0, np.cumsum(expected)] / y.sum()


def change_stats(values, draws=5000, seed=20261005, tolerance=1e-10):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, mean=None, ci95=None, increase=0, unchanged=0, decrease=0)
    ix = np.random.default_rng(seed).integers(0, len(v), (draws, len(v)))
    return dict(
        n=len(v),
        mean=float(v.mean()),
        ci95=np.quantile(v[ix].mean(1), [0.025, 0.975]).tolist(),
        increase=int((v > tolerance).sum()),
        unchanged=int((abs(v) <= tolerance).sum()),
        decrease=int((v < -tolerance).sum()),
    )


def read_score(path, expected):
    frame = (
        pd.read_csv(path, float_precision="round_trip")
        .set_index(["record_id", "frame_index"])
        .sort_index()
    )
    if not frame.index.is_unique or not frame.index.equals(expected):
        raise ValueError(f"Score cohort mismatch: {path}")
    if not np.isfinite(frame[[f"pls_even_{g}" for g in GROUPS]]).all().all():
        raise ValueError(f"Nonfinite scores: {path}")
    return frame


def analyze(bundle, out, scores_dir=None):
    if scores_dir is None:
        raise ValueError("scores_dir is required")
    bundle = Path(bundle).resolve()
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    manifest = load_manifest(bundle)
    out.mkdir(parents=True, exist_ok=True)
    frames, pairs, split, normalization = prepare_data(
        bundle / manifest["dataset"], manifest["seed"]
    )
    save_protocol(out, split, normalization)
    assert len(frames) == 2064 and len(pairs) == 94
    expected = pd.MultiIndex.from_frame(
        frames[["record_id", "frame_index"]]
    ).sort_values()
    entries = {(e["variant"], e["model"]): e for e in manifest["entries"]}
    scores = Path(scores_dir).resolve()
    tables = {"energy_extrema": [], "target_symmetry": [], "special_angles": []}
    results = {}
    composition = None
    for variant in ["energy", "unimol"]:
        for model in MODELS:
            before = read_score(scores / "random" / f"{model}.csv", expected)
            after = read_score(scores / variant / f"{model}.csv", expected)
            rows = []
            pair_rows = []
            for pair in pairs.itertuples():
                f = frames[frames.record_id == pair.record_id].sort_values("angle_deg")
                assert len(f) == 24 and np.allclose(np.diff(f.angle_deg), 15)
                subset = (
                    "train"
                    if pair.record_id in split["train_record_ids"]
                    else "validation"
                )
                assert (f.split == subset).all()
                ix = pd.MultiIndex.from_frame(f[["record_id", "frame_index"]])
                x = before.loc[ix, f"pls_even_{pair.target}"].to_numpy()
                y = after.loc[ix, f"pls_even_{pair.target}"].to_numpy()
                lab = labels(f.relative_energy_kcal_mol.to_numpy())
                present = f["pg_tol0.1"].isin(CONTAINED_IN[pair.target]).to_numpy()
                special = np.isclose(f.angle_deg.to_numpy() % 90, 0, atol=1e-8)
                delta = y - x
                retention = delta - (np.roll(delta, 1) + np.roll(delta, -1)) / 2
                record = dict(
                    record_id=pair.record_id, target=pair.target, subset=subset
                )
                for when, score in [("before", x), ("after", y)]:
                    masks = dict(
                        peak=lab["maximum"],
                        trough=lab["minimum"],
                        present=present,
                        absent=~present,
                        special=special,
                        other=~special,
                    )
                    for label, mask in masks.items():
                        record[f"{label}_{when}"] = (
                            float(score[mask].mean()) if mask.any() else np.nan
                        )
                    record[f"gap_{when}"] = (
                        record[f"peak_{when}"] - record[f"trough_{when}"]
                    )
                    record[f"contrast_{when}"] = (
                        record[f"present_{when}"] - record[f"absent_{when}"]
                    )
                    record[f"angle_gap_{when}"] = (
                        record[f"special_{when}"] - record[f"other_{when}"]
                    )
                    for kind in ["all", "maximum", "minimum"]:
                        curve = recall_curve(score, lab[kind])
                        area = float(np.trapz(curve, np.arange(25) / 24))
                        assert (
                            abs(
                                area
                                - (1 - (rankdata(-score)[lab[kind]].mean() - 0.5) / 24)
                            )
                            < 1e-12
                        )
                        record[f"aurc_{kind}_{when}"] = area
                        record[f"recall25_{kind}_{when}"] = float(curve[6])
                    record[f"structural_auc_{when}"] = (
                        roc_auc_score(present, score)
                        if present.any() and (~present).any()
                        else np.nan
                    )
                for key in list(record):
                    if key.endswith("_before"):
                        metric = key.removesuffix("_before")
                        record[f"delta_{metric}"] = (
                            record[f"{metric}_after"] - record[key]
                        )
                pair_rows.append(record)
                for j, frame in enumerate(f.itertuples()):
                    a = frame.angle_deg % 180
                    rows.append(
                        dict(
                            record_id=pair.record_id,
                            target=pair.target,
                            subset=subset,
                            frame_index=frame.frame_index,
                            angle_deg=frame.angle_deg,
                            folded_angle=min(a, 180 - a),
                            energy_kcal_mol=frame.relative_energy_kcal_mol,
                            is_maximum=bool(lab["maximum"][j]),
                            is_minimum=bool(lab["minimum"][j]),
                            target_present=bool(present[j]),
                            special_angle=bool(special[j]),
                            pls_before=x[j],
                            pls_after=y[j],
                            delta_pls=delta[j],
                            local_retention=retention[j],
                        )
                    )
            long = pd.DataFrame(rows)
            # Explicit lookup avoids tuple-name rewriting of dotted point-group columns.
            pg = frames.set_index(["record_id", "frame_index"])["pg_tol0.1"]
            long["point_group"] = pg.loc[
                pd.MultiIndex.from_frame(long[["record_id", "frame_index"]])
            ].to_numpy()
            pairframe = pd.DataFrame(pair_rows)
            numeric = [
                c for c in pairframe if c not in ["record_id", "target", "subset"]
            ]
            mol = pairframe.groupby(["record_id", "subset"], as_index=False)[
                numeric
            ].mean()
            key = f"{variant}_{model}"
            dest = out / key
            dest.mkdir()
            long.to_csv(dest / "all_frame_changes.csv", index=False)
            pairframe.to_csv(dest / "pair_metrics.csv", index=False)
            mol.to_csv(dest / "molecule_metrics.csv", index=False)
            summaries = {}
            for subset in ["all", "train", "validation"]:
                g = mol if subset == "all" else mol[mol.subset == subset]
                summaries[subset] = dict(
                    n=len(g),
                    means=g[numeric].mean().to_dict(),
                    changes={
                        name: change_stats(
                            g[f"delta_{name}"],
                            tolerance=(
                                1e-6 if name in ["contrast", "angle_gap"] else 1e-10
                            ),
                        )
                        for name in [
                            "gap",
                            "contrast",
                            "angle_gap",
                            "aurc_all",
                            "aurc_maximum",
                            "aurc_minimum",
                            "structural_auc",
                        ]
                    },
                )
            result = dict(epoch=entries[(variant, model)]["epoch"], subsets=summaries)
            (dest / "summary.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
            results[key] = result
            all_ = summaries["all"]
            val = summaries["validation"]
            m = all_["means"]

            def row(metric):
                return dict(
                    pretraining=variant,
                    model=model,
                    all_n=all_["n"],
                    validation_n=val["n"],
                    improved_all=all_["changes"][metric]["increase"],
                    improved_validation=val["changes"][metric]["increase"],
                    improved_all_pct=100
                    * all_["changes"][metric]["increase"]
                    / all_["n"],
                    improved_validation_pct=100
                    * val["changes"][metric]["increase"]
                    / val["n"],
                )

            tables["energy_extrema"].append(
                dict(
                    **row("gap"),
                    peak_delta=m["delta_peak"],
                    trough_delta=m["delta_trough"],
                    gap_delta=m["delta_gap"],
                    aurc_delta=m["delta_aurc_all"],
                )
            )
            tables["target_symmetry"].append(
                dict(
                    **row("contrast"),
                    present_delta=m["delta_present"],
                    absent_delta=m["delta_absent"],
                    gap_delta=m["delta_contrast"],
                )
            )
            tables["special_angles"].append(
                dict(
                    **row("angle_gap"),
                    special_delta=m["delta_special"],
                    other_delta=m["delta_other"],
                    gap_delta=m["delta_angle_gap"],
                )
            )
            # Preserve per-molecule angle and special-minimum local comparisons.
            long["energy_kind"] = np.select(
                [long.is_maximum, long.is_minimum],
                ["maximum", "minimum"],
                default="nonextremum",
            )
            for group_columns, name in [
                (["angle_deg"], "angle"),
                (["special_angle", "energy_kind"], "angle_energy_kind"),
            ]:
                grouped = long.groupby(
                    ["record_id", "target", "subset"] + group_columns, as_index=False
                )[["pls_before", "pls_after", "delta_pls", "local_retention"]].mean()
                grouped = grouped.groupby(
                    ["record_id", "subset"] + group_columns, as_index=False
                )[["pls_before", "pls_after", "delta_pls", "local_retention"]].mean()
                grouped.to_csv(dest / f"{name}_molecule_changes.csv", index=False)
            if composition is None:
                unique = long.drop_duplicates(["record_id", "frame_index"])
                composition = unique.groupby(
                    ["angle_deg", "energy_kind"], as_index=False
                ).size()
                composition.to_csv(out / "angle_energy_counts.csv", index=False)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(
            out / f"table_{name}.csv", index=False, float_format="%.10g"
        )
    report = dict(
        protocol="Raw pLS; same initialization; fixed molecule split; exact-tie expected recall; molecule equal weighting; 5000 molecule bootstrap draws. Positive gap change, not generic retrieval improvement.",
        inputs=dict(
            manifest_sha256=sha(bundle / "configs/torsion_pretraining.json"),
            scores={
                f"{v}/{m}.csv": sha(scores / v / f"{m}.csv")
                for v in ["random", "energy", "unimol"]
                for m in MODELS
            },
        ),
        results=results,
    )
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report
