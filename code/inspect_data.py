"""Inspect portable datasets and verify the complete SHA-256 inventory."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    root = args.root
    if args.verify:
        manifest = json.loads((root / "files.json").read_text())
        for r in manifest:
            f = root / r["path"]
            if f.stat().st_size != r["bytes"] or sha(f) != r["sha256"]:
                raise ValueError("Checksum mismatch: " + r["path"])
        print("Verified", len(manifest), "files")
        archives = json.loads((root / "archives.json").read_text())
        for archive in archives:
            for r in archive["members"]:
                f = root / r["path"]
                if not f.is_file():
                    raise FileNotFoundError(
                        "Extract " + archive["path"] + " first: " + r["path"]
                    )
                if f.stat().st_size != r["bytes"] or sha(f) != r["sha256"]:
                    raise ValueError("Extracted-file checksum mismatch: " + r["path"])
        print("Verified all extracted archive members")
    seen = set()
    for split, n in [("train", 2560), ("val", 320), ("test", 320)]:
        with np.load(root / f"datasets/toy/{split}.npz", allow_pickle=False) as a:
            assert len(a["ids"]) == n and len(a["offsets"]) == n + 1
            assert a["offsets"][-1] == len(a["pos"]) == len(a["z"])
            assert not seen.intersection(a["ids"])
            seen.update(a["ids"])
            groups, counts = np.unique(a["point_group"], return_counts=True)
            assert len(groups) == 32 and np.all(counts == n // 32)
            print("toy", split, n, "structures;32 groups;", len(a["pos"]), "points")
    for system in ["p97", "ha", "gluk2", "e18"]:
        for f in sorted((root / "datasets" / system).glob("*.npz")):
            with np.load(f, allow_pickle=False) as a:
                for k in a.files:
                    assert a[k].dtype.kind != "O"
                print(system, f.name, len(a["z"]), "atoms")
    pairs = json.loads((root / "datasets/torsion/evaluated_pairs.json").read_text())
    assert len(pairs) == 94 and len({r["record_id"] for r in pairs}) == 86
    layers = json.loads((root / "datasets/gluk2/layers.json").read_text())["windows"]
    assert len(layers) == 44
    print("torsion:94 pairs across86 molecules; GluK2:44 layers")


if __name__ == "__main__":
    main()
