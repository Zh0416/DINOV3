"""Hash all decision artifacts before a separate process can open annotations."""
from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pvseg.label_free_data import load_rgb_manifest
from pvseg.label_free_training import variants_for_config
from pvseg.utils import atomic_write_json, read_json


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seal_predictions(config, root):
    if (root / "prediction_seal.json").exists():
        raise RuntimeError("Seal already exists; do not overwrite evaluation history")
    samples = load_rgb_manifest(root / "rgb_manifest.json")
    test_ids = {s.sample_id for s in samples if s.split == "test"}
    train_ids = {s.sample_id for s in samples if s.split == "train"}
    if not test_ids or test_ids & train_ids:
        raise RuntimeError("Train/test image content hashes must be distinct")
    variants = variants_for_config(config)
    expected = {f"{name}_seed{seed}" for name in variants for seed in config["ablation"]["seeds"]}
    training = read_json(root / "reports" / "label_free_training.json")
    if {f"{r['variant']}_seed{r['seed']}" for r in training["runs"]} != expected:
        raise RuntimeError("Missing preregistered experiment runs")
    for run in expected:
        if {p.stem for p in (root / "predictions" / run).glob("*.png")} != test_ids:
            raise RuntimeError(f"Missing or extra held-out predictions: {run}")
        if not (root / "checkpoints" / f"{run}_last.pt").is_file():
            raise RuntimeError(f"Missing checkpoint: {run}")
    paths = [root / name for name in ("rgb_manifest.json", "protocol_config.json", "semantic_prompt.json",
                                       "rgb_dino_prototypes.npy", "reports/label_free_training.json",
                                       "reports/pseudo_reliability.json", "reports/prototype_provenance.json")]
    teacher_report = root / "reports/multiview_teacher_report.json"
    if teacher_report.is_file():
        paths.append(teacher_report)
    for name in ("teacher_pair_checks.json", "prototype_seed_support.json"):
        diagnostic = root / "reports" / name
        if diagnostic.is_file():
            paths.append(diagnostic)
    for name in ("predictions", "checkpoints", "pseudo_cache"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file())
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "code_commit": commit,
               "manual_annotations_opened_before_seal": 0,
               "threshold": config["training"]["threshold"], "checkpoint_rule": "fixed last epoch",
               "variants": list(variants),
               "config_sha256": file_hash(root / "protocol_config.json"),
               "test_images": len(test_ids), "runs": sorted(expected),
               "files": {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths)}}
    atomic_write_json(root / "prediction_seal.json", payload)
    return payload


def verify_seal(root):
    seal = read_json(root / "prediction_seal.json")
    for relative, digest in seal["files"].items():
        if file_hash(root / relative) != digest:
            raise RuntimeError(f"Artifact changed after predictions were frozen: {relative}")
    return seal
