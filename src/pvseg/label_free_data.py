"""Image-only selection; annotation paths/categories never enter training records."""
from __future__ import annotations

import hashlib
import shutil
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pvseg.data import find_dataset_root, load_image_tensor
from pvseg.utils import atomic_write_json, read_json


_protect_labels = ContextVar("protect_manual_labels", default=False)
_audit_installed = False


def _audit_open(event, args):
    if event != "open" or not _protect_labels.get() or not args:
        return
    path = str(args[0]).replace("\\", "/").lower()
    if set(path.split("/")) & {"labels", "masks", "mask"}:
        raise PermissionError(f"Manual annotation access blocked before final evaluation: {path}")


@contextmanager
def manual_label_firewall():
    """Accidental-read guard, not a security sandbox. Apply in every pre-eval process."""
    global _audit_installed
    if not _audit_installed:
        sys.addaudithook(_audit_open)
        _audit_installed = True
    token = _protect_labels.set(True)
    try:
        yield
    finally:
        _protect_labels.reset(token)


@dataclass(frozen=True)
class RGBSample:
    sample_id: str
    image_path: str
    rgb_sha256: str
    split: str

    def to_dict(self):
        return asdict(self)


def rgb_hash(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        content = str(image.size).encode() + image.tobytes()
    return hashlib.sha256(content).hexdigest()


def load_rgb_manifest(path):
    payload = read_json(path)
    return [RGBSample(**row) for row in payload["samples"]]


def prepare_rgb_manifest(config, output_root, logger):
    frozen = output_root / "protocol_config.json"
    if frozen.is_file() and read_json(frozen) != config:
        raise RuntimeError("Config changed after selection; use a new output directory")
    data = config["data"]
    dataset_root = find_dataset_root(data["input_root"], data["dataset_slug"], data["categories"])
    extensions = set(data["extensions"])
    # Only image directories are enumerated. No mask existence/pairing check occurs.
    paths = sorted(p for folder in dataset_root.rglob(data["image_dir_name"])
                   if folder.is_dir() for p in folder.rglob("*")
                   if p.is_file() and p.suffix.lower() in extensions)
    excluded_names = set()
    excluded_hashes_from_protocol = set()
    hash_file = data.get("exclude_rgb_hash_file")
    if hash_file:
        hash_path = Path(hash_file)
        if not hash_path.is_absolute():
            hash_path = Path(config.get("_config_path", "")).resolve().parent / hash_path
        if not hash_path.is_file():
            raise RuntimeError(f"Configured RGB exclusion hash file is missing: {hash_path}")
        payload = read_json(hash_path)
        excluded_hashes_from_protocol = set(payload.get("rgb_sha256", []))
        if payload.get("manual_labels_used", True):
            raise RuntimeError("RGB exclusion manifest must not be derived from manual labels")
    cache_name = data.get("exclude_cache")
    explicit_manifest = data.get("exclude_manifest_file")
    cache_manifests = ([Path(explicit_manifest)] if explicit_manifest else
                       [p for p in Path(data["input_root"]).rglob("split_manifest.json")
                        if cache_name and cache_name in p.as_posix()])
    if (cache_name or explicit_manifest) and (len(cache_manifests) != 1 or not cache_manifests[0].is_file()):
        raise RuntimeError("Expected exactly one historical manifest to exclude prior pilot RGBs")
    for path in cache_manifests:
        # Extract image filenames only; no annotation or category fields are consumed.
        excluded_names.update(Path(row["image_path"]).name for row in read_json(path)["samples"])
    hashes = [(p, rgb_hash(p)) for p in paths]
    excluded_hashes = {h for p, h in hashes if p.name in excluded_names} | excluded_hashes_from_protocol
    unique = {}
    for p, h in hashes:
        if h not in excluded_hashes:
            unique.setdefault(h, p)
    def rank(h):
        return hashlib.sha256(f"{config['seed']}:{h}".encode()).hexdigest()
    selected = sorted(unique, key=rank)[:int(data["max_samples"])]
    if len(selected) < int(data["max_samples"]):
        raise RuntimeError("Insufficient fresh unique RGB images")
    staging = Path(data["rgb_staging_dir"])
    staging.mkdir(parents=True, exist_ok=True)
    train_count = int(len(selected) * float(data["train_fraction"]))
    samples, dimensions = [], {}
    for index, digest in enumerate(selected):
        source = unique[digest]
        destination = staging / f"{digest}{source.suffix.lower()}"
        shutil.copyfile(source, destination)
        with Image.open(destination) as image:
            key = f"{image.size[0]}x{image.size[1]}:{image.mode}:bands={image.getbands()}"
            dimensions[key] = dimensions.get(key, 0) + 1
        samples.append(RGBSample(digest, str(destination), digest,
                                 "train" if index < train_count else "test"))
    payload = {"protocol": config["protocol"], "selection": "salted RGB content hash, no category stratification",
               "rgb_files_found": len(paths), "historical_unique_rgb_excluded": len(excluded_hashes),
               "protocol_rgb_hashes_excluded": len(excluded_hashes_from_protocol),
               "duplicates_removed": len(hashes) - len({h for _, h in hashes}),
               "selected": len(samples), "train": train_count, "test": len(samples)-train_count,
               "rgb_dimensions_modes": dimensions, "manual_annotations_opened": 0,
               "samples": [s.to_dict() for s in samples]}
    atomic_write_json(output_root / "rgb_manifest.json", payload)
    # A frozen copy of config is sealed with predictions before annotation access.
    atomic_write_json(output_root / "protocol_config.json", config)
    logger.info("Image-only pilot: total=%d fresh=%d train=%d final-test=%d, no manual labels",
                len(paths), len(samples), train_count, len(samples)-train_count)
    return payload


class RGBDataset(Dataset):
    def __init__(self, samples, image_size):
        self.samples, self.image_size = samples, image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {"sample_id": sample.sample_id,
                "image": load_image_tensor(sample.image_path, self.image_size)}


class PseudoFeatureDataset(Dataset):
    def __init__(self, samples, output_root, pseudo_kind, attention_kind, donors=None, base_attention=.2):
        self.samples, self.root = samples, Path(output_root)
        self.pseudo_kind, self.attention_kind = pseudo_kind, attention_kind
        self.donors = donors or {}
        self.base_attention = float(base_attention)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        features = torch.load(self.root / "feature_cache" / f"{sample.sample_id}.pt",
                              map_location="cpu", weights_only=True).float()
        with np.load(self.root / "pseudo_cache" / f"{sample.sample_id}.npz") as cache:
            target = cache[f"target_{self.pseudo_kind}"].copy()
            if self.pseudo_kind == "soft" and "weight_soft" in cache:
                pseudo_weight = cache["weight_soft"].copy()
            else:
                pseudo_weight = np.ones_like(target, dtype=np.float32)
        if self.attention_kind in ("zero", "ones", "base"):
            value = self.base_attention if self.attention_kind == "base" else float(self.attention_kind == "ones")
            attention = np.full(features.shape[-2:], value, np.float32)
        else:
            donor = self.donors.get(sample.sample_id, sample.sample_id)
            with np.load(self.root / "pseudo_cache" / f"{donor}.npz") as cache:
                key = "attention_soft" if self.attention_kind == "shuffled" else f"attention_{self.attention_kind}"
                attention = cache[key].copy()
        item = {"sample_id": sample.sample_id, "feature": features,
                "pseudo_target": torch.from_numpy(target.astype(np.int64)),
                "attention": torch.from_numpy(attention).float()}
        if self.pseudo_kind == "soft":
            item["pseudo_weight"] = torch.from_numpy(pseudo_weight).float()
        return item
