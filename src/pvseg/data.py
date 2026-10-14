from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    sample_id: str
    category: str
    image_path: str
    mask_path: str
    split: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _has_categories(path: Path, categories: Iterable[str]) -> bool:
    return all((path / category).is_dir() for category in categories)


def find_dataset_root(input_root: Union[str, Path], dataset_slug: str, categories: list[str]) -> Path:
    base = Path(input_root)
    candidates = [
        base / dataset_slug,
        base / "datasets" / "zh0416" / dataset_slug,
        base,
    ]
    for candidate in candidates:
        if candidate.is_dir() and _has_categories(candidate, categories):
            return candidate
    if not base.exists():
        raise FileNotFoundError(f"Input root does not exist: {base}")
    for candidate in base.rglob("manifest.csv"):
        if _has_categories(candidate.parent, categories):
            return candidate.parent
    for first_category in base.rglob(categories[0]):
        if first_category.is_dir() and _has_categories(first_category.parent, categories):
            return first_category.parent
    raise FileNotFoundError(
        f"Could not locate a dataset containing {categories} below {base}"
    )


def _find_mask(
    category_root: Path,
    image_path: Path,
    image_root: Path,
    label_dirs: list[str],
    extensions: set[str],
) -> Optional[Path]:
    relative = image_path.relative_to(image_root)
    for label_dir in label_dirs:
        label_root = category_root / label_dir
        direct = label_root / relative
        if direct.is_file():
            return direct
        parent = direct.parent
        if parent.is_dir():
            for extension in extensions:
                candidate = parent / f"{image_path.stem}{extension}"
                if candidate.is_file():
                    return candidate
    return None


def discover_samples(root: Union[str, Path], data_config: dict[str, Any]) -> list[Sample]:
    dataset_root = Path(root)
    extensions = {value.lower() for value in data_config["extensions"]}
    samples: list[Sample] = []
    missing_masks: list[str] = []
    for category in data_config["categories"]:
        category_root = dataset_root / category
        image_root = category_root / data_config["image_dir_name"]
        if not image_root.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_root}")
        image_paths = sorted(
            path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in extensions
        )
        for image_path in image_paths:
            mask_path = _find_mask(
                category_root,
                image_path,
                image_root,
                data_config["label_dir_names"],
                extensions,
            )
            if mask_path is None:
                missing_masks.append(str(image_path))
                continue
            sample_id = f"{category}__{image_path.stem}"
            samples.append(Sample(sample_id, category, str(image_path), str(mask_path)))
    if missing_masks:
        preview = "\n".join(missing_masks[:10])
        raise FileNotFoundError(f"Missing masks for {len(missing_masks)} images. First entries:\n{preview}")
    if not samples:
        raise RuntimeError(f"No paired samples found in {dataset_root}")
    return samples


def make_stratified_split(
    samples: list[Sample],
    samples_per_category: int,
    fractions: list[float],
    seed: int,
) -> list[Sample]:
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("data.split must contain train/val/test fractions summing to 1")
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.category].append(sample)
    selected: list[Sample] = []
    rng = random.Random(seed)
    split_names = ("train", "val", "test")
    for category in sorted(grouped):
        category_samples = grouped[category][:]
        rng.shuffle(category_samples)
        category_samples = category_samples[: min(samples_per_category, len(category_samples))]
        count = len(category_samples)
        train_count = int(round(count * fractions[0]))
        val_count = int(round(count * fractions[1]))
        train_count = min(train_count, count)
        val_count = min(val_count, count - train_count)
        boundaries = (train_count, train_count + val_count, count)
        start = 0
        for split_name, end in zip(split_names, boundaries):
            selected.extend(
                Sample(item.sample_id, item.category, item.image_path, item.mask_path, split_name)
                for item in category_samples[start:end]
            )
            start = end
    rng.shuffle(selected)
    return selected


def create_split_manifest(config: dict[str, Any], output_root: Path) -> tuple[Path, list[Sample]]:
    manifest_path = output_root / "split_manifest.json"
    if manifest_path.is_file() and config["runtime"].get("resume", True):
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest_path, [Sample(**item) for item in raw["samples"]]
    data_config = config["data"]
    root = find_dataset_root(
        data_config["input_root"], data_config["dataset_slug"], data_config["categories"]
    )
    all_samples = discover_samples(root, data_config)
    samples_per_category = min(
        int(data_config["samples_per_category"]),
        max(1, int(data_config["max_samples"]) // len(data_config["categories"])),
    )
    selected = make_stratified_split(
        all_samples,
        samples_per_category,
        [float(value) for value in data_config["split"]],
        int(config["seed"]),
    )
    payload = {
        "dataset_root": str(root),
        "seed": int(config["seed"]),
        "all_counts": dict(Counter(sample.category for sample in all_samples)),
        "selected_counts": dict(Counter(sample.category for sample in selected)),
        "split_counts": dict(Counter(sample.split for sample in selected)),
        "samples": [sample.to_dict() for sample in selected],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path, selected


def load_split_manifest(path: Union[str, Path], split: Optional[str] = None) -> list[Sample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [Sample(**item) for item in payload["samples"]]
    return [sample for sample in samples if split is None or sample.split == split]


def load_image_tensor(path: Union[str, Path], size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mask_tensor(path: Union[str, Path], size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    array = np.asarray(mask, dtype=np.uint8)
    return torch.from_numpy((array > 0).astype(np.int64))


class ImageMaskDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "image": load_image_tensor(sample.image_path, self.image_size),
            "mask": load_mask_tensor(sample.mask_path, self.image_size),
            **sample.to_dict(),
        }


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        feature_dir: Union[str, Path],
        attention_dir: Union[str, Path],
        image_size: int,
    ):
        self.samples = samples
        self.feature_dir = Path(feature_dir)
        self.attention_dir = Path(attention_dir)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        feature_path = self.feature_dir / f"{sample.sample_id}.pt"
        attention_path = self.attention_dir / f"{sample.sample_id}.npy"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Missing DINO feature: {feature_path}")
        if not attention_path.is_file():
            raise FileNotFoundError(f"Missing attention map: {attention_path}")
        feature = torch.load(feature_path, map_location="cpu", weights_only=True)
        if isinstance(feature, dict):
            feature = feature["feature"]
        attention = torch.from_numpy(np.load(attention_path)).float()
        return {
            "feature": feature.float(),
            "attention": attention,
            "mask": load_mask_tensor(sample.mask_path, self.image_size),
            **sample.to_dict(),
        }

