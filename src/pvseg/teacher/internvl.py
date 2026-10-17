from __future__ import annotations

import gc
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
from PIL import Image

from pvseg.attention import sanitize_boxes
from pvseg.utils import model_reference


GRID_PARSER_VERSION = 4
PRESENCE_PARSER_VERSION = 1
IMAGE_PREPROCESS_VERSION = 2

_GRID_STATUSES = {"p": "positive", "n": "negative", "u": "unknown",
                  "positive": "positive", "negative": "negative", "unknown": "unknown"}


def _grid_status(value: Any) -> str:
    return _GRID_STATUSES.get(str(value).strip().lower(), "unknown")


def _is_compact_grid_cell(value: Any) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and not isinstance(value[0], (list, dict)) and isinstance(value[1], str))


def _probability(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    if result > 1.0:
        result /= 100.0
    return max(0.0, min(1.0, result))


def _grid_item_from_fragment(fragment: str, index: int, grid_size: int) -> dict[str, Any] | None:
    """Recover one compact grid item from a complete JSON fragment.

    InternVL can stop mid-response when a verbose 4x4 answer exceeds the token
    budget.  Complete cell objects before the truncation point are still useful
    soft evidence, so recover only their numeric fields and never infer missing
    cells as background.
    """
    def field(pattern: str) -> str | None:
        match = re.search(pattern, fragment, flags=re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else None

    probability = field(
        r'"(?:pv_probability|probability|confidence)"\s*:\s*'
        r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
    )
    if probability is None:
        return None
    row_text = field(r'"row"\s*:\s*(-?\d+)')
    col_text = field(r'"col"\s*:\s*(-?\d+)')
    if row_text is None or col_text is None:
        row, col = divmod(index, grid_size)
    else:
        row, col = int(row_text), int(col_text)
    status = _grid_status(field(r'"status"\s*:\s*"([^"]*)"'))
    return {
        "row": row,
        "col": col,
        "pv_probability": _probability(probability),
        "status": status,
        "reason": (field(r'"reason"\s*:\s*"([^"]*)"') or "")[:300],
    }


def _normalize_grid_coordinates(items: list[dict[str, Any]], grid_size: int = 4) -> list[dict[str, Any]]:
    # Infer the convention once for the entire response. An observed 4 without
    # any zero disambiguates 1..4; an incomplete 1..3 prefix remains zero-based.
    one_based = bool(items) and all(
        1 <= item["row"] <= grid_size and 1 <= item["col"] <= grid_size for item in items
    ) and any(item["row"] == grid_size or item["col"] == grid_size for item in items)
    normalized = {}
    for item in items:
        row, col = item["row"] - int(one_based), item["col"] - int(one_based)
        if 0 <= row < grid_size and 0 <= col < grid_size:
            normalized[(row, col)] = {**item, "row": row, "col": col}
    return [normalized[key] for key in sorted(normalized)]


def _recover_grid_from_truncated_text(text: str, grid_size: int = 4) -> list[dict[str, Any]]:
    """Extract complete grid-cell objects from otherwise invalid JSON text."""
    recovered = []
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        item = _grid_item_from_fragment(match.group(0), len(recovered), grid_size)
        if item is not None:
            recovered.append(item)
    if not recovered:
        grid_start = re.search(r'"grid"\s*:', text)
        # Consume only a contiguous prefix. Searching for arbitrary later pairs
        # would shift coordinates past a malformed/missing cell.
        remaining = text[grid_start.end():].lstrip() if grid_start else ""
        decoder = json.JSONDecoder()
        if not remaining.startswith("["):
            return []
        remaining = remaining[1:].lstrip()
        try:
            first, _ = decoder.raw_decode(remaining)
        except json.JSONDecodeError:
            first = None
        if _is_compact_grid_cell(first):
            # Some teachers flatten the compact 4x4 grid. Recover only a
            # contiguous prefix, retaining the original row-major positions.
            while True:
                try:
                    pair, end = decoder.raw_decode(remaining)
                except json.JSONDecodeError:
                    return recovered
                if not _is_compact_grid_cell(pair):
                    return recovered
                row, col = divmod(len(recovered), grid_size)
                recovered.append({"row": row, "col": col, "pv_probability": _probability(pair[0]),
                                  "status": _grid_status(pair[1]), "reason": ""})
                remaining = remaining[end:].lstrip()
                if not remaining.startswith(","):
                    return recovered
                if len(recovered) == grid_size ** 2:
                    # A further cell would overflow the declared grid.
                    return []
                remaining = remaining[1:].lstrip()
        for row in range(grid_size):
            if not remaining.startswith("["):
                break
            remaining = remaining[1:].lstrip()
            for col in range(grid_size):
                try:
                    pair, end = decoder.raw_decode(remaining)
                except json.JSONDecodeError:
                    return recovered
                if not isinstance(pair, list) or len(pair) != 2:
                    return recovered
                recovered.append({"row": row, "col": col, "pv_probability": _probability(pair[0]),
                                  "status": _grid_status(pair[1]), "reason": ""})
                remaining = remaining[end:].lstrip()
                delimiter = "," if col < grid_size - 1 else "]"
                if not remaining.startswith(delimiter):
                    return recovered
                remaining = remaining[1:].lstrip()
            if not remaining.startswith(","):
                break
            remaining = remaining[1:].lstrip()
    return _normalize_grid_coordinates(recovered, grid_size)


def parse_teacher_response(raw: str, image_size: int = 1024) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    warning = None
    recovered_grid: list[dict[str, Any]] = []
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        recovered_grid = _recover_grid_from_truncated_text(text)
        warning = "truncated_json_recovered_grid" if recovered_grid else "fallback_parser"
        exists_match = re.search(r"(?:pv_exist|存在|光伏).{0,20}(true|false|是|否)", text, re.I)
        confidence_match = re.search(r"(?:confidence|置信度).{0,20}(\d+(?:\.\d+)?)", text, re.I)
        bbox_match = re.search(r"bbox.{0,10}(\[.*?\])", text, re.I | re.DOTALL)
        bbox: Any = []
        if bbox_match:
            try:
                bbox = json.loads(bbox_match.group(1))
            except json.JSONDecodeError:
                bbox = []
        value = {
            "pv_exist": bool(exists_match and exists_match.group(1).lower() in {"true", "是"}),
            "confidence": confidence_match.group(1) if confidence_match else 0.0,
            "bbox": bbox,
            "reason": text[:500],
        }
    if not isinstance(value, dict):
        value = {}
        warning = "not_json_object"
    raw_exists = value.get("pv_exist", value.get("pv_exists", False))
    if isinstance(raw_exists, str):
        exists = raw_exists.strip().lower() in {"true", "1", "yes", "是"}
    else:
        exists = bool(raw_exists)
    confidence = _probability(value.get("confidence", value.get("image_confidence", value.get("pv_probability", 0.0))))
    boxes = sanitize_boxes(value.get("bbox", value.get("bboxes", [])), image_size) if exists else []
    grid = []
    raw_grid = value.get("grid", [])
    if isinstance(raw_grid, list):
        # Compact cells can arrive as four rows or as sixteen row-major pairs.
        # Do not silently trim an oversized response into a complete grid.
        if raw_grid and all(_is_compact_grid_cell(item) for item in raw_grid):
            if len(raw_grid) > 16:
                raw_grid = []
                warning = "invalid_grid_size"
            else:
                raw_grid = [raw_grid[start:start + 4] for start in range(0, len(raw_grid), 4)]
        # Also accept legacy flat {row,col,...} objects and nested object grids.
        if raw_grid and all(isinstance(row, list) for row in raw_grid):
            for row_index, row in enumerate(raw_grid):
                for col_index, item in enumerate(row):
                    if isinstance(item, list) and len(item) == 2:
                        item = {"pv_probability": item[0], "status": item[1]}
                    if isinstance(item, dict):
                        item = {**item, "row": row_index, "col": col_index}
                    if isinstance(item, dict):
                        grid.append({
                            "row": row_index, "col": col_index,
                            "pv_probability": _probability(item.get("pv_probability", item.get("confidence", 0.0))),
                            "status": _grid_status(item.get("status")),
                            "reason": str(item.get("reason", ""))[:300],
                        })
        else:
            for item in raw_grid:
                if not isinstance(item, dict):
                    continue
                try:
                    row, col = int(item["row"]), int(item["col"])
                except (KeyError, TypeError, ValueError):
                    continue
                grid.append({
                    "row": row, "col": col,
                    "pv_probability": _probability(item.get("pv_probability", item.get("confidence", 0.0))),
                    "status": _grid_status(item.get("status")),
                    "reason": str(item.get("reason", ""))[:300],
                })
    grid = _normalize_grid_coordinates(grid)
    if not grid and recovered_grid:
        grid = recovered_grid
    result = {
        "pv_exist": exists,
        "confidence": confidence,
        "bbox": boxes,
        "grid": grid,
        "grid_cell_count": len(grid),
        "grid_complete": len(grid) == 16,
        "confusion_negatives": value.get("confusion_negatives", []),
        "reason": str(value.get("reason", ""))[:1000],
        "raw_response": raw,
    }
    if warning:
        result["parse_warning"] = warning
    # A grid teacher intentionally does not emit a bbox.  This warning is only
    # valid when neither a bbox nor any usable grid evidence is present.
    if exists and not boxes and not grid:
        result["parse_warning"] = "pv_exists_without_valid_bbox"
    return result


def parse_presence_response(raw: str) -> dict[str, Any]:
    """Parse a strict whole-image PV-presence probe without inventing a class.

    This probe is diagnostic only.  Its generated probability is retained for
    protocol checks, but is never treated as a calibrated probability or a
    source of pixel labels.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                  flags=re.IGNORECASE | re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    candidate = match.group(0) if match else text
    warning = None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        value = {}
        warning = "invalid_json"
    if not isinstance(value, dict):
        value = {}
        warning = "not_json_object"
    raw_status = value.get("status")
    status = _grid_status(raw_status)
    raw_probability = value.get("pv_probability")
    probability_valid = (isinstance(raw_probability, (int, float))
                         and not isinstance(raw_probability, bool)
                         and math.isfinite(raw_probability) and 0 <= raw_probability <= 1)
    probability = float(raw_probability) if probability_valid else 0.0
    if not probability_valid:
        warning = warning or "invalid_pv_probability"
    if str(raw_status).strip().lower() not in _GRID_STATUSES:
        warning = warning or "invalid_status"
    result = {
        "status_label": status,
        "pv_probability": probability,
        "parse_valid": warning is None,
        "raw_response": raw,
        "presence_parser_version": PRESENCE_PARSER_VERSION,
    }
    if warning:
        result["parse_warning"] = warning
    return result


def _dynamic_preprocess(image: Image.Image, image_size: int, max_patches: int,
                        use_thumbnail: bool = False) -> list[Image.Image]:
    width, height = image.size
    aspect_ratio = width / height
    layouts = {(1, 1)}
    for rows in range(1, max_patches + 1):
        for columns in range(1, max_patches + 1):
            if 1 <= rows * columns <= max_patches:
                layouts.add((rows, columns))
    # Official area tie-break: choose extra tiles only when the source has enough
    # pixels. Ordering a set previously made equal-aspect-ratio choices arbitrary.
    rows, columns = 1, 1
    best_difference = float("inf")
    for candidate in sorted(layouts, key=lambda layout: (layout[0]*layout[1], layout)):
        difference = abs(aspect_ratio - candidate[1] / candidate[0])
        if difference < best_difference or (
            difference == best_difference
            and width*height > .5 * image_size**2 * candidate[0]*candidate[1]
        ):
            rows, columns = candidate
            best_difference = difference
    resized = image.resize((image_size * columns, image_size * rows), Image.Resampling.BICUBIC)
    tiles = [
        resized.crop(
            (
                column * image_size,
                row * image_size,
                (column + 1) * image_size,
                (row + 1) * image_size,
            )
        )
        for row in range(rows)
        for column in range(columns)
    ]
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


def _pixel_values(image_path: Union[str, Path], model: Any, max_patches: int,
                  view: str = "original", crop_box: Optional[list[int]] = None,
                  use_thumbnail: bool = False) -> tuple[torch.Tensor, int]:
    from torchvision import transforms

    image_size = int(getattr(model.config, "force_image_size", 448) or 448)
    image = Image.open(image_path).convert("RGB")
    if crop_box is not None:
        left, top, right, bottom = crop_box
        scale_x, scale_y = image.width / 1024.0, image.height / 1024.0
        image = image.crop((
            int(max(0, left) * scale_x), int(max(0, top) * scale_y),
            int(min(1024, right) * scale_x), int(min(1024, bottom) * scale_y),
        ))
    if view == "hflip":
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif view != "original":
        raise ValueError(f"Unknown teacher view: {view}")
    tiles = _dynamic_preprocess(image, image_size, max_patches, use_thumbnail)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    values = torch.stack([transform(tile) for tile in tiles])
    device = next(model.parameters()).device
    dtype = next(
        (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()),
        torch.float16,
    )
    return values.to(device=device, dtype=dtype), len(tiles)


@dataclass
class InternVLTeacher:
    model: Any
    tokenizer: Any
    config: dict[str, Any]
    runtime_spec: Optional[dict[str, Any]] = None

    @classmethod
    def load(cls, config: dict[str, Any], logger: logging.Logger) -> "InternVLTeacher":
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        from transformers.modeling_utils import PreTrainedModel

        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            def get_tied_weights(model):
                return (
                    getattr(model, "_all_tied_weights_keys", None)
                    or getattr(model, "_tied_weights_keys", None)
                    or {}
                )

            def set_tied_weights(model, value):
                model._all_tied_weights_keys = value

            PreTrainedModel.all_tied_weights_keys = property(get_tied_weights, set_tied_weights)

        reference, local_only = model_reference(config)
        token = None if local_only else __import__("os").getenv("HF_TOKEN")
        common = {
            "trust_remote_code": True,
            "local_files_only": local_only,
            "token": token,
        }
        tokenizer = AutoTokenizer.from_pretrained(
            reference, use_fast=False, fix_mistral_regex=True, **common
        )
        dtype = torch.float16
        if str(config.get("dtype", "float16")).lower() == "bfloat16" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        requested = str(config.get("quantization", "auto")).lower()
        attempts = ["8bit", "4bit", "none"] if requested == "auto" else [requested]
        last_error: Optional[Exception] = None
        for quantization in attempts:
            kwargs: dict[str, Any] = {
                **common,
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "device_map": "auto",
            }
            if quantization == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            try:
                logger.info("Loading InternVL from %s with quantization=%s", reference, quantization)
                model = AutoModel.from_pretrained(reference, **kwargs).eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                return cls(model=model, tokenizer=tokenizer, config=config,
                           runtime_spec={"dtype": str(dtype), "quantization": quantization,
                                         "model_reference": str(reference)})
            except (RuntimeError, ValueError, ImportError, OSError, AttributeError) as error:
                last_error = error
                logger.exception("InternVL load failed with quantization=%s", quantization)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise RuntimeError(f"Could not load InternVL teacher: {last_error}")

    def _chat(self, image_path: Union[str, Path], view: str,
              crop_box: Optional[list[int]], prompt: str,
              max_new_tokens: int) -> tuple[str, int]:
        values, patch_count = _pixel_values(
            image_path, self.model, int(self.config.get("max_dynamic_patches", 6)), view, crop_box,
            bool(self.config.get("use_thumbnail", False))
        )
        generation = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "temperature": 0.0,
        }
        try:
            raw = self.model.chat(
                self.tokenizer,
                values,
                prompt,
                generation,
                num_patches_list=[patch_count],
                verbose=False,
            )
        finally:
            del values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return raw, patch_count

    @torch.inference_mode()
    def infer(self, image_path: Union[str, Path], view: str = "original",
              crop_box: Optional[list[int]] = None, prompt: Optional[str] = None) -> dict[str, Any]:
        raw, patch_count = self._chat(
            image_path, view, crop_box, prompt or self.config["prompt"],
            int(self.config.get("max_new_tokens", 256))
        )
        result = parse_teacher_response(raw, image_size=1024)
        result["patch_count"] = patch_count
        result["runtime_spec"] = self.runtime_spec
        if crop_box is not None:
            result["crop_box"] = list(crop_box)
        result["flipped"] = view == "hflip"
        return result

    @torch.inference_mode()
    def infer_presence(self, image_path: Union[str, Path], view: str = "original",
                       prompt: Optional[str] = None) -> dict[str, Any]:
        presence_prompt = prompt or self.config["presence_prompt"]
        raw, patch_count = self._chat(
            image_path, view, None, presence_prompt,
            int(self.config.get("presence_max_new_tokens", 64))
        )
        result = parse_presence_response(raw)
        result.update({"patch_count": patch_count, "runtime_spec": self.runtime_spec,
                       "flipped": view == "hflip"})
        return result

