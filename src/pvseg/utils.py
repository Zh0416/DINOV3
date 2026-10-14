from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Union


def setup_logging(output_root: Path, stage: str) -> logging.Logger:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def atomic_write_json(path: Union[str, Path], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def read_json(path: Union[str, Path]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_reference(config_section: dict[str, Any]) -> tuple[str, bool]:
    env_name = config_section.get("local_path_env")
    local_path = os.getenv(env_name, "") if env_name else ""
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"{env_name} points to a missing path: {path}")
        return str(path), True
    return str(config_section["model_id"]), False

