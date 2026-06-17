"""Configuration persistence helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from .domain import RunConfig


def load_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as handle:
        return RunConfig.model_validate(yaml.safe_load(handle))


def write_config(config: RunConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.model_dump(mode="json"), handle, sort_keys=False)
