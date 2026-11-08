"""Load the YAML config as a plain dict, with repo-relative paths resolved."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.paths import repo_path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/config.yaml`` (or an explicit path) as a dict."""
    cfg_path = Path(path) if path is not None else repo_path("config", "config.yaml")
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
