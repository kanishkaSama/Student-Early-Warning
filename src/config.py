"""Load config.yaml and resolve every path relative to the project root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and turn every entry under `paths` into an absolute Path."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["paths"] = {
        name: (PROJECT_ROOT / value if not Path(value).is_absolute() else Path(value))
        for name, value in config["paths"].items()
    }
    return config


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create output folders if they do not exist yet."""
    for key in ("processed_dir", "models_dir", "figures_dir", "reports_dir"):
        config["paths"][key].mkdir(parents=True, exist_ok=True)
