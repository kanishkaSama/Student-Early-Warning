"""Shared helpers for the tests: tiny synthetic data and a throw-away config."""
from __future__ import annotations

import copy
import functools
import tempfile
from pathlib import Path

from src.config import load_config
from src.make_sample_data import generate, write_tables


@functools.lru_cache(maxsize=1)
def _cached_tables(students: int):
    return generate(students_per_presentation=students, seed=11)


def sample_tables(students: int = 40) -> dict:
    """A fresh copy of a small synthetic OULAD-shaped dataset."""
    return {name: df.copy() for name, df in _cached_tables(students).items()}


def temp_project(students: int = 40) -> tuple[dict, Path]:
    """Write sample CSVs into a temp folder and return (config, data_dir) pointing there."""
    root = Path(tempfile.mkdtemp(prefix="sew_test_"))
    data_dir = root / "data_sample"
    write_tables(sample_tables(students), data_dir)
    config = copy.deepcopy(load_config())
    config["paths"] = {
        "raw_data_dir": data_dir, "sample_data_dir": data_dir,
        "processed_dir": root / "processed", "models_dir": root / "models",
        "figures_dir": root / "figures", "reports_dir": root / "reports",
        "database": root / "processed" / "test.db",
    }
    config["model"]["candidates"] = ["logistic_regression", "hist_gradient_boosting"]
    config["llm"]["enabled"] = False
    return config, data_dir
