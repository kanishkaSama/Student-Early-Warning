"""Load and validate the seven OULAD tables.

OULAD facts this module relies on (from the official dataset description):
- All dates are days relative to the start of the course presentation (day 0).
  Negative values mean "before the course started".
- Missing values are written as "?" in the CSV files.
- If an exam has no date, it takes place at the end of the presentation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.logging_utils import get_logger

logger = get_logger(__name__)

NA_VALUES = ["?", ""]

# Required columns for each table. A missing column stops the pipeline early
# with a clear message instead of a confusing error later.
SCHEMA: dict[str, list[str]] = {
    "courses": ["code_module", "code_presentation", "module_presentation_length"],
    "assessments": ["code_module", "code_presentation", "id_assessment",
                    "assessment_type", "date", "weight"],
    "vle": ["id_site", "code_module", "code_presentation", "activity_type",
            "week_from", "week_to"],
    "studentInfo": ["code_module", "code_presentation", "id_student", "gender",
                    "region", "highest_education", "imd_band", "age_band",
                    "num_of_prev_attempts", "studied_credits", "disability",
                    "final_result"],
    "studentRegistration": ["code_module", "code_presentation", "id_student",
                            "date_registration", "date_unregistration"],
    "studentAssessment": ["id_assessment", "id_student", "date_submitted",
                          "is_banked", "score"],
    "studentVle": ["code_module", "code_presentation", "id_student", "id_site",
                   "date", "sum_click"],
}

# studentVle has ~10.6 million rows, so compact types matter on a laptop.
VLE_DTYPES = {
    "code_module": "category",
    "code_presentation": "category",
    "id_student": "int32",
    "id_site": "int32",
    "date": "int16",
    "sum_click": "int32",
}

VALID_RESULTS = {"Pass", "Fail", "Withdrawn", "Distinction"}


class DataValidationError(ValueError):
    """Raised when the dataset does not look like OULAD."""


def _csv_path(data_dir: Path, table: str) -> Path:
    path = data_dir / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name} in {data_dir}. "
            "Download OULAD and unzip the CSV files into this folder "
            "(see README, section 'Get the dataset')."
        )
    return path


def _check_columns(df: pd.DataFrame, table: str) -> None:
    missing = [col for col in SCHEMA[table] if col not in df.columns]
    if missing:
        raise DataValidationError(f"{table}.csv is missing columns: {missing}")


def _load_student_vle(data_dir: Path, cache_dir: Path | None) -> pd.DataFrame:
    """Read the large click table, using a parquet cache when pyarrow is available."""
    cache_file = cache_dir / f"studentVle_{data_dir.name}.parquet" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        try:
            logger.info("Reading studentVle from cache %s", cache_file.name)
            return pd.read_parquet(cache_file)
        except (ImportError, OSError, ValueError) as error:
            logger.warning("Could not read parquet cache (%s); falling back to CSV", error)

    logger.info("Reading studentVle.csv (large file, this can take ~30 s the first time)")
    df = pd.read_csv(_csv_path(data_dir, "studentVle"), dtype=VLE_DTYPES,
                     na_values=NA_VALUES)

    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file, index=False)
            logger.info("Cached studentVle to %s", cache_file.name)
        except ImportError:
            logger.info("pyarrow not installed; skipping parquet cache")
    return df


def load_tables(data_dir: str | Path, cache_dir: str | Path | None = None,
                include_vle_clicks: bool = True) -> dict[str, pd.DataFrame]:
    """Load all OULAD tables into a dict of DataFrames and validate them."""
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir else None
    tables: dict[str, pd.DataFrame] = {}

    for table in SCHEMA:
        if table == "studentVle":
            continue
        tables[table] = pd.read_csv(_csv_path(data_dir, table), na_values=NA_VALUES)

    # The real file calls this column "module_presentation_length"; accept "length" too.
    if "length" in tables["courses"].columns:
        tables["courses"] = tables["courses"].rename(
            columns={"length": "module_presentation_length"})

    if include_vle_clicks:
        tables["studentVle"] = _load_student_vle(data_dir, cache_dir)

    for name, df in tables.items():
        _check_columns(df, name)

    _fill_missing_exam_dates(tables)
    validate_tables(tables)
    logger.info("Loaded %s", ", ".join(f"{k}={len(v):,}" for k, v in tables.items()))
    return tables


def _fill_missing_exam_dates(tables: dict[str, pd.DataFrame]) -> None:
    """Exams without a date happen at the end of the presentation (dataset rule)."""
    assessments = tables["assessments"].merge(
        tables["courses"], on=["code_module", "code_presentation"], how="left")
    missing = assessments["date"].isna()
    assessments.loc[missing, "date"] = assessments.loc[missing, "module_presentation_length"]
    tables["assessments"] = assessments.drop(columns="module_presentation_length")
    if missing.any():
        logger.info("Filled %d missing assessment dates with course length", int(missing.sum()))


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Sanity checks that catch a wrong or corrupted download early."""
    info = tables["studentInfo"]
    unknown = set(info["final_result"].dropna().unique()) - VALID_RESULTS
    if unknown:
        raise DataValidationError(f"Unexpected final_result values: {sorted(unknown)}")

    key = ["code_module", "code_presentation", "id_student"]
    duplicates = int(info.duplicated(subset=key).sum())
    if duplicates:
        raise DataValidationError(f"studentInfo has {duplicates} duplicate student rows")

    if "studentVle" in tables and (tables["studentVle"]["sum_click"] < 0).any():
        raise DataValidationError("studentVle has negative click counts")

    scores = tables["studentAssessment"]["score"].dropna()
    if ((scores < 0) | (scores > 100)).any():
        raise DataValidationError("studentAssessment has scores outside 0-100")
