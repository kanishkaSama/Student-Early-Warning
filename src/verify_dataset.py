"""Safely unpack and verify the OULAD download.

Usage:
    python -m src.verify_dataset --zip ~/Downloads/open+university+learning+analytics+dataset.zip
    python -m src.verify_dataset            # only verify files already in data/raw

What it checks:
1. The zip contains only CSV data (plus harmless documentation files, which are
   skipped) - no programs, scripts or nested archives - and no unsafe paths.
2. All 7 expected tables are present with the expected columns.
3. Row counts match the published dataset (a mismatch is a warning, not an error).
It also prints the SHA-256 fingerprint of the zip so you can record it in your report.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from src.config import load_config
from src.data_loader import NA_VALUES, SCHEMA
from src.logging_utils import get_logger

logger = get_logger("verify_dataset")

# Row counts of the public release, as documented for OULAD (Kuzilek et al., 2017).
EXPECTED_ROWS = {
    "courses": 22,
    "assessments": 206,
    "vle": 6_364,
    "studentInfo": 32_593,
    "studentRegistration": 32_593,
    "studentAssessment": 173_912,
    "studentVle": 10_655_280,
}
MAX_UNCOMPRESSED_BYTES = 1_500_000_000  # guards against "zip bombs"
DOCUMENTATION_SUFFIXES = {".txt", ".md", ".pdf", ".rtf", ".names"}  # skipped, never extracted


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_zip(zip_path: Path) -> list[zipfile.ZipInfo]:
    """Return the CSV members of the zip, refusing anything suspicious."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]

    problems, csv_members = [], []
    for member in members:
        name = PurePosixPath(member.filename)
        if name.is_absolute() or ".." in name.parts:
            problems.append(f"unsafe path: {member.filename}")
        elif name.name.startswith("._") or "__MACOSX" in name.parts:
            continue  # harmless macOS metadata, skipped
        elif name.suffix.lower() in DOCUMENTATION_SUFFIXES:
            logger.info("Skipping documentation file %s", member.filename)
        elif name.suffix.lower() != ".csv":
            problems.append(f"non-CSV file: {member.filename}")
        else:
            csv_members.append(member)

    total = sum(m.file_size for m in csv_members)
    if total > MAX_UNCOMPRESSED_BYTES:
        problems.append(f"uncompressed size {total / 1e9:.1f} GB is larger than expected")
    if problems:
        raise SystemExit("Refusing to extract:\n  " + "\n  ".join(problems))

    logger.info("Zip contents look safe: %d CSV files, %.0f MB uncompressed",
                len(csv_members), total / 1e6)
    return csv_members


def extract_csvs(zip_path: Path, members: list[zipfile.ZipInfo], out_dir: Path) -> None:
    """Extract CSV files flat into out_dir (folder structure inside the zip is ignored)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in members:
            target = out_dir / PurePosixPath(member.filename).name
            with archive.open(member) as src, target.open("wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            logger.info("Extracted %s", target.name)


def count_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle) - 1  # minus header


def verify_folder(data_dir: Path) -> bool:
    ok = True
    for table, columns in SCHEMA.items():
        path = data_dir / f"{table}.csv"
        if not path.exists():
            logger.error("MISSING  %s", path.name)
            ok = False
            continue
        header = pd.read_csv(path, nrows=5, na_values=NA_VALUES).columns.tolist()
        if table == "courses" and "length" in header:
            header = [("module_presentation_length" if c == "length" else c) for c in header]
        missing_cols = [c for c in columns if c not in header]
        rows = count_rows(path)
        expected = EXPECTED_ROWS[table]
        if missing_cols:
            logger.error("BAD      %-20s missing columns %s", path.name, missing_cols)
            ok = False
        elif rows != expected:
            logger.warning("CHECK    %-20s %12s rows (published release: %s)",
                           path.name, f"{rows:,}", f"{expected:,}")
        else:
            logger.info("OK       %-20s %12s rows", path.name, f"{rows:,}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, help="path to the downloaded OULAD zip")
    parser.add_argument("--data-dir", type=Path, help="folder with the CSV files")
    args = parser.parse_args()

    config = load_config()
    data_dir = args.data_dir or config["paths"]["raw_data_dir"]

    if args.zip:
        zip_path = args.zip.expanduser()
        if not zip_path.exists():
            logger.error("Zip not found: %s", zip_path)
            return 1
        logger.info("SHA-256 of %s: %s", zip_path.name, sha256_of(zip_path))
        extract_csvs(zip_path, inspect_zip(zip_path), data_dir)

    if verify_folder(data_dir):
        logger.info("Dataset verified in %s", data_dir)
        return 0
    logger.error("Dataset check failed. Fix the problems above before training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
