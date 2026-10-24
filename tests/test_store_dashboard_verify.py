"""Storage, dashboard helper and dataset-safety tests."""
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.dashboard_data import filter_predictions, risk_band, student_table, summary
from src.store import latest_message, load_predictions, save_message, save_predictions
from src.verify_dataset import extract_csvs, inspect_zip

REASONS = {"raises": [{"feature": "active_days", "label": "Days active online so far",
                       "value": "4", "typical": "21", "contribution": 1.0,
                       "effect": "raises risk"}], "lowers": []}


def _predictions():
    return pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB"], "code_presentation": ["2014J"] * 3,
        "id_student": [101, 202, 303], "cutoff_day": [60] * 3,
        "risk_score": [0.91, 0.12, 0.55], "at_risk": [1, 0, 1],
        "reasons_json": [json.dumps(REASONS)] * 3,
        "final_result": ["Withdrawn", "Pass", "Pass"],
        "explain_method": ["baseline_replacement"] * 3, "model_name": ["test"] * 3,
    })


def _db() -> Path:
    return Path(tempfile.mkdtemp(prefix="sew_db_")) / "test.db"


def test_predictions_round_trip_and_rerun_does_not_duplicate():
    db = _db()
    save_predictions(db, _predictions())
    save_predictions(db, _predictions())
    stored = load_predictions(db)
    assert len(stored) == 3
    assert stored.loc[stored["id_student"] == 101, "reasons"].iloc[0] == REASONS


def test_latest_message_is_returned():
    db = _db()
    key = {"code_module": "AAA", "code_presentation": "2014J", "id_student": 101, "cutoff_day": 60}
    assert latest_message(db, key) is None
    save_message(db, key, "first", "template")
    save_message(db, key, "second", "ollama:llama3.2:3b")
    assert latest_message(db, key)["message"] == "second"


def test_risk_bands():
    assert risk_band(0.0) == "Low"
    assert risk_band(0.3) == "Medium"
    assert risk_band(1.0) == "High"
    with pytest.raises(ValueError):
        risk_band(1.5)


def test_filters_and_summary():
    db = _db()
    save_predictions(db, _predictions())
    frame = load_predictions(db)
    flagged = filter_predictions(frame, "2014J", ["AAA", "BBB"], only_flagged=True)
    assert flagged["id_student"].tolist() == [101, 303]          # sorted by risk
    stats = summary(frame)
    assert stats == {"students": 3, "flagged": 2, "flagged_pct": 66.7, "actual_at_risk": 1,
                     "recall_pct": 100.0, "precision_pct": 50.0}
    assert student_table(flagged)["Main reason"].iloc[0] == "Days active online so far"


def test_student_search_rejects_non_digits():
    with pytest.raises(ValueError):
        filter_predictions(_predictions(), student_query="1; DROP TABLE")


def _zip(files: dict) -> Path:
    path = Path(tempfile.mkdtemp(prefix="sew_zip_")) / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return path


def test_zip_with_a_program_is_refused():
    with pytest.raises(SystemExit, match="non-CSV"):
        inspect_zip(_zip({"courses.csv": "a,b\n", "run_me.sh": "echo hi"}))


def test_zip_with_a_nested_archive_is_refused():
    with pytest.raises(SystemExit, match="non-CSV"):
        inspect_zip(_zip({"courses.csv": "a,b\n", "more.zip": "PK"}))


def test_zip_with_path_traversal_is_refused():
    with pytest.raises(SystemExit, match="unsafe path"):
        inspect_zip(_zip({"../../evil.csv": "a,b\n"}))


def test_csvs_in_a_subfolder_are_extracted_flat():
    path = _zip({"anonymisedData/courses.csv": "a,b\n1,2\n", "__MACOSX/._x.csv": "junk",
                 "README.txt": "about the data"})
    out = Path(tempfile.mkdtemp(prefix="sew_out_"))
    extract_csvs(path, inspect_zip(path), out)
    assert [p.name for p in out.iterdir()] == ["courses.csv"]
