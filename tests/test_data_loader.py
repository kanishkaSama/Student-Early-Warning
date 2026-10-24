"""Data validation tests: the loader must catch bad or incomplete downloads."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataValidationError, load_tables
from src.make_sample_data import write_tables
from tests.helpers import sample_tables


def _write(tables) -> Path:
    folder = Path(tempfile.mkdtemp(prefix="sew_loader_"))
    write_tables(tables, folder)
    return folder


def test_loads_all_seven_tables_and_parses_question_marks_as_missing():
    tables = load_tables(_write(sample_tables()))
    assert set(tables) == {"courses", "assessments", "vle", "studentInfo",
                           "studentRegistration", "studentAssessment", "studentVle"}
    # "?" in the CSV must become NaN, not the text "?"
    assert not (tables["studentInfo"]["imd_band"] == "?").any()
    assert tables["studentInfo"]["imd_band"].isna().any()


def test_missing_exam_dates_are_filled_with_course_length():
    tables = load_tables(_write(sample_tables()))
    exams = tables["assessments"][tables["assessments"]["assessment_type"] == "Exam"]
    assert exams["date"].notna().all()
    merged = exams.merge(tables["courses"], on=["code_module", "code_presentation"])
    assert (merged["date"] == merged["module_presentation_length"]).all()


def test_missing_file_gives_a_helpful_error():
    folder = _write(sample_tables())
    (folder / "studentVle.csv").unlink()
    with pytest.raises(FileNotFoundError, match="studentVle.csv"):
        load_tables(folder)


def test_missing_column_is_rejected():
    tables = sample_tables()
    tables["studentInfo"] = tables["studentInfo"].drop(columns="final_result")
    with pytest.raises(DataValidationError, match="final_result"):
        load_tables(_write(tables))


def test_unknown_final_result_is_rejected():
    tables = sample_tables()
    tables["studentInfo"].loc[0, "final_result"] = "Graduated"
    with pytest.raises(DataValidationError, match="Graduated"):
        load_tables(_write(tables))


def test_out_of_range_scores_are_rejected():
    tables = sample_tables()
    tables["studentAssessment"].loc[0, "score"] = 140
    with pytest.raises(DataValidationError, match="0-100"):
        load_tables(_write(tables))


def test_duplicate_student_rows_are_rejected():
    tables = sample_tables()
    info = tables["studentInfo"]
    tables["studentInfo"] = pd.concat([info, info.head(1)], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        load_tables(_write(tables))
