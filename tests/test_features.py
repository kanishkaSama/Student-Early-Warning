"""Feature tests. The most important one: NO DATA LEAKAGE from after the cutoff day."""
import numpy as np
import pandas as pd

from src.features import KEY, NUMERIC_FEATURES, build_features, make_target
from tests.helpers import sample_tables

CUTOFF = 60
POSITIVE = ["Fail", "Withdrawn"]


def _features(tables):
    return build_features(tables, CUTOFF, POSITIVE).set_index(KEY).sort_index()


def test_target_marks_fail_and_withdrawn_as_at_risk():
    results = pd.Series(["Pass", "Fail", "Withdrawn", "Distinction"])
    assert make_target(results, POSITIVE).tolist() == [0, 1, 1, 0]


def test_one_row_per_student_per_course():
    features = build_features(sample_tables(), CUTOFF, POSITIVE)
    assert not features.duplicated(subset=KEY).any()


def test_future_activity_does_not_change_any_feature():
    """If we add clicks and submissions AFTER the cutoff, features must stay identical."""
    tables = sample_tables()
    before = _features(tables)

    future = tables["studentVle"].copy()
    future["date"] = future["date"] + 500          # all far in the future
    future["sum_click"] = future["sum_click"] * 50
    tables["studentVle"] = pd.concat([tables["studentVle"], future], ignore_index=True)

    late = tables["studentAssessment"].copy()
    late["date_submitted"] = CUTOFF + 10
    late["score"] = 0
    tables["studentAssessment"] = pd.concat([tables["studentAssessment"], late], ignore_index=True)

    after = _features(tables)
    pd.testing.assert_frame_equal(before[NUMERIC_FEATURES], after[NUMERIC_FEATURES])


def test_students_who_left_before_cutoff_are_excluded():
    tables = sample_tables()
    registration = tables["studentRegistration"]
    left_early = registration[registration["date_unregistration"] < CUTOFF]
    assert len(left_early) > 0, "sample should contain early leavers"
    features = build_features(tables, CUTOFF, POSITIVE)
    merged = features.merge(left_early[KEY], on=KEY)
    assert merged.empty


def test_unregistration_date_is_not_a_feature():
    features = build_features(sample_tables(), CUTOFF, POSITIVE)
    assert "date_unregistration" not in features.columns


def test_student_with_no_clicks_gets_zero_not_missing():
    tables = sample_tables()
    student = tables["studentInfo"].iloc[0]
    clicks = tables["studentVle"]
    tables["studentVle"] = clicks[clicks["id_student"] != student["id_student"]]
    tables["studentRegistration"].loc[
        tables["studentRegistration"]["id_student"] == student["id_student"], "date_unregistration"] = np.nan
    features = build_features(tables, CUTOFF, POSITIVE)
    row = features[features["id_student"] == student["id_student"]].iloc[0]
    assert row["total_clicks"] == 0
    assert row["days_since_last_activity"] == CUTOFF + 30


def test_submission_rate_is_between_zero_and_one():
    features = build_features(sample_tables(), CUTOFF, POSITIVE)
    rate = features["submission_rate"].dropna()
    assert ((rate >= 0) & (rate <= 1)).all()
