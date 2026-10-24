"""Turn the raw OULAD tables into one row of features per student per course.

THE KEY RULE (prevents data leakage):
    Every feature uses only information that existed BEFORE `cutoff_day`.
    A tutor on day 60 cannot see clicks from day 61, so neither can the model.

Students who had already withdrawn before the cutoff are removed: they have
already left, so "predicting" them is pointless and would inflate the scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
COURSE_KEY = ["code_module", "code_presentation"]

CATEGORICAL_FEATURES = ["code_module", "region", "highest_education", "imd_band",
                        "age_band", "gender", "disability"]
NUMERIC_FEATURES = [
    "num_of_prev_attempts", "studied_credits", "date_registration", "course_length",
    "total_clicks", "active_days", "distinct_resources", "clicks_last_14d",
    "clicks_prev_14d", "click_trend", "days_since_last_activity", "clicks_vs_cohort",
    "active_days_vs_cohort", "forum_clicks", "quiz_clicks", "content_clicks",
    "assessments_due", "assessments_submitted", "submission_rate", "assessments_missed",
    "late_submissions", "avg_days_late", "mean_score", "min_score", "weighted_score",
]
CLICK_GROUPS = {
    "forum_clicks": {"forumng", "ouwiki", "oucollaborate"},
    "quiz_clicks": {"quiz", "externalquiz", "questionnaire"},
    "content_clicks": {"oucontent", "resource", "subpage", "page", "url"},
}


def make_target(final_result: pd.Series, positive_results: list[str]) -> pd.Series:
    """1 = at risk (fail or withdraw), 0 = pass or distinction."""
    return final_result.isin(positive_results).astype(int)


def _base_table(tables: dict[str, pd.DataFrame], cutoff_day: int) -> pd.DataFrame:
    base = tables["studentInfo"].merge(tables["studentRegistration"], on=KEY, how="left")
    base = base.merge(
        tables["courses"].rename(columns={"module_presentation_length": "course_length"}),
        on=COURSE_KEY, how="left")

    already_left = base["date_unregistration"].notna() & (base["date_unregistration"] < cutoff_day)
    base = base.loc[~already_left].copy()
    # Unregistration date is the outcome itself - it must never become a feature.
    return base.drop(columns=["date_unregistration"])


def _click_features(tables: dict[str, pd.DataFrame], cutoff_day: int) -> pd.DataFrame:
    clicks = tables["studentVle"]
    clicks = clicks.loc[clicks["date"] < cutoff_day,
                        ["code_module", "code_presentation", "id_student", "id_site",
                         "date", "sum_click"]].copy()
    for col in COURSE_KEY:
        clicks[col] = clicks[col].astype(str)

    grouped = clicks.groupby(KEY, observed=True)
    features = pd.DataFrame({
        "total_clicks": grouped["sum_click"].sum(),
        "active_days": grouped["date"].nunique(),
        "distinct_resources": grouped["id_site"].nunique(),
        "last_active_day": grouped["date"].max(),
    })

    recent = clicks["date"] >= cutoff_day - 14
    previous = (clicks["date"] >= cutoff_day - 28) & ~recent
    features["clicks_last_14d"] = clicks.loc[recent].groupby(KEY, observed=True)["sum_click"].sum()
    features["clicks_prev_14d"] = clicks.loc[previous].groupby(KEY, observed=True)["sum_click"].sum()

    site_type = tables["vle"].set_index("id_site")["activity_type"]
    clicks["activity_type"] = clicks["id_site"].map(site_type)
    for feature, types in CLICK_GROUPS.items():
        subset = clicks.loc[clicks["activity_type"].isin(types)]
        features[feature] = subset.groupby(KEY, observed=True)["sum_click"].sum()

    return features.reset_index()


def _assessment_features(tables: dict[str, pd.DataFrame], cutoff_day: int,
                         students: pd.DataFrame) -> pd.DataFrame:
    """Assessments that were DUE before the cutoff, and what the student did by then."""
    due = tables["assessments"]
    due = due.loc[(due["assessment_type"] != "Exam") & (due["date"] < cutoff_day),
                  ["code_module", "code_presentation", "id_assessment", "date", "weight"]]
    due = due.rename(columns={"date": "due_date"})

    # every student x every assessment due in their course
    expected = students[KEY].merge(due, on=COURSE_KEY, how="inner")
    submitted = tables["studentAssessment"]
    submitted = submitted.loc[submitted["date_submitted"] < cutoff_day,
                              ["id_assessment", "id_student", "date_submitted", "score"]]
    merged = expected.merge(submitted, on=["id_assessment", "id_student"], how="left")
    merged["was_submitted"] = merged["date_submitted"].notna()
    merged["days_late"] = (merged["date_submitted"] - merged["due_date"]).clip(lower=0)
    merged["is_late"] = merged["days_late"] > 0
    merged["weighted_points"] = merged["score"].fillna(0) * merged["weight"]

    grouped = merged.groupby(KEY)
    features = pd.DataFrame({
        "assessments_due": grouped["id_assessment"].count(),
        "assessments_submitted": grouped["was_submitted"].sum(),
        "late_submissions": grouped["is_late"].sum(),
        "avg_days_late": grouped["days_late"].mean(),
        "mean_score": grouped["score"].mean(),
        "min_score": grouped["score"].min(),
        "weight_due": grouped["weight"].sum(),
        "weighted_points": grouped["weighted_points"].sum(),
    })
    features["assessments_missed"] = features["assessments_due"] - features["assessments_submitted"]
    features["submission_rate"] = features["assessments_submitted"] / features["assessments_due"]
    features["weighted_score"] = np.where(features["weight_due"] > 0,
                                          features["weighted_points"] / features["weight_due"],
                                          np.nan)
    return features.drop(columns=["weight_due", "weighted_points"]).reset_index()


def build_features(tables: dict[str, pd.DataFrame], cutoff_day: int,
                   positive_results: list[str]) -> pd.DataFrame:
    """Return one row per (module, presentation, student) with features and the target."""
    base = _base_table(tables, cutoff_day)
    for col in COURSE_KEY:
        base[col] = base[col].astype(str)

    data = base.merge(_click_features(tables, cutoff_day), on=KEY, how="left")
    data = data.merge(_assessment_features(tables, cutoff_day, base), on=KEY, how="left")

    # No clicks recorded means zero clicks, not "unknown".
    count_cols = ["total_clicks", "active_days", "distinct_resources", "clicks_last_14d",
                  "clicks_prev_14d", "forum_clicks", "quiz_clicks", "content_clicks",
                  "assessments_due", "assessments_submitted", "late_submissions",
                  "assessments_missed"]
    data[count_cols] = data[count_cols].fillna(0)

    data["click_trend"] = (data["clicks_last_14d"] + 1) / (data["clicks_prev_14d"] + 1)
    # Never active: count from the earliest possible activity, capped for stability.
    data["days_since_last_activity"] = (cutoff_day - data["last_active_day"]).fillna(cutoff_day + 30)

    # Compare each student with classmates in the SAME course run (no label used).
    cohort = data.groupby(COURSE_KEY)
    data["clicks_vs_cohort"] = data["total_clicks"] / (cohort["total_clicks"].transform("median") + 1)
    data["active_days_vs_cohort"] = data["active_days"] / (cohort["active_days"].transform("median") + 1)

    data["target"] = make_target(data["final_result"], positive_results)
    data["cutoff_day"] = cutoff_day
    columns = KEY + ["cutoff_day"] + CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["final_result", "target"]
    columns = list(dict.fromkeys(columns))  # code_module is both a key and a feature
    return data[columns].reset_index(drop=True)


def model_feature_columns(sensitive_columns: list[str], use_sensitive: bool) -> tuple[list[str], list[str]]:
    """Feature lists the model is allowed to see."""
    categorical = [c for c in CATEGORICAL_FEATURES if use_sensitive or c not in sensitive_columns]
    numeric = [c for c in NUMERIC_FEATURES if use_sensitive or c not in sensitive_columns]
    return categorical, numeric
