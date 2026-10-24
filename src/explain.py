"""Explain each prediction: WHICH facts about a student push their risk up or down.

Two methods, same meaning ("compared with a typical student"):

1. SHAP (used when the `shap` package is installed)
   Game-theory based attribution. We give SHAP one background row - the
   typical training student - so every number answers: "how much does this
   student's value of feature X move the prediction away from the typical
   student?" Units: log-odds.

2. Baseline replacement (always available, no extra package)
   For each feature, swap the student's value for the typical value and see
   how much the predicted risk changes. Measured in log-odds (like SHAP), so
   very confident predictions (risk near 0% or 100%) still get readable reasons.
   Simple, model-agnostic, easy to defend in a viva.

If SHAP fails for any reason, we fall back to method 2 and log a warning,
so the dashboard never breaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.logging_utils import get_logger

logger = get_logger(__name__)

# Plain-English names used in the dashboard and in the LLM prompt.
FEATURE_LABELS = {
    "code_module": "Module",
    "region": "Region",
    "highest_education": "Prior education",
    "imd_band": "Deprivation band (IMD)",
    "age_band": "Age band",
    "gender": "Gender",
    "disability": "Declared disability",
    "num_of_prev_attempts": "Previous attempts at this module",
    "studied_credits": "Credits being studied",
    "date_registration": "Registration day (negative = before start)",
    "course_length": "Course length (days)",
    "total_clicks": "Total online clicks so far",
    "active_days": "Days active online so far",
    "distinct_resources": "Different learning resources opened",
    "clicks_last_14d": "Clicks in the last 2 weeks",
    "clicks_prev_14d": "Clicks in the 2 weeks before that",
    "click_trend": "Activity trend (above 1 = rising)",
    "days_since_last_activity": "Days since last online activity",
    "clicks_vs_cohort": "Clicks compared with classmates (1 = typical)",
    "active_days_vs_cohort": "Active days compared with classmates (1 = typical)",
    "forum_clicks": "Forum clicks",
    "quiz_clicks": "Quiz clicks",
    "content_clicks": "Course content clicks",
    "assessments_due": "Assessments due so far",
    "assessments_submitted": "Assessments submitted",
    "submission_rate": "Share of due assessments submitted",
    "assessments_missed": "Assessments missed",
    "late_submissions": "Late submissions",
    "avg_days_late": "Average days late",
    "mean_score": "Average assessment score",
    "min_score": "Lowest assessment score",
    "weighted_score": "Weighted assessment score",
}
# Never shown to tutors or sent to the LLM as a "reason" (fairness and dignity).
NEVER_A_REASON = {"gender", "disability", "age_band", "imd_band", "region"}


def _format(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not available"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


MIN_EFFECT = 0.01  # ignore effects smaller than this (log-odds) - they are noise


def _log_odds(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def baseline_replacement(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Contribution = log-odds(actual) - log-odds(with this feature set to the typical value)."""
    pipeline, reference = bundle["pipeline"], bundle["reference"]
    columns = bundle["categorical"] + bundle["numeric"]
    x = frame[columns].copy()
    actual = _log_odds(pipeline.predict_proba(x)[:, 1])
    contributions = {}
    for col in columns:
        swapped = x.copy()
        swapped[col] = reference[col]
        contributions[col] = actual - _log_odds(pipeline.predict_proba(swapped)[:, 1])
    return pd.DataFrame(contributions, index=frame.index)


def shap_values(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions grouped back to the original (pre one-hot) features."""
    import shap  # optional dependency

    pipeline = bundle["pipeline"]
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    columns = bundle["categorical"] + bundle["numeric"]
    x = prep.transform(frame[columns])
    background = prep.transform(pd.DataFrame([bundle["reference"]])[columns])

    if bundle["model_name"] == "logistic_regression":
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.TreeExplainer(model, data=background,
                                       feature_perturbation="interventional")
    values = explainer.shap_values(x)
    values = np.asarray(values[1] if isinstance(values, list) else values)
    if values.ndim == 3:          # (students, features, classes) -> class "at risk"
        values = values[:, :, 1]

    names = prep.get_feature_names_out()
    grouped = {}
    for col in bundle["numeric"]:
        grouped[col] = values[:, list(names).index(f"num__{col}")]
    for col in bundle["categorical"]:
        idx = [i for i, n in enumerate(names) if n.startswith(f"cat__{col}_")]
        grouped[col] = values[:, idx].sum(axis=1)
    return pd.DataFrame(grouped, index=frame.index)[columns]


def contributions(bundle: dict, frame: pd.DataFrame, method: str = "auto") -> tuple[pd.DataFrame, str]:
    """Return (contributions table, method actually used)."""
    if method in ("auto", "shap"):
        try:
            return shap_values(bundle, frame), "shap"
        except ImportError:
            if method == "shap":
                logger.warning("shap is not installed; using baseline replacement")
        except Exception as error:  # noqa: BLE001 - never let explanations crash the app
            logger.warning("SHAP failed (%s); using baseline replacement", error)
    return baseline_replacement(bundle, frame), "baseline_replacement"


def top_reasons(row: pd.Series, contribution_row: pd.Series, reference: dict,
                n: int = 3, direction: str = "up") -> list[dict]:
    """The n facts that push risk up (or down) the most, in plain English."""
    usable = contribution_row.drop(labels=[c for c in NEVER_A_REASON if c in contribution_row.index])
    ordered = usable.sort_values(ascending=(direction == "down"))
    reasons = []
    for feature, value in ordered.items():
        if (direction == "up" and value < MIN_EFFECT) or (direction == "down" and value > -MIN_EFFECT):
            break
        reasons.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "value": _format(row[feature]),
            "typical": _format(reference.get(feature)),
            "contribution": round(float(value), 4),
            "effect": "raises risk" if value > 0 else "lowers risk",
        })
        if len(reasons) == n:
            break
    return reasons


def reason_text(reason: dict) -> str:
    return (f"{reason['label']}: {reason['value']} "
            f"(typical student: {reason['typical']}) - {reason['effect']}")
