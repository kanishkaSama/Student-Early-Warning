"""Data helpers for the Streamlit dashboard. Kept separate so they can be unit-tested."""
from __future__ import annotations

import pandas as pd

RISK_BANDS = [(0.0, 0.3, "Low"), (0.3, 0.6, "Medium"), (0.6, 1.01, "High")]


def risk_band(score: float) -> str:
    for low, high, name in RISK_BANDS:
        if low <= score < high:
            return name
    raise ValueError(f"risk score out of range: {score}")


def filter_predictions(predictions: pd.DataFrame, presentation: str | None = None,
                       modules: list[str] | None = None, only_flagged: bool = False,
                       student_query: str = "") -> pd.DataFrame:
    frame = predictions
    if presentation:
        frame = frame[frame["code_presentation"] == presentation]
    if modules:
        frame = frame[frame["code_module"].isin(modules)]
    if only_flagged:
        frame = frame[frame["at_risk"] == 1]
    query = student_query.strip()
    if query:
        if not query.isdigit():
            raise ValueError("Student ID must contain digits only")
        frame = frame[frame["id_student"].astype(str).str.contains(query, regex=False)]
    return frame.sort_values("risk_score", ascending=False)


def student_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns a tutor needs, in readable form."""
    table = pd.DataFrame({
        "Student ID": frame["id_student"].astype(str),
        "Module": frame["code_module"],
        "Risk %": (frame["risk_score"] * 100).round(0).astype(int),
        "Band": frame["risk_score"].map(risk_band),
        "Flagged": frame["at_risk"].map({1: "Yes", 0: "No"}),
        "Main reason": frame["reasons"].map(
            lambda r: r["raises"][0]["label"] if r.get("raises") else "-"),
    })
    return table.reset_index(drop=True)


def summary(frame: pd.DataFrame) -> dict:
    """Headline numbers. Outcome-based checks only make sense for past course runs."""
    flagged = frame[frame["at_risk"] == 1]
    result = {
        "students": len(frame),
        "flagged": len(flagged),
        "flagged_pct": round(100 * len(flagged) / len(frame), 1) if len(frame) else 0.0,
    }
    known = frame["final_result"].notna().all() and len(frame) > 0
    if known:
        actual = frame["final_result"].isin(["Fail", "Withdrawn"])
        caught = (actual & (frame["at_risk"] == 1)).sum()
        result["actual_at_risk"] = int(actual.sum())
        result["recall_pct"] = round(100 * caught / actual.sum(), 1) if actual.sum() else None
        result["precision_pct"] = round(100 * caught / len(flagged), 1) if len(flagged) else None
    return result


def reasons_frame(reasons: dict) -> pd.DataFrame:
    rows = [{"Fact": r["label"], "This student": r["value"], "Typical student": r["typical"],
             "Effect": r["effect"], "Strength": r["contribution"]}
            for r in reasons.get("raises", []) + reasons.get("lowers", [])]
    return pd.DataFrame(rows, columns=["Fact", "This student", "Typical student", "Effect", "Strength"])


def flagged_by_module(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("code_module")["at_risk"].agg(["count", "sum"])
    grouped["Flagged %"] = (100 * grouped["sum"] / grouped["count"]).round(1)
    return grouped.rename(columns={"count": "Students", "sum": "Flagged"}).reset_index(names="Module")
