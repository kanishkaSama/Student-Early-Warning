"""Exploratory data analysis (Week 1).

Usage:
    python -m src.eda                      # real data in data/raw
    python -m src.eda --data-dir data/sample

Writes charts to reports/figures/eda_*.png and a summary of the actual numbers
to reports/eda_summary.md, so every claim in your report comes from the data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ensure_dirs, load_config
from src.data_loader import load_tables
from src.features import build_features, make_target
from src.logging_utils import get_logger
from src.plot_style import BLUE, INK_SECONDARY, OUTCOME_COLORS, apply_style, save

import matplotlib.pyplot as plt  # noqa: E402  (after style backend is set)

logger = get_logger("eda")
RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]


def _label_outcome(target: pd.Series) -> pd.Series:
    return target.map({1: "At risk", 0: "Not at risk"})


def plot_outcomes(info: pd.DataFrame, out: Path) -> dict:
    counts = info["final_result"].value_counts().reindex(RESULT_ORDER).fillna(0)
    share = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(counts.index, counts.values, color=BLUE, width=0.6)
    for bar, pct in zip(bars, share.values):
        ax.annotate(f"{pct:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9, color=INK_SECONDARY,
                    xytext=(0, 3), textcoords="offset points")
    ax.set_title("Final results of all student registrations")
    ax.set_ylabel("Registrations")
    ax.grid(axis="x", visible=False)
    save(fig, out / "eda_01_outcomes.png")
    return {k: (int(counts[k]), round(float(share[k]), 1)) for k in RESULT_ORDER}


def plot_risk_rate_by(info: pd.DataFrame, column: str, title: str, out: Path,
                      filename: str) -> pd.Series:
    groups = info[column].astype("object").where(info[column].notna(), "Unknown").astype(str)
    rate = info.groupby(groups)["target"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(rate) + 1.2))
    ax.barh(rate.index, rate.values, color=BLUE, height=0.6)
    for y, value in enumerate(rate.values):
        ax.annotate(f"{value:.0f}%", (value, y), va="center", fontsize=9,
                    color=INK_SECONDARY, xytext=(3, 0), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("% of students who failed or withdrew")
    ax.grid(axis="y", visible=False)
    save(fig, out / filename)
    return rate.round(1)


def plot_weekly_clicks(tables: dict, info: pd.DataFrame, out: Path, cutoff: int) -> None:
    clicks = tables["studentVle"][["code_module", "code_presentation", "id_student",
                                   "date", "sum_click"]].copy()
    for col in ("code_module", "code_presentation"):
        clicks[col] = clicks[col].astype(str)
    clicks["week"] = np.floor_divide(clicks["date"], 7)
    key = ["code_module", "code_presentation", "id_student"]
    labels = info[key + ["outcome"]].copy()
    for col in ("code_module", "code_presentation"):
        labels[col] = labels[col].astype(str)
    weekly = clicks.merge(labels, on=key).groupby(["week", "outcome"])["sum_click"].sum()
    students = labels["outcome"].value_counts()
    per_student = weekly.div(students, level="outcome").unstack("outcome")
    per_student = per_student.loc[(per_student.index >= -3) & (per_student.index <= 38)]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for outcome, color in OUTCOME_COLORS.items():
        if outcome in per_student:
            ax.plot(per_student.index, per_student[outcome], color=color, label=outcome)
    ax.axvline(cutoff / 7, color=INK_SECONDARY, linestyle="--", linewidth=1)
    ax.annotate(f"day {cutoff} cutoff", (cutoff / 7, ax.get_ylim()[1] * 0.92), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=INK_SECONDARY)
    ax.set_title("Average online clicks per student, by week of the course")
    ax.set_xlabel("Week of course (0 = course start)")
    ax.set_ylabel("Clicks per student")
    ax.legend(loc="center right")
    save(fig, out / "eda_04_weekly_clicks.png")


def plot_feature_by_outcome(features: pd.DataFrame, column: str, title: str,
                            xlabel: str, out: Path, filename: str, log: bool = False) -> dict:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    groups, medians = [], {}
    for outcome in OUTCOME_COLORS:
        values = features.loc[features["outcome"] == outcome, column].dropna()
        groups.append(np.log10(values + 1) if log else values)
        medians[outcome] = round(float(values.median()), 2) if len(values) else None
    box = ax.boxplot(groups, orientation="horizontal", widths=0.5, patch_artist=True,
                     showfliers=False, tick_labels=list(OUTCOME_COLORS))
    for patch, color in zip(box["boxes"], OUTCOME_COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in box["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    ax.set_title(title)
    ax.set_xlabel(xlabel + (" (log10 scale: 1 = 10, 2 = 100, 3 = 1000)" if log else ""))
    ax.grid(axis="y", visible=False)
    save(fig, out / filename)
    return medians


def run(data_dir: Path, config: dict) -> Path:
    apply_style()
    ensure_dirs(config)
    figures = config["paths"]["figures_dir"]
    cutoff = config["problem"]["cutoff_day"]
    positive = config["problem"]["positive_results"]

    tables = load_tables(data_dir, config["paths"]["processed_dir"])
    info = tables["studentInfo"].copy()
    info["target"] = make_target(info["final_result"], positive)
    info["outcome"] = _label_outcome(info["target"])

    outcome_counts = plot_outcomes(info, figures)
    rate_module = plot_risk_rate_by(info, "code_module", "At-risk rate by module", figures,
                                    "eda_02_risk_by_module.png")
    rate_imd = plot_risk_rate_by(info, "imd_band", "At-risk rate by deprivation band (IMD)",
                                 figures, "eda_03_risk_by_imd.png")
    plot_weekly_clicks(tables, info, figures, cutoff)

    features = build_features(tables, cutoff, positive)
    features["outcome"] = _label_outcome(features["target"])
    click_medians = plot_feature_by_outcome(
        features, "total_clicks", f"Online clicks before day {cutoff}", "Total clicks",
        figures, "eda_05_clicks_before_cutoff.png", log=True)
    submit_medians = plot_feature_by_outcome(
        features, "submission_rate", f"Share of due assessments submitted before day {cutoff}",
        "Submission rate (1.0 = submitted everything)", figures, "eda_06_submission_rate.png")

    total = len(info)
    removed = total - len(features)
    summary = [
        "# EDA summary (generated by src/eda.py)",
        "",
        f"Data folder: `{data_dir}`",
        "",
        "> If the data folder is `data/sample`, every number below is SYNTHETIC and must not be reported.",
        "",
        "## Dataset size",
        *[f"- {name}: {len(df):,} rows" for name, df in tables.items()],
        "",
        "## Final results",
        *[f"- {k}: {v[0]:,} ({v[1]}%)" for k, v in outcome_counts.items()],
        f"- At risk (Fail + Withdrawn): {info['target'].mean() * 100:.1f}%",
        "",
        f"## Prediction population at day {cutoff}",
        f"- Registrations: {total:,}",
        f"- Removed because they had already withdrawn before day {cutoff}: {removed:,}",
        f"- Students left to predict: {len(features):,}, of whom "
        f"{features['target'].mean() * 100:.1f}% end up at risk",
        "",
        "## At-risk rate by module (%)",
        *[f"- {k}: {v}" for k, v in rate_module.items()],
        "",
        "## At-risk rate by IMD band (%)",
        *[f"- {k}: {v}" for k, v in rate_imd.items()],
        "",
        f"## Medians before day {cutoff}",
        f"- Total clicks: {click_medians}",
        f"- Submission rate: {submit_medians}",
        "",
        "Figures: `reports/figures/eda_0*.png`",
    ]
    out_file = config["paths"]["reports_dir"] / "eda_summary.md"
    out_file.write_text("\n".join(summary) + "\n", encoding="utf-8")
    logger.info("Saved 6 figures to %s and summary to %s", figures, out_file)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis.")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config()
    run(args.data_dir or config["paths"]["raw_data_dir"], config)


if __name__ == "__main__":
    main()
