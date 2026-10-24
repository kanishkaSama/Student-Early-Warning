"""Train and evaluate the early-warning models (Week 2).

Usage:
    python -m src.train                         # real data in data/raw
    python -m src.train --data-dir data/sample  # dry run on synthetic data

Steps:
1. Build leakage-safe features at the cutoff day.
2. Split by course run: 2013 = train, 2014B = validation, 2014J = test.
3. Train a "predict the majority" baseline and the candidate models.
4. Pick the model with the best PR-AUC on VALIDATION, and its threshold.
5. Score the TEST set exactly once and save everything to models/ and reports/.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import PrecisionRecallDisplay

from src.config import ensure_dirs, load_config
from src.data_loader import load_tables
from src.features import build_features, model_feature_columns
from src.logging_utils import get_logger
from src.modeling import choose_threshold, evaluate, group_report, make_pipeline
from src.plot_style import BLUE, INK_SECONDARY, ORANGE, apply_style, save

import matplotlib.pyplot as plt  # noqa: E402

logger = get_logger("train")


def split_by_presentation(features: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    split_cfg = config["split"]
    parts = {
        "train": features[features["code_presentation"].isin(split_cfg["train_presentations"])],
        "validation": features[features["code_presentation"].isin(split_cfg["validation_presentations"])],
        "test": features[features["code_presentation"].isin(split_cfg["test_presentations"])],
    }
    for name, part in parts.items():
        if part.empty or part["target"].nunique() < 2:
            raise ValueError(f"The {name} split is empty or has only one class - check config.yaml")
        logger.info("%-10s %6d students, %.1f%% at risk", name, len(part), part["target"].mean() * 100)
    return {name: part.reset_index(drop=True) for name, part in parts.items()}


def reference_values(train: pd.DataFrame, categorical: list[str], numeric: list[str]) -> dict:
    """A 'typical student' from the training data, used by the explanation module."""
    reference = {col: float(train[col].median()) for col in numeric}
    for col in categorical:
        mode = train[col].mode(dropna=True)
        reference[col] = str(mode.iloc[0]) if len(mode) else "Unknown"
    return reference


def plot_pr_curves(curves: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    colors = {"logistic_regression": "#6250d6", "random_forest": "#1baf7a",
              "hist_gradient_boosting": BLUE}
    for name, (y, scores) in curves.items():
        PrecisionRecallDisplay.from_predictions(
            y, scores, ax=ax, name=name,
            curve_kwargs={"color": colors.get(name, INK_SECONDARY)})
    ax.set_title("Precision-recall on the validation set")
    ax.legend(loc="lower left")
    save(fig, out / "model_01_pr_curves.png")


def plot_confusion(cm: dict, out: Path, title: str) -> None:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(matrix, cmap="Blues")
    ax.grid(False)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=12,
                color="white" if value > matrix.max() / 2 else "#0b0b0b")
    ax.set_xticks([0, 1], ["Predicted OK", "Predicted at risk"])
    ax.set_yticks([0, 1], ["Actually OK", "Actually at risk"])
    ax.set_title(title)
    save(fig, out / "model_02_confusion_matrix_test.png")


def plot_importance(importance: pd.Series, out: Path) -> None:
    top = importance.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(top) + 1.2))
    colors = [ORANGE if v > 0 else INK_SECONDARY for v in top.values]
    ax.barh(top.index, top.values, color=colors, height=0.6)
    ax.set_title("Which features matter most (permutation importance)")
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    save(fig, out / "model_03_feature_importance.png")


def run(data_dir: Path, config: dict) -> dict:
    apply_style()
    ensure_dirs(config)
    paths = config["paths"]
    cutoff = config["problem"]["cutoff_day"]
    seed = config["model"]["random_state"]
    categorical, numeric = model_feature_columns(config["features"]["sensitive_columns"],
                                                 config["features"]["use_sensitive_columns"])
    feature_cols = categorical + numeric

    tables = load_tables(data_dir, paths["processed_dir"])
    features = build_features(tables, cutoff, config["problem"]["positive_results"])
    splits = split_by_presentation(features, config)
    x_train, y_train = splits["train"][feature_cols], splits["train"]["target"].to_numpy()
    x_val, y_val = splits["validation"][feature_cols], splits["validation"]["target"].to_numpy()

    results, curves, fitted = {}, {}, {}
    for name in ["majority_baseline"] + config["model"]["candidates"]:
        pipeline = make_pipeline(name, categorical, numeric, seed).fit(x_train, y_train)
        scores = pipeline.predict_proba(x_val)[:, 1]
        threshold = 0.5 if name == "majority_baseline" else choose_threshold(
            y_val, scores, config["model"]["target_recall"])
        results[name] = evaluate(y_val, scores, threshold)
        fitted[name] = (pipeline, threshold)
        if name != "majority_baseline":
            curves[name] = (y_val, scores)
        logger.info("%-24s val PR-AUC=%s ROC-AUC=%s recall=%.3f precision=%.3f",
                    name, results[name]["pr_auc"], results[name]["roc_auc"],
                    results[name]["recall"], results[name]["precision"])

    best_name = max(config["model"]["candidates"], key=lambda n: results[n]["pr_auc"])
    best_pipeline, best_threshold = fitted[best_name]
    logger.info("Selected %s (best validation PR-AUC)", best_name)

    test = splits["test"]
    test_scores = best_pipeline.predict_proba(test[feature_cols])[:, 1]
    test_metrics = evaluate(test["target"].to_numpy(), test_scores, best_threshold)
    baseline_test = evaluate(test["target"].to_numpy(),
                             fitted["majority_baseline"][0].predict_proba(test[feature_cols])[:, 1], 0.5)
    fairness = []
    for col in config["features"]["sensitive_columns"]:
        fairness += group_report(test, col, test_scores, best_threshold)

    importance = permutation_importance(best_pipeline, x_val, y_val, scoring="average_precision",
                                        n_repeats=5, random_state=seed, n_jobs=-1)
    importance_series = pd.Series(importance.importances_mean, index=feature_cols)

    plot_pr_curves(curves, paths["figures_dir"])
    plot_confusion(test_metrics["confusion_matrix"], paths["figures_dir"],
                   f"Test set ({', '.join(config['split']['test_presentations'])})")
    plot_importance(importance_series, paths["figures_dir"])

    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "threshold": best_threshold,
        "categorical": categorical,
        "numeric": numeric,
        "cutoff_day": cutoff,
        "reference": reference_values(splits["train"], categorical, numeric),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(data_dir),
    }
    joblib.dump(bundle, paths["models_dir"] / "model.joblib")

    report = {
        "data_dir": str(data_dir),
        "synthetic_data": "sample" in Path(data_dir).name,
        "cutoff_day": cutoff,
        "selected_model": best_name,
        "validation": results,
        "test": test_metrics,
        "test_majority_baseline": baseline_test,
        "fairness_test": fairness,
        "permutation_importance": importance_series.sort_values(ascending=False).round(4).to_dict(),
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    (paths["reports_dir"] / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, paths["reports_dir"] / "model_report.md")
    logger.info("TEST  PR-AUC=%s ROC-AUC=%s recall=%.3f precision=%.3f F1=%.3f",
                test_metrics["pr_auc"], test_metrics["roc_auc"], test_metrics["recall"],
                test_metrics["precision"], test_metrics["f1"])
    logger.info("Saved model to %s and reports to %s", paths["models_dir"], paths["reports_dir"])
    return report


def write_markdown_report(report: dict, path: Path) -> None:
    lines = ["# Model report (generated by src/train.py)", ""]
    if report["synthetic_data"]:
        lines += ["> **SYNTHETIC DATA - these numbers only prove the code runs. Do not report them.**", ""]
    lines += [f"- Data: `{report['data_dir']}`", f"- Cutoff day: {report['cutoff_day']}",
              f"- Split sizes: {report['split_sizes']}",
              f"- Selected model: **{report['selected_model']}** (best validation PR-AUC)", "",
              "## Validation set (used to choose the model and threshold)", "",
              "| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 | Flagged |",
              "|---|---|---|---|---|---|---|"]
    for name, m in report["validation"].items():
        lines.append(f"| {name} | {m['pr_auc']} | {m['roc_auc']} | {m['recall']} | "
                     f"{m['precision']} | {m['f1']} | {m['flagged_rate']} |")
    t, b = report["test"], report["test_majority_baseline"]
    lines += ["", "## Test set (scored once, final honest result)", "",
              f"- PR-AUC {t['pr_auc']} (a random model would score about {t['at_risk_rate']})",
              f"- ROC-AUC {t['roc_auc']}", f"- Threshold {t['threshold']}",
              f"- Recall {t['recall']}, Precision {t['precision']}, F1 {t['f1']}",
              f"- Confusion matrix: {t['confusion_matrix']}",
              f"- Majority baseline F1: {b['f1']} (it never flags anyone)", "",
              "## Fairness check on the test set", "",
              "| Group | Students | At-risk rate | Flagged rate | Recall |", "|---|---|---|---|---|"]
    for row in report["fairness_test"]:
        lines.append(f"| {row['group']} | {row['students']} | {row['at_risk_rate']} | "
                     f"{row['flagged_rate']} | {row['recall']} |")
    lines += ["", "## Top 10 features (permutation importance on validation)", ""]
    for name, value in list(report["permutation_importance"].items())[:10]:
        lines.append(f"- {name}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the early-warning model.")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config()
    run(args.data_dir or config["paths"]["raw_data_dir"], config)


if __name__ == "__main__":
    main()
