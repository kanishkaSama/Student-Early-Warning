"""Model definitions, metrics and threshold selection (used by train.py and tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    """Fill gaps, scale numbers and one-hot encode categories.

    Fitted on TRAINING data only, so the medians and categories it learns
    never come from the validation or test students.
    """
    numeric_steps = Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric_steps, numeric),
                              ("cat", categorical_steps, categorical)])


def make_estimator(name: str, random_state: int):
    if name == "majority_baseline":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      class_weight="balanced", n_jobs=-1,
                                      random_state=random_state)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(learning_rate=0.06, max_iter=300,
                                              max_leaf_nodes=31, l2_regularization=1.0,
                                              class_weight="balanced",
                                              early_stopping=True, random_state=random_state)
    raise ValueError(f"Unknown model name: {name}")


def make_pipeline(name: str, categorical: list[str], numeric: list[str],
                  random_state: int) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(categorical, numeric)),
                     ("model", make_estimator(name, random_state))])


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Highest threshold that still catches at least `target_recall` of at-risk students.

    Missing a struggling student costs more than an extra check-in message, so we
    fix recall first and then take the best precision we can get.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # recall[i] belongs to thresholds[i]; the last recall value has no threshold
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds.min())
    return float(thresholds[ok.max()])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n_students": int(len(y_true)),
        "at_risk_rate": round(float(np.mean(y_true)), 4),
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4) if both_classes else None,
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4) if both_classes else None,
        "precision": round(float(precision_score(y_true, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predicted, zero_division=0)), 4),
        "flagged_rate": round(float(predicted.mean()), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def group_report(frame: pd.DataFrame, group_col: str, scores: np.ndarray,
                 threshold: float) -> list[dict]:
    """Recall and flag rate per group - a simple fairness check."""
    rows = []
    predicted = scores >= threshold
    for value, idx in frame.groupby(frame[group_col].astype(str)).groups.items():
        mask = frame.index.isin(idx)
        y = frame.loc[mask, "target"].to_numpy()
        p = predicted[mask]
        positives = y.sum()
        rows.append({
            "group": f"{group_col}={value}",
            "students": int(mask.sum()),
            "at_risk_rate": round(float(y.mean()), 4),
            "flagged_rate": round(float(p.mean()), 4),
            "recall": round(float((p & (y == 1)).sum() / positives), 4) if positives else None,
        })
    return rows
