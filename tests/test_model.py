"""Model tests: metrics, threshold choice, and an end-to-end training run."""
import json

import joblib
import numpy as np

from src import predict, train
from src.modeling import choose_threshold, evaluate
from src.store import load_predictions
from tests.helpers import temp_project


def test_evaluate_counts_the_confusion_matrix_correctly():
    y_true = np.array([1, 1, 0, 0, 1])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.7])
    metrics = evaluate(y_true, scores, threshold=0.5)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert metrics["recall"] == round(2 / 3, 4)
    assert metrics["precision"] == round(2 / 3, 4)


def test_threshold_reaches_the_target_recall():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    scores = np.clip(y_true * 0.3 + rng.random(500) * 0.7, 0, 1)
    threshold = choose_threshold(y_true, scores, target_recall=0.8)
    recall = ((scores >= threshold) & (y_true == 1)).sum() / y_true.sum()
    assert recall >= 0.8


def test_training_and_prediction_end_to_end():
    config, data_dir = temp_project(students=40)
    report = train.run(data_dir, config)

    paths = config["paths"]
    assert (paths["models_dir"] / "model.joblib").exists()
    assert json.loads((paths["reports_dir"] / "metrics.json").read_text())["selected_model"]
    # the trained model must do better than guessing on the held-out test set
    assert report["test"]["pr_auc"] > report["test"]["at_risk_rate"]

    bundle = joblib.load(paths["models_dir"] / "model.joblib")
    assert 0 < bundle["threshold"] < 1
    sensitive = set(config["features"]["sensitive_columns"])
    assert sensitive.isdisjoint(bundle["categorical"] + bundle["numeric"])

    scored = predict.run(data_dir, config)
    stored = load_predictions(paths["database"])
    assert len(stored) == len(scored)
    assert stored["risk_score"].between(0, 1).all()


def test_same_seed_gives_same_results():
    config, data_dir = temp_project(students=40)
    first = train.run(data_dir, config)["test"]
    second = train.run(data_dir, config)["test"]
    assert first == second
