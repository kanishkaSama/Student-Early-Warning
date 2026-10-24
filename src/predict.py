"""Score students, explain each score, and save the results for the dashboard (Week 3).

Usage:
    python -m src.predict                                # test presentation(s) from config
    python -m src.predict --data-dir data/sample
    python -m src.predict --presentations 2014B 2014J --messages-top 20

`--messages-top N` pre-generates intervention messages for the N highest-risk
students (useful before deploying the demo, where Ollama is not available).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config
from src.data_loader import load_tables
from src.explain import contributions, top_reasons
from src.features import build_features
from src.interventions import generate_message
from src.logging_utils import get_logger
from src.store import reset, save_message, save_predictions

logger = get_logger("predict")


def load_bundle(models_dir: Path) -> dict:
    path = models_dir / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def score(bundle: dict, features: pd.DataFrame, explain_method: str = "auto",
          n_reasons: int = 3) -> pd.DataFrame:
    """Return one row per student with risk score, flag and top reasons."""
    columns = bundle["categorical"] + bundle["numeric"]
    frame = features.reset_index(drop=True)
    risk = bundle["pipeline"].predict_proba(frame[columns])[:, 1]
    contrib, method = contributions(bundle, frame, explain_method)

    out = frame[["code_module", "code_presentation", "id_student", "cutoff_day",
                 "final_result"]].copy()
    out["risk_score"] = risk.round(4)
    out["at_risk"] = (risk >= bundle["threshold"]).astype(int)
    out["reasons_json"] = [
        json.dumps({
            "raises": top_reasons(frame.loc[i], contrib.loc[i], bundle["reference"], n_reasons, "up"),
            "lowers": top_reasons(frame.loc[i], contrib.loc[i], bundle["reference"], n_reasons, "down"),
        })
        for i in frame.index
    ]
    out["explain_method"] = method
    out["model_name"] = bundle["model_name"]
    return out


def run(data_dir: Path, config: dict, presentations: list[str] | None = None,
        messages_top: int = 0) -> pd.DataFrame:
    paths = config["paths"]
    bundle = load_bundle(paths["models_dir"])
    presentations = presentations or config["split"]["test_presentations"]

    tables = load_tables(data_dir, paths["processed_dir"])
    features = build_features(tables, bundle["cutoff_day"], config["problem"]["positive_results"])
    features = features[features["code_presentation"].isin(presentations)]
    if features.empty:
        raise ValueError(f"No students found for presentations {presentations}")

    scored = score(bundle, features, config["explain"]["method"], config["explain"]["top_reasons"])
    reset(paths["database"])  # a new run replaces the old model's results
    saved = save_predictions(paths["database"], scored)
    logger.info("Saved %d predictions (%d flagged at risk, explanations: %s) to %s",
                saved, int(scored["at_risk"].sum()), scored["explain_method"].iat[0],
                paths["database"].name)

    if messages_top:
        top = scored.sort_values("risk_score", ascending=False).head(messages_top)
        for _, row in top.iterrows():
            reasons = json.loads(row["reasons_json"])["raises"]
            message = generate_message(row["code_module"], reasons, config["llm"])
            save_message(paths["database"], row.to_dict(), message.text, message.source, message.note)
        logger.info("Generated %d messages", len(top))
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Score students and store explanations.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--presentations", nargs="+", default=None)
    parser.add_argument("--messages-top", type=int, default=0)
    args = parser.parse_args()
    config = load_config()
    run(args.data_dir or config["paths"]["raw_data_dir"], config, args.presentations,
        args.messages_top)


if __name__ == "__main__":
    main()
