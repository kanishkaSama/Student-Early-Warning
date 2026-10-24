"""Explanation and intervention-message tests (no real LLM needed)."""
from unittest import mock

import pandas as pd
import requests

from src import train
from src.explain import NEVER_A_REASON, contributions, top_reasons
from src.features import build_features
from src.interventions import (build_prompt, check_message, generate_message,
                               template_message)
from src.data_loader import load_tables
from tests.helpers import temp_project

import joblib

REASONS = [
    {"feature": "active_days", "label": "Days active online so far", "value": "4",
     "typical": "21", "contribution": 1.2, "effect": "raises risk"},
    {"feature": "assessments_missed", "label": "Assessments missed", "value": "1",
     "typical": "0", "contribution": 0.8, "effect": "raises risk"},
]
LLM = {"enabled": True, "base_url": "http://localhost:11434", "model": "llama3.2:3b",
       "temperature": 0.3, "timeout_seconds": 5}


def _bundle_and_features():
    config, data_dir = temp_project(students=40)
    train.run(data_dir, config)
    bundle = joblib.load(config["paths"]["models_dir"] / "model.joblib")
    features = build_features(load_tables(data_dir), 60, ["Fail", "Withdrawn"]).head(25)
    return bundle, features.reset_index(drop=True)


def test_explanations_cover_every_feature_and_never_use_sensitive_ones():
    bundle, features = _bundle_and_features()
    contrib, method = contributions(bundle, features, method="baseline")
    assert method == "baseline_replacement"
    assert list(contrib.columns) == bundle["categorical"] + bundle["numeric"]
    assert len(contrib) == len(features)
    for i in features.index:
        reasons = top_reasons(features.loc[i], contrib.loc[i], bundle["reference"], 3, "up")
        assert len(reasons) <= 3
        assert not {r["feature"] for r in reasons} & NEVER_A_REASON
        strengths = [r["contribution"] for r in reasons]
        assert strengths == sorted(strengths, reverse=True)


def test_reasons_skip_sensitive_features_even_if_they_matter():
    row = pd.Series({"gender": "M", "active_days": 3})
    contrib = pd.Series({"gender": 5.0, "active_days": 1.0})
    reasons = top_reasons(row, contrib, {"gender": "F", "active_days": 20}, 3, "up")
    assert [r["feature"] for r in reasons] == ["active_days"]


def test_prompt_contains_facts_but_no_sensitive_words():
    prompt = build_prompt("AAA", REASONS)
    assert "Days active online so far" in prompt
    assert "imd" not in prompt.lower().replace("do not mention", "")


def test_template_message_passes_its_own_safety_check():
    assert check_message(template_message(REASONS)) is None
    assert check_message(template_message([])) is None


def test_safety_check_blocks_risk_talk_and_long_replies():
    assert check_message("Hi, our model predicts you will fail.") is not None
    assert check_message("word " * 200) is not None
    assert check_message("") is not None


def test_falls_back_to_template_when_ollama_is_down():
    with mock.patch("src.interventions.requests.post",
                    side_effect=requests.ConnectionError("refused")):
        message = generate_message("AAA", REASONS, LLM)
    assert message.source == "template"
    assert "not running" in message.note


def test_uses_llm_reply_when_it_is_safe():
    reply = mock.Mock()
    reply.json.return_value = {"response": "Hi, I noticed you have been less active online. "
                                           "Could we have a quick chat? Best wishes, Your tutor"}
    reply.raise_for_status.return_value = None
    with mock.patch("src.interventions.requests.post", return_value=reply) as post:
        message = generate_message("AAA", REASONS, LLM)
    assert message.source == "ollama:llama3.2:3b"
    sent = post.call_args.kwargs["json"]
    assert sent["model"] == "llama3.2:3b" and sent["stream"] is False


def test_rejects_unsafe_llm_reply():
    reply = mock.Mock()
    reply.json.return_value = {"response": "Hi, you are at high risk of failing."}
    reply.raise_for_status.return_value = None
    with mock.patch("src.interventions.requests.post", return_value=reply):
        message = generate_message("AAA", REASONS, LLM)
    assert message.source == "template"
    assert "rejected" in message.note
