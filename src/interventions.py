"""Turn a prediction and its reasons into a supportive message a tutor can send.

The LLM runs LOCALLY through Ollama (free, no API key, student data never leaves
the laptop). If Ollama is not running, a rule-based template is used instead,
so the system still works - and the dashboard shows which one produced the text.

Guardrails:
- The prompt contains only behavioural facts (clicks, submissions, scores),
  never gender, disability, age, region or deprivation band.
- The reply is checked; if it mentions a sensitive topic, mentions "risk"/
  "predict"/"model", or is too long, the template is used instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import requests

from src.explain import reason_text
from src.logging_utils import get_logger

logger = get_logger(__name__)

MAX_WORDS = 160
BLOCKED_PATTERNS = [r"\bgender\b", r"\bdisab", r"\bage\b", r"\bdeprivation\b", r"\bimd\b",
                    r"\bincome\b", r"\bpoor\b", r"\bethnic", r"\brisk\b", r"\bpredict",
                    r"\bmodel\b", r"\balgorithm", r"\bfail(ing|ed)?\b", r"\bdrop ?out\b"]

ACTION_HINTS = {
    "active_days": "set aside two short study sessions this week",
    "active_days_vs_cohort": "set aside two short study sessions this week",
    "total_clicks": "spend 20 minutes on this week's online materials",
    "clicks_vs_cohort": "spend 20 minutes on this week's online materials",
    "clicks_last_14d": "log in and look at this week's materials",
    "days_since_last_activity": "log in and look at this week's materials",
    "click_trend": "log in and look at this week's materials",
    "assessments_missed": "talk to me about the assignment you missed - we can plan how to catch up",
    "submission_rate": "talk to me about the assignment you missed - we can plan how to catch up",
    "late_submissions": "plan the next assignment's deadline in your calendar",
    "avg_days_late": "plan the next assignment's deadline in your calendar",
    "mean_score": "book a short call so we can go through your last feedback together",
    "min_score": "book a short call so we can go through your last feedback together",
    "weighted_score": "book a short call so we can go through your last feedback together",
    "forum_clicks": "join this week's discussion in the module forum",
    "quiz_clicks": "try the practice quiz for this unit",
}


@dataclass
class Message:
    text: str
    source: str          # "ollama:<model>" or "template"
    note: str = ""       # why the template was used, if it was


def build_prompt(module: str, reasons: list[dict]) -> str:
    facts = "\n".join(f"- {reason_text(r)}" for r in reasons) or "- No specific concerns found"
    return f"""You are a kind, encouraging university tutor writing a short check-in message
to one of your students on the module {module}.

Facts about the student's recent study activity (compared with a typical student):
{facts}

Write the message:
- 80 to 120 words, warm and non-judgemental, written directly to the student ("you").
- Mention one or two of the facts in a gentle, specific way.
- Suggest two small, concrete next steps.
- Offer help and invite them to reply.
- Do NOT mention grades predictions, risk, models, algorithms, or failing.
- Do NOT mention gender, age, disability, where they live, or their finances.
- Start with "Hi," and end with "Best wishes, Your tutor". Output only the message."""


def template_message(reasons: list[dict]) -> str:
    actions = []
    for reason in reasons:
        hint = ACTION_HINTS.get(reason["feature"])
        if hint and hint not in actions:
            actions.append(hint)
    fallbacks = ["look at this week's study planner",
                 "reply to this message and tell me how the module is going for you"]
    actions = (actions + [f for f in fallbacks if f not in actions])[:2]
    if reasons:
        first = reasons[0]
        noticed = (f"something in your {first['label'].lower()} ({first['value']}, compared "
                   f"with about {first['typical']} for most students)")
    else:
        noticed = "a change in your recent study activity"
    return (
        "Hi,\n\n"
        f"I'm checking in because I noticed {noticed}, and I want "
        "to make sure you have everything you need. Lots of students hit a busy patch at this "
        "point in the course, and small steps make a real difference.\n\n"
        f"Two things that could help this week: {actions[0]}, and {actions[1]}.\n\n"
        "If anything is getting in the way, just reply - I'm happy to help.\n\n"
        "Best wishes,\nYour tutor"
    )


def check_message(text: str) -> str | None:
    """Return a problem description, or None if the message is acceptable."""
    if not text.strip():
        return "empty reply"
    if len(text.split()) > MAX_WORDS:
        return f"too long ({len(text.split())} words)"
    lowered = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return f"mentions a blocked topic ({pattern})"
    return None


def ollama_generate(prompt: str, llm_config: dict) -> str:
    response = requests.post(
        f"{llm_config['base_url'].rstrip('/')}/api/generate",
        json={"model": llm_config["model"], "prompt": prompt, "stream": False,
              "options": {"temperature": llm_config.get("temperature", 0.3)}},
        timeout=llm_config.get("timeout_seconds", 120),
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def ollama_available(llm_config: dict) -> bool:
    try:
        response = requests.get(f"{llm_config['base_url'].rstrip('/')}/api/tags", timeout=3)
        response.raise_for_status()
        names = {m.get("name", "") for m in response.json().get("models", [])}
        return llm_config["model"] in names or f"{llm_config['model']}:latest" in names
    except requests.RequestException:
        return False


def generate_message(module: str, reasons: list[dict], llm_config: dict) -> Message:
    if not llm_config.get("enabled", True):
        return Message(template_message(reasons), "template", "LLM disabled in config")
    try:
        text = ollama_generate(build_prompt(module, reasons), llm_config)
    except requests.RequestException as error:
        logger.warning("Ollama not reachable (%s); using template", error.__class__.__name__)
        return Message(template_message(reasons), "template", "Ollama not running")

    problem = check_message(text)
    if problem:
        logger.warning("LLM reply rejected: %s", problem)
        return Message(template_message(reasons), "template", f"LLM reply rejected: {problem}")
    return Message(text, f"ollama:{llm_config['model']}")
