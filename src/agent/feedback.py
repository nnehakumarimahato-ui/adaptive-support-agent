import json
import os
from pathlib import Path
from typing import Any, Dict

ARTIFACTS_DIR = Path("artifacts")
FEEDBACK_LOG = ARTIFACTS_DIR / "feedback.jsonl"
ANALYTICS_LOG = ARTIFACTS_DIR / "analytics.jsonl"


def _ensure_dir():
    ARTIFACTS_DIR.mkdir(exist_ok=True)


def _write_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_feedback(user_text: str, persona: str, confidence: float, top_score: float, sentiment: Dict[str, Any], escalation: bool, escalation_reason: str, feedback: str) -> None:
    payload = {
        "event": "feedback",
        "user_text": user_text,
        "persona": persona,
        "confidence": confidence,
        "top_score": top_score,
        "sentiment": sentiment,
        "escalation": escalation,
        "escalation_reason": escalation_reason,
        "feedback": feedback,
    }
    _write_jsonl(FEEDBACK_LOG, payload)


def log_interaction(user_text: str, persona: str, confidence: float, top_score: float, sentiment: Dict[str, Any], escalation: bool, escalation_reason: str) -> None:
    payload = {
        "event": "interaction",
        "user_text": user_text,
        "persona": persona,
        "confidence": confidence,
        "top_score": top_score,
        "sentiment": sentiment,
        "escalation": escalation,
        "escalation_reason": escalation_reason,
    }
    _write_jsonl(ANALYTICS_LOG, payload)
