import os
from typing import Dict, List, Tuple

DEFAULT_CONFIG = {
    "min_score": 0.24,
    "max_turns": 3,
    "sensitive_keywords": [
        "billing",
        "charge",
        "refund",
        "legal",
        "law",
        "account locked",
        "password",
        "security",
        "downtime",
    ],
    "urgent_keywords": ["urgent", "immediately", "asap", "now", "demand", "can't", "cannot"],
}


def load_escalation_config() -> Dict:
    config = DEFAULT_CONFIG.copy()
    try:
        config["min_score"] = float(os.getenv("ESCALATION_MIN_SCORE", config["min_score"]))
    except ValueError:
        pass
    try:
        config["max_turns"] = int(os.getenv("ESCALATION_MAX_TURNS", config["max_turns"]))
    except ValueError:
        pass

    sensitive_raw = os.getenv("ESCALATION_SENSITIVE_KEYWORDS", "")
    if sensitive_raw.strip():
        config["sensitive_keywords"] = [item.strip().lower() for item in sensitive_raw.split(",") if item.strip()]

    return config


def should_escalate(
    retrieved: List[Dict],
    scores: List[float],
    user_text: str,
    convo_turns: int = 1,
    config: Dict = None,
) -> Tuple[bool, str]:
    config = {**DEFAULT_CONFIG, **(config or {})}
    min_score = config["min_score"]
    user_text_lower = user_text.lower()

    if not retrieved or not scores:
        return True, "no_retrieval"

    top_score = max(scores)
    if top_score < min_score:
        return True, "low_confidence"

    if any(keyword in user_text_lower for keyword in config["sensitive_keywords"]):
        return True, "sensitive_issue"

    if any(keyword in user_text_lower for keyword in config["urgent_keywords"]) and top_score < (min_score + 0.1):
        return True, "urgent_low_confidence"

    if convo_turns >= config["max_turns"]:
        return True, "repeat_interactions"

    return False, "ok"


def make_handoff_summary(
    persona: str,
    issue: str,
    convo_history: List[str],
    docs_used: List[str],
    attempted: List[str],
    recommendation: str,
    escalation_reason: str = "",
) -> Dict:
    return {
        "persona": persona,
        "issue_summary": issue,
        "conversation_history": convo_history,
        "documents_used": docs_used,
        "attempted_steps": attempted,
        "recommendation": recommendation,
        "escalation_reason": escalation_reason,
    }
