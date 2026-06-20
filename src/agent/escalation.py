import os
import re
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
        "duplicate charge",
        "duplicate charges",
        "billing dispute",
        "billing statement",
    ],
    "urgent_keywords": ["urgent", "immediately", "asap", "now", "demand", "can't", "cannot"],
    "billing_keywords": [
        "billing",
        "charge",
        "refund",
        "duplicate",
        "credit card",
        "payment",
        "invoice",
        "statement",
        "subscription",
        "billing dispute",
    ],
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


def is_simple_interaction(text: str) -> bool:
    normalized = text.strip().lower()
    return bool(
        re.fullmatch(
            r"^(hi|hello|hey|hiya|thanks?|thank you|good (morning|afternoon|evening)|yo)[.!?]*$",
            normalized,
        )
    )


def detect_billing_issue(user_text: str, config: Dict = None) -> bool:
    """Detect if the user query is about billing-related issues."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    user_text_lower = user_text.lower()
    billing_keywords = config.get("billing_keywords", DEFAULT_CONFIG["billing_keywords"])
    return any(keyword in user_text_lower for keyword in billing_keywords)


def should_escalate(
    retrieved: List[Dict],
    scores: List[float],
    user_text: str,
    convo_turns: int = 1,
    persona: str = "",
    config: Dict = None,
) -> Tuple[bool, str]:
    config = {**DEFAULT_CONFIG, **(config or {})}
    min_score = config["min_score"]
    user_text_lower = user_text.lower()

    if not retrieved or not scores:
        if is_simple_interaction(user_text):
            return False, "greeting"
        return True, "no_retrieval"

    top_score = max(scores)
    
    # Check for billing issue with frustrated persona - always escalate
    if detect_billing_issue(user_text, config) and persona == "Frustrated User":
        if any(keyword in user_text_lower for keyword in config["urgent_keywords"]):
            return True, "billing_dispute_urgent"
        return True, "billing_dispute_sensitive"
    
    # Check for billing issues with any emotional intensity
    if detect_billing_issue(user_text, config):
        exclamation_count = user_text.count("!")
        if exclamation_count >= 2 or any(keyword in user_text_lower for keyword in config["urgent_keywords"]):
            return True, "sensitive_issue_urgent"
        if top_score < (min_score + 0.15):
            return True, "sensitive_issue_low_confidence"
    
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
    sentiment: Dict = None,
    confidence: float = 0.0,
) -> Dict:
    """Create a structured handoff summary for human agents."""
    is_billing = detect_billing_issue(issue)
    
    handoff = {
        "metadata": {
            "escalation_reason": escalation_reason,
            "is_billing_issue": is_billing,
            "priority": "HIGH" if (is_billing and "urgent" in escalation_reason.lower()) else "MEDIUM",
            "persona": persona,
            "confidence": confidence,
        },
        "issue": {
            "summary": issue,
            "documents_consulted": docs_used,
        },
        "conversation": {
            "history": convo_history,
            "sentiment": sentiment or {},
        },
        "agent_context": {
            "attempted_steps": attempted,
            "recommendation": recommendation,
        },
    }
    
    # Add special handling for billing issues
    if is_billing:
        handoff["metadata"]["urgency"] = "CRITICAL" if "duplicate" in issue.lower() else "HIGH"
        handoff["action_required"] = {
            "type": "BILLING_REVIEW",
            "next_steps": [
                "Verify billing records for reported issues",
                "Check for duplicate charges",
                "Review account history",
                "Prepare refund assessment if applicable",
                "Contact customer within 24 hours"
            ],
        }
    
    return handoff
