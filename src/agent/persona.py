import re
from typing import Literal


def detect_persona(text: str) -> Literal["Technical Expert", "Frustrated User", "Business Executive"]:
    t = text.lower()
    tech_keywords = ["api", "logs", "stack trace", "error", "debug", "config", "endpoint", "request body"]
    frustrated_keywords = ["angry", "frustrat", "not working", "doesn't work", "nothing works", "tried everything", "help now"]
    exec_keywords = ["impact", "downtime", "sla", "when will", "business", "cost", "ops", "revenue"]

    if any(k in t for k in tech_keywords):
        return "Technical Expert"
    if any(k in t for k in exec_keywords):
        return "Business Executive"
    if any(k in t for k in frustrated_keywords):
        return "Frustrated User"

    # fallback: simple heuristics
    if len(t.split()) > 12 and ("how" in t or "why" in t or "error" in t):
        return "Technical Expert"
    if any(p in t for p in ["please", "help", "urgent"]):
        return "Frustrated User"
    return "Business Executive"
