import re
from typing import Literal


def detect_persona(text: str) -> Literal["Technical Expert", "Frustrated User", "Business Executive"]:
    t = text.lower()
    
    # Expanded keyword lists for better detection
    tech_keywords = [
        "api", "logs", "stack trace", "error", "debug", "config", "endpoint", 
        "request body", "bearer token", "header", "parameter", "database", 
        "integration", "http", "json", "code", "implementation", "port", "connection"
    ]
    frustrated_keywords = [
        "angry", "frustrat", "not working", "doesn't work", "nothing works", 
        "tried everything", "help now", "been an hour", "demand", "urgent", 
        "immediate", "asap", "can't", "cannot", "refuse"
    ]
    exec_keywords = [
        "impact", "downtime", "sla", "when will", "business", "cost", "ops", 
        "revenue", "uptime", "operational", "timeline", "resolution time", "business impacts"
    ]
    
    # Check emotional intensity indicators
    exclamation_count = text.count("!")
    emotion_markers = ["!", "?", "..."]
    emotion_score = sum(text.count(marker) for marker in emotion_markers)
    has_high_emotion = exclamation_count >= 2 or emotion_score >= 3
    
    # Tech expert detection
    if any(k in t for k in tech_keywords):
        return "Technical Expert"
    
    # Executive detection (often mentions business metrics/timeline)
    if any(k in t for k in exec_keywords):
        return "Business Executive"
    
    # Frustrated user detection - higher priority with emotion
    if has_high_emotion and any(k in t for k in frustrated_keywords):
        return "Frustrated User"
    if any(k in t for k in frustrated_keywords):
        return "Frustrated User"
    
    # fallback: simple heuristics
    if len(t.split()) > 12 and ("how" in t or "why" in t or "error" in t):
        return "Technical Expert"
    if any(p in t for p in ["please", "help", "urgent", "asap"]):
        return "Frustrated User"
    
    # If very short and emotional, likely frustrated
    if len(t.split()) <= 10 and has_high_emotion:
        return "Frustrated User"
    
    return "Business Executive"
