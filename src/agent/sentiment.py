from typing import Dict

POSITIVE_KEYWORDS = {
    "good",
    "great",
    "excellent",
    "thanks",
    "thank you",
    "helpful",
    "happy",
    "resolved",
    "working",
    "fixed",
    "yes",
}

NEGATIVE_KEYWORDS = {
    "bad",
    "frustrated",
    "angry",
    "upset",
    "problem",
    "issue",
    "error",
    "failure",
    "not working",
    "can't",
    "cannot",
    "slow",
    "down",
    "refund",
    "billing",
    "cancel",
}


def analyze_sentiment(text: str) -> Dict[str, object]:
    t = text.lower()
    score = 0
    for word in POSITIVE_KEYWORDS:
        if word in t:
            score += 1
    for word in NEGATIVE_KEYWORDS:
        if word in t:
            score -= 1

    if score > 1:
        label = "positive"
    elif score < -1:
        label = "negative"
    else:
        label = "neutral"

    return {"label": label, "score": float(score)}
