import json
import os
import re
from typing import Dict

try:
    import openai
except Exception:
    openai = None

from .persona import detect_persona as heuristic_persona

OPENAI_KEY_ENV = "OPENAI_API_KEY"
PERSONAS = ["Technical Expert", "Frustrated User", "Business Executive"]


def _extract_json(text: str) -> Dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def classify_persona_llm(text: str) -> Dict:
    """Return a dict: {persona, confidence, reasoning}.

    Uses OpenAI ChatCompletion if OPENAI_API_KEY is present; otherwise falls back to heuristics.
    """
    api_key = os.getenv(OPENAI_KEY_ENV)
    if api_key and openai is not None:
        openai.api_key = api_key
        system = (
            "You are a strict persona classifier. Classify the user's message into exactly one of "
            "'Technical Expert', 'Frustrated User', 'Business Executive'. Return valid JSON with keys: persona, confidence (0-1), reasoning."
        )
        prompt = f"User message: {text}\n\nRespond only with JSON."
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = resp["choices"][0]["message"]["content"]
            j = _extract_json(content)
            if j and j.get("persona") in PERSONAS:
                return {
                    "persona": j["persona"],
                    "confidence": float(j.get("confidence", 0.75)),
                    "reasoning": j.get("reasoning", "LLM classification"),
                }
        except Exception:
            pass

    persona = heuristic_persona(text)
    return {"persona": persona, "confidence": 0.6, "reasoning": "heuristic fallback"}
