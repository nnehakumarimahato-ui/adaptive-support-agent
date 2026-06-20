import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent.classifier import classify_persona_llm
from src.agent.rag import RAG
from src.agent.escalation import load_escalation_config, should_escalate, make_handoff_summary

config = load_escalation_config()
rag = RAG()

user = "Can you explain the API authentication failure and provide error details?"
persona_info = classify_persona_llm(user)
persona = persona_info["persona"]
confidence = persona_info.get("confidence", 0.6)
retrieved = rag.retrieve(user, k=4)
scores = [r[0] for r in retrieved]
resp = rag.generate_response(persona, user, retrieved)
escalate, reason = should_escalate([r[1] for r in retrieved], scores, user, convo_turns=1, config=config)

sources = []
for r in retrieved:
    source = r[1].get('source', 'unknown')
    page = r[1].get('page')
    if page:
        source = f"{source} (page {page})"
    sources.append(source)

summary = {
    "user": user,
    "persona": persona,
    "confidence": confidence,
    "retrieved_sources": sources,
    "scores": scores,
    "escalate": escalate,
    "reason": reason,
    "answer": resp["answer"],
    "handoff_summary": make_handoff_summary(
        persona,
        user,
        [user],
        sources,
        [],
        "Escalate to human support specialist if unresolved.",
        escalation_reason=reason,
    ) if escalate else None,
}

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/technical_query_transcript.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
