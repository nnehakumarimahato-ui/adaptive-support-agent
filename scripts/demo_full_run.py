import sys
import os
import json

# Ensure project root is on sys.path so `src` package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent.classifier import classify_persona_llm
from src.agent.rag import RAG
from src.agent.escalation import load_escalation_config, should_escalate, make_handoff_summary


def run_demo():
    config = load_escalation_config()
    rag = RAG()
    convo_history = []
    attempted = []

    # Multi-turn scenario covering three personas
    messages = [
        # Frustrated user with multiple turns
        "Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!",
        "I already tried clearing cache and it didn't help. This is urgent!",
        # Technical expert
        "What are the header parameter requirements for your bearer token auth implementation?",
        # Business executive with billing
        "Our operational uptime is decreasing. We need a timeline of when billing disputes are resolved.",
    ]

    outputs = []

    for i, user in enumerate(messages, start=1):
        persona_info = classify_persona_llm(user)
        persona = persona_info["persona"]
        confidence = persona_info.get("confidence", 0.6)

        retrieved = rag.retrieve(user, k=4)
        scores = [r[0] for r in retrieved]
        resp = rag.generate_response(persona, user, retrieved)

        convo_history.append(f"User: {user}")
        convo_history.append(f"Agent: {resp['answer']}")

        attempted_steps = []
        if any(tok in user.lower() for tok in ["tried", "cleared", "restart", "restarted"]):
            attempted_steps.append(user)
            attempted.extend(attempted_steps)

        escalate, reason = should_escalate(
            [r[1] for r in retrieved],
            scores,
            user,
            convo_turns=len(convo_history) // 2,
            config=config,
        )

        sources = []
        for _, record in retrieved:
            src = record.get("source", "unknown")
            page = record.get("page")
            sources.append(f"{src}{f' (page {page})' if page else ''}".strip())

        handoff = None
        if escalate:
            handoff = make_handoff_summary(
                persona,
                user,
                convo_history.copy(),
                sources,
                attempted,
                "Escalate to human specialist with account and billing access.",
                escalation_reason=reason,
            )

        out = {
            "turn": i,
            "user": user,
            "persona": persona,
            "confidence": confidence,
            "retrieved_sources": sources,
            "scores": scores,
            "escalate": escalate,
            "reason": reason,
            "answer": resp["answer"],
            "handoff_summary": handoff,
        }
        outputs.append(out)

    # Save transcript
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/demo_transcript.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    # Print a readable transcript
    for o in outputs:
        print("\n---\nTurn:", o["turn"]) 
        print("User:", o["user"]) 
        print("Detected persona:", o["persona"], f"(confidence={o['confidence']:.2f})")
        print("Retrieved:", o["retrieved_sources"]) 
        print("Escalation:", o["escalate"], o["reason"]) 
        print("Agent response:\n", o["answer"][:1200])
        if o["handoff_summary"]:
            print("Handoff Summary:", o["handoff_summary"]) 


if __name__ == '__main__':
    run_demo()
