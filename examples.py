"""Run the five example queries specified in the assignment and print outputs."""
from src.agent.ingest import build_index
from src.agent.rag import RAG
from src.agent.persona import detect_persona
from src.agent.escalation import should_escalate, make_handoff_summary

EXAMPLES = [
    ("Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!", "Frustrated User"),
    ("What are the header parameter requirements for your bearer token auth implementation?", "Technical Expert"),
    ("Our operational uptime is decreasing. We need a timeline of when billing disputes are resolved.", "Business Executive"),
    ("I'm experiencing an issue with your database integration that's causing internal errors.", "Technical Expert"),
    ("My billing statement has unexpected duplicate charges. I demand an immediate refund!", "Frustrated User"),
]


def run_examples():
    print("Building index (if not already built)...")
    build_index("data/docs")
    rag = RAG()
    for q, expected in EXAMPLES:
        print("\n---\nUser:", q)
        persona = detect_persona(q)
        print("Detected persona:", persona, "(expected:", expected, ")")
        retrieved = rag.retrieve(q, k=4)
        scores = [r[0] for r in retrieved]
        resp = rag.generate_response(persona, q, retrieved)
        escalate, reason = should_escalate([r[1] for r in retrieved], scores, q)
        print("Retrieved sources:", [r[1]["source"] for r in retrieved])
        print("Escalate:", escalate, reason)
        print("Agent response:\n", resp["answer"][:1000])
        if escalate:
            handoff = make_handoff_summary(persona, q, [q], [r[1]["source"] for r in retrieved], [], "Investigate billing/account")
            print("Handoff Summary:", handoff)


if __name__ == "__main__":
    run_examples()
