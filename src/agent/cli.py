import argparse
from typing import List

from dotenv import load_dotenv
from rich import print

from .classifier import classify_persona_llm
from .escalation import load_escalation_config, make_handoff_summary, should_escalate
from .feedback import log_feedback, log_interaction
from .ingest import build_index
from .rag import RAG
from .sentiment import analyze_sentiment


def ingest_docs(docs_folder: str = "data/docs"):
    print("Ingesting documents and building index...")
    idx = build_index(docs_folder)
    print("Index written to:", idx)


def extract_attempted_steps(user_text: str) -> List[str]:
    triggers = ["reset", "clear", "checked", "restarted", "tried", "updated", "logged in", "confirmed"]
    text = user_text.lower()
    if any(trigger in text for trigger in triggers):
        return [user_text.strip()]
    return []


def _format_retrieved_sources(retrieved: List[tuple]) -> List[str]:
    return [
        f"{record.get('source', 'unknown')} (page {record.get('page')})" if record.get('page') else record.get('source', 'unknown')
        for _, record in retrieved
    ]


def _handle_escalation(user: str, persona: str, confidence: float, retrieved: List[tuple], docs: List[str], convo_history: List[str], attempted: List[str], sentiment: dict, escalate: bool, reason: str) -> None:
    if not escalate:
        return

    summary = make_handoff_summary(
        persona,
        user,
        convo_history,
        docs,
        attempted,
        "Escalate to a human support specialist with account access and billing context.",
        escalation_reason=reason,
    )
    print("\n[bold]Handoff Summary (JSON):[/bold]\n", summary)
    feedback = input("Would you like to add feedback for this handoff? ")
    if feedback.strip():
        log_feedback(user, persona, confidence, max([r[0] for r in retrieved]) if retrieved else 0.0, sentiment, escalate, reason, feedback)


def _print_interaction(user: str, persona: str, confidence: float, sentiment: dict, docs: List[str], resp: dict, escalate: bool, reason: str) -> None:
    print(f"[bold]Detected persona:[/bold] {persona} (confidence={confidence:.2f})")
    print(f"[bold]Sentiment:[/bold] {sentiment['label']} (score={sentiment['score']})")
    print(f"[bold]Retrieved sources:[/bold] {docs}")
    print(f"[bold]Escalation:[/bold] {escalate} ({reason})")
    print("\n[bold]Agent:[/bold]\n", resp["answer"])


def _process_user_message(user: str, rag: RAG, config: dict, convo_history: List[str], attempted: List[str]) -> None:
    persona_info = classify_persona_llm(user)
    persona = persona_info["persona"]
    confidence = persona_info.get("confidence", 0.6)

    retrieved = rag.retrieve(user, k=4)
    scores = [r[0] for r in retrieved]
    docs = _format_retrieved_sources(retrieved)
    sentiment = analyze_sentiment(user)
    resp = rag.generate_response(persona, user, retrieved)
    attempted.extend(extract_attempted_steps(user))

    convo_history.extend([f"User: {user}", f"Agent: {resp['answer']}"])

    escalate, reason = should_escalate(
        [r[1] for r in retrieved],
        scores,
        user,
        convo_turns=len(convo_history) // 2,
        config=config,
    )

    _print_interaction(user, persona, confidence, sentiment, docs, resp, escalate, reason)
    log_interaction(user, persona, confidence, max(scores) if scores else 0.0, sentiment, escalate, reason)
    _handle_escalation(user, persona, confidence, retrieved, docs, convo_history, attempted, sentiment, escalate, reason)


def chat_loop():
    load_dotenv()
    config = load_escalation_config()
    rag = RAG()
    convo_history: List[str] = []
    attempted: List[str] = []

    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting")
            break

        if not user.strip():
            continue
        if user.lower() in ["exit", "quit"]:
            print("Goodbye")
            break

        _process_user_message(user, rag, config, convo_history, attempted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Ingest docs and build index")
    args = parser.parse_args()
    if args.ingest:
        ingest_docs()
    else:
        chat_loop()


if __name__ == "__main__":
    main()
