import argparse
from typing import List

from dotenv import load_dotenv
from rich import print

from .classifier import classify_persona_llm
from .escalation import load_escalation_config, make_handoff_summary, should_escalate
from .ingest import build_index
from .rag import RAG


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

        persona_info = classify_persona_llm(user)
        persona = persona_info["persona"]
        confidence = persona_info.get("confidence", 0.6)

        retrieved = rag.retrieve(user, k=4)
        scores = [r[0] for r in retrieved]
        docs = []
        for _, record in retrieved:
            source = record.get("source", "unknown")
            page = record.get("page")
            if page:
                docs.append(f"{source} (page {page})")
            else:
                docs.append(source)
        resp = rag.generate_response(persona, user, retrieved)
        attempted.extend(extract_attempted_steps(user))

        convo_history.append(f"User: {user}")
        convo_history.append(f"Agent: {resp['answer']}")

        escalate, reason = should_escalate(
            [r[1] for r in retrieved],
            scores,
            user,
            convo_turns=len(convo_history) // 2,
            config=config,
        )

        print(f"[bold]Detected persona:[/bold] {persona} (confidence={confidence:.2f})")
        print(f"[bold]Retrieved sources:[/bold] {docs}")
        print(f"[bold]Escalation:[/bold] {escalate} ({reason})")
        print("\n[bold]Agent:[/bold]\n", resp["answer"])

        if escalate:
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
