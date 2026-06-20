import sys
import os

# Ensure project root is on sys.path so `src` package can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent.classifier import classify_persona_llm
from src.agent.rag import RAG
from src.agent.escalation import load_escalation_config, should_escalate, make_handoff_summary

sample_messages = [
    "Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!",
    "My billing statement has unexpected duplicate charges. I demand an immediate refund!",
    "I'm experiencing an issue with your database integration that's causing internal errors."
]

config = load_escalation_config()
rag = RAG()

for msg in sample_messages:
    print('\n---')
    print('User:', msg)
    persona_info = classify_persona_llm(msg)
    persona = persona_info['persona']
    confidence = persona_info.get('confidence', 0.6)
    retrieved = rag.retrieve(msg, k=4)
    scores = [r[0] for r in retrieved]
    resp = rag.generate_response(persona, msg, retrieved)
    escalate, reason = should_escalate([r[1] for r in retrieved], scores, msg, convo_turns=1, config=config)

    sources = []
    for _, r in retrieved:
        src = r.get('source', 'unknown')
        page = r.get('page')
        sources.append(f"{src}{f' (page {page})' if page else ''}".strip())

    print('Detected persona:', persona, f"(confidence: {confidence:.2f})")
    print('Retrieved sources:', sources)
    print('Escalation:', escalate, reason)
    print('\nAgent response:\n', resp['answer'][:1500])
    if escalate:
        handoff = make_handoff_summary(persona, msg, [msg], sources, [], 'Escalate to human', escalation_reason=reason)
        print('\nHandoff Summary:', handoff)
