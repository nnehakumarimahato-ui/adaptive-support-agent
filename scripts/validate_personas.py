import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent.classifier import classify_persona_llm
from src.agent.rag import RAG
from src.agent.escalation import load_escalation_config, should_escalate, make_handoff_summary
from src.agent.sentiment import analyze_sentiment

config = load_escalation_config()
rag = RAG()

test_cases = [
    {
        "id": 1,
        "query": "Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!",
        "expected_persona": "Frustrated User",
        "expected_behavior": "Empathize with their trouble, validate the inconvenience, and list clear, simple troubleshooting steps."
    },
    {
        "id": 2,
        "query": "What are the header parameter requirements for your bearer token auth implementation?",
        "expected_persona": "Technical Expert",
        "expected_behavior": "Output code blocks, detailed parameters, and raw HTTP header details."
    },
    {
        "id": 3,
        "query": "Our operational uptime is decreasing. We need a timeline of when billing disputes are resolved.",
        "expected_persona": "Business Executive",
        "expected_behavior": "Keep it highly professional, short, and focused on estimated resolution times and business impacts."
    },
    {
        "id": 4,
        "query": "I'm experiencing an issue with your database integration that's causing internal errors.",
        "expected_persona": "Technical Expert",
        "expected_behavior": "Retrieve relevant documentation and outline step-by-step resolution pathways."
    },
    {
        "id": 5,
        "query": "My billing statement has unexpected duplicate charges. I demand an immediate refund!",
        "expected_persona": "Frustrated User",
        "expected_behavior": "Trigger Escalation: Detect billing sensitivity and generate a structured human handoff JSON."
    },
]

results = []

for test in test_cases:
    query = test["query"]
    persona_info = classify_persona_llm(query)
    persona = persona_info["persona"]
    confidence = persona_info.get("confidence", 0.6)
    
    retrieved = rag.retrieve(query, k=4)
    scores = [r[0] for r in retrieved]
    sources = [r[1].get("source", "unknown") for r in retrieved]
    
    sentiment = analyze_sentiment(query)
    resp = rag.generate_response(persona, query, retrieved)
    
    escalate, reason = should_escalate(
        [r[1] for r in retrieved],
        scores,
        query,
        convo_turns=1,
        config=config,
    )
    
    handoff = None
    if escalate:
        handoff = make_handoff_summary(
            persona,
            query,
            [query],
            sources,
            [],
            "Escalate to a human support specialist.",
            escalation_reason=reason,
        )
    
    result = {
        "test_id": test["id"],
        "query": query,
        "expected_persona": test["expected_persona"],
        "expected_behavior": test["expected_behavior"],
        "detected_persona": persona,
        "confidence": confidence,
        "sentiment": sentiment,
        "escalation": escalate,
        "escalation_reason": reason,
        "retrieved_sources": sources,
        "response_excerpt": resp["answer"][:200] + "..." if len(resp["answer"]) > 200 else resp["answer"],
        "handoff_summary": handoff,
    }
    results.append(result)

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/persona_validation_tests.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

for r in results:
    print(f"\n{'='*80}")
    print(f"TEST {r['test_id']}: {r['query'][:60]}...")
    print(f"Expected Persona: {r['expected_persona']}")
    print(f"Detected Persona: {r['detected_persona']} (confidence={r['confidence']:.2f})")
    print(f"Sentiment: {r['sentiment']['label']} (score={r['sentiment']['score']})")
    print(f"Escalation: {r['escalation']} ({r['escalation_reason']})")
    print(f"Sources: {r['retrieved_sources']}")
    print(f"\nResponse:\n{r['response_excerpt']}\n")
    if r['handoff_summary']:
        print(f"Handoff Summary:\n{json.dumps(r['handoff_summary'], indent=2)}\n")

print(f"\n\nAll tests completed. Results saved to artifacts/persona_validation_tests.json")
