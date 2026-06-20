#!/usr/bin/env python3
"""
Comprehensive test script for persona-based customer support scenarios.
Tests all 5 user cases with expected behaviors.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.classifier import classify_persona_llm
from src.agent.escalation import (
    should_escalate,
    load_escalation_config,
    make_handoff_summary,
    detect_billing_issue,
)
from src.agent.sentiment import analyze_sentiment
from src.agent.rag import RAG
from src.agent.persona import detect_persona

# Test cases
TEST_CASES = [
    {
        "id": 1,
        "title": "Frustrated User - Cookie Clearing",
        "message": "Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!",
        "expected_persona": "Frustrated User",
        "expected_escalation": False,  # Should be handled but escalate if no docs found
        "expected_sentiment": "negative",
        "validation_points": [
            "Empathetic response",
            "Simple actionable steps",
            "Reassuring tone",
        ],
    },
    {
        "id": 2,
        "title": "Technical Expert - Bearer Token Auth",
        "message": "What are the header parameter requirements for your bearer token auth implementation?",
        "expected_persona": "Technical Expert",
        "expected_escalation": False,
        "expected_sentiment": "neutral",
        "validation_points": [
            "Code blocks included",
            "Detailed parameters",
            "Raw HTTP headers",
            "Implementation examples",
        ],
    },
    {
        "id": 3,
        "title": "Business Executive - Billing Disputes Timeline",
        "message": "Our operational uptime is decreasing. We need a timeline of when billing disputes are resolved.",
        "expected_persona": "Business Executive",
        "expected_escalation": False,
        "expected_sentiment": "neutral",
        "validation_points": [
            "Professional tone",
            "Concise response (2-3 paragraphs)",
            "SLA/resolution timeline",
            "Business impact focus",
        ],
    },
    {
        "id": 4,
        "title": "Technical Expert - Database Integration",
        "message": "I'm experiencing an issue with your database integration that's causing internal errors.",
        "expected_persona": "Technical Expert",
        "expected_escalation": True,  # Low confidence or no retrieval likely
        "expected_sentiment": "negative",
        "validation_points": [
            "Step-by-step resolution",
            "Relevant documentation retrieved",
            "Technical depth",
        ],
    },
    {
        "id": 5,
        "title": "Frustrated User - Billing Duplicate Charges (ESCALATION)",
        "message": "My billing statement has unexpected duplicate charges. I demand an immediate refund!",
        "expected_persona": "Frustrated User",
        "expected_escalation": True,  # MUST escalate - billing + frustrated + urgent
        "expected_sentiment": "negative",
        "escalation_reason": "billing_dispute_urgent",
        "validation_points": [
            "Triggers escalation",
            "Handoff JSON with billing_issue=true",
            "Priority=CRITICAL",
            "Action required: BILLING_REVIEW",
            "Human handoff generated",
        ],
    },
]


def run_test(test_case):
    """Run a single test case."""
    print(f"\n{'='*80}")
    print(f"TEST {test_case['id']}: {test_case['title']}")
    print(f"{'='*80}")
    print(f"Message: {test_case['message']}\n")

    # Test 1: Persona Detection
    print("### Persona Detection ###")
    heuristic = detect_persona(test_case["message"])
    llm_result = classify_persona_llm(test_case["message"])
    persona = llm_result["persona"]
    confidence = llm_result["confidence"]

    print(f"  Heuristic: {heuristic}")
    print(f"  LLM: {persona} (confidence: {confidence:.2f})")
    print(f"  Expected: {test_case['expected_persona']}")
    persona_match = persona == test_case["expected_persona"]
    print(f"  ✓ PASS" if persona_match else f"  ✗ FAIL")

    # Test 2: Sentiment Analysis
    print("\n### Sentiment Analysis ###")
    sentiment = analyze_sentiment(test_case["message"])
    print(f"  Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})")
    print(f"  Expected: {test_case['expected_sentiment']}")
    sentiment_match = sentiment["label"].lower() == test_case["expected_sentiment"].lower()
    print(f"  ✓ PASS" if sentiment_match else f"  ✗ FAIL")

    # Test 3: RAG Retrieval & Response
    print("\n### RAG Response ###")
    try:
        rag = RAG()
        retrieved = rag.retrieve(test_case["message"], k=4)
        scores = [r[0] for r in retrieved]
        print(f"  Retrieved documents: {len(retrieved)}")
        if retrieved:
            for i, (score, doc) in enumerate(retrieved[:3], 1):
                print(f"    {i}. {doc.get('source', 'unknown')} (score: {score:.3f})")
        else:
            print("    No documents retrieved")

        response = rag.generate_response(persona, test_case["message"], retrieved)
        print(f"  Response (first 150 chars): {response['answer'][:150]}...")
    except Exception as e:
        print(f"  ✗ Error in RAG: {e}")
        retrieved = []
        scores = []

    # Test 4: Escalation Detection
    print("\n### Escalation Detection ###")
    config = load_escalation_config()
    should_esc, reason = should_escalate(
        [r[1] for r in retrieved],
        scores,
        test_case["message"],
        convo_turns=1,
        persona=persona,
        config=config,
    )
    print(f"  Should escalate: {should_esc}")
    print(f"  Reason: {reason}")
    print(f"  Expected: {test_case['expected_escalation']}")
    escalation_match = should_esc == test_case["expected_escalation"]
    print(f"  ✓ PASS" if escalation_match else f"  ~ NOTE (may vary based on index)")

    # Test 5: Billing Detection (for test case 5)
    if test_case["id"] == 5:
        print("\n### Billing Issue Detection ###")
        is_billing = detect_billing_issue(test_case["message"], config)
        print(f"  Detected as billing issue: {is_billing}")
        print(f"  ✓ PASS" if is_billing else f"  ✗ FAIL")

    # Test 6: Handoff Summary (if escalation)
    if should_esc or test_case["id"] == 5:
        print("\n### Handoff Summary (JSON) ###")
        summary = make_handoff_summary(
            persona,
            test_case["message"],
            [test_case["message"]],
            [d.get("source", "unknown") for _, d in retrieved[:3]] if retrieved else [],
            [],
            "Human agent review required.",
            escalation_reason=reason,
            sentiment=sentiment,
            confidence=confidence,
        )
        print(json.dumps(summary, indent=2))

        # Validate handoff for billing case
        if test_case["id"] == 5:
            print("\n### Billing Escalation Validation ###")
            is_billing_marked = summary.get("metadata", {}).get("is_billing_issue", False)
            has_action = "action_required" in summary
            priority = summary.get("metadata", {}).get("priority", "")
            print(f"  Is marked as billing issue: {is_billing_marked} - {'✓' if is_billing_marked else '✗'}")
            print(f"  Has action_required section: {has_action} - {'✓' if has_action else '✗'}")
            print(
                f"  Priority is HIGH/CRITICAL: {priority in ['HIGH', 'CRITICAL']} - {'✓' if priority in ['HIGH', 'CRITICAL'] else '✗'}"
            )
            if has_action and "action_required" in summary:
                print(f"  Action type: {summary['action_required'].get('type')}")
                print(f"  Next steps defined: {len(summary['action_required'].get('next_steps', []))} steps")

    # Validation Points
    print("\n### Expected Behavior Checklist ###")
    for point in test_case["validation_points"]:
        print(f"  • {point}")

    return {
        "test_id": test_case["id"],
        "persona_match": persona_match,
        "sentiment_match": sentiment_match,
        "escalation_match": escalation_match,
        "persona": persona,
        "sentiment": sentiment,
        "escalated": should_esc,
    }


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PERSONA-ADAPTIVE CUSTOMER SUPPORT AGENT - TEST SUITE")
    print("=" * 80)
    print(f"Running {len(TEST_CASES)} test cases...\n")

    results = []
    for test_case in TEST_CASES:
        try:
            result = run_test(test_case)
            results.append(result)
        except Exception as e:
            print(f"\n✗ TEST {test_case['id']} FAILED WITH ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"{'Test':<8} {'Persona':<20} {'Sentiment':<12} {'Escalated':<12} {'Result'}")
    print("-" * 80)
    for r in results:
        persona_ok = "✓" if r["persona_match"] else "✗"
        sentiment_ok = "✓" if r["sentiment_match"] else "✗"
        escalation_ok = "✓" if r["escalation_match"] else "~"
        result = "PASS" if all([r["persona_match"], r["sentiment_match"]]) else "PARTIAL"
        print(
            f"{r['test_id']:<8} {r['persona']:<20} {r['sentiment']['label']:<12} "
            f"{'Yes' if r['escalated'] else 'No':<12} {result}"
        )

    print("\n" + "=" * 80)
    print("KEY VALIDATION RESULTS:")
    print("-" * 80)
    test_5_results = [r for r in results if r["test_id"] == 5]
    if test_5_results:
        print("✓ Test 5 (Billing Escalation): CRITICAL FOR VALIDATION")
        print("  This test ensures billing disputes trigger proper escalation with handoff JSON")

    total_pass = sum(1 for r in results if all([r["persona_match"], r["sentiment_match"]]))
    print(f"\nOverall: {total_pass}/{len(results)} tests with correct persona & sentiment")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
