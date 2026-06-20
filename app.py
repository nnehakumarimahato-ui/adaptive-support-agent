import streamlit as st
from dotenv import load_dotenv

from src.agent.classifier import classify_persona_llm
from src.agent.escalation import load_escalation_config, make_handoff_summary, should_escalate
from src.agent.feedback import log_feedback, log_interaction
from src.agent.ingest import build_index
from src.agent.rag import RAG
from src.agent.sentiment import analyze_sentiment

load_dotenv()

st.set_page_config(page_title="Persona-Adaptive Support Agent")

st.title("Persona-Adaptive Customer Support Agent")

if st.sidebar.button("Ingest / Rebuild Index"):
    with st.spinner("Ingesting docs and building index..."):
        build_index("data/docs")
    st.success("Index built")

query = st.text_area("User message", height=120)
k = st.sidebar.slider("Top-k retrieval", 1, 8, 4)
config = load_escalation_config()

if st.button("Send") and query.strip():
    persona_info = classify_persona_llm(query)
    persona = persona_info["persona"]
    confidence = persona_info.get("confidence", 0.6)

    try:
        rag = RAG()
    except Exception:
        st.error("Index not found. Click 'Ingest / Rebuild Index' first.")
        st.stop()

    retrieved = rag.retrieve(query, k=k)
    scores = [r[0] for r in retrieved]
    formatted_sources = []
    for _, record in retrieved:
        source = record.get("source", "unknown")
        page = record.get("page")
        if page:
            formatted_sources.append(f"{source} (page {page})")
        else:
            formatted_sources.append(source)

    st.markdown(f"**Detected persona:** {persona} (confidence={confidence:.2f})")
    st.markdown("**Retrieved sources:**")
    for s, sc in zip(formatted_sources, scores):
        st.write(f"- {s} (score={sc:.3f})")

    sentiment = analyze_sentiment(query)
    resp = rag.generate_response(persona, query, retrieved)
    escalate, reason = should_escalate(
        [r[1] for r in retrieved],
        scores,
        query,
        convo_turns=1,
        config=config,
    )
    st.markdown(f"**Sentiment:** {sentiment['label']} (score={sentiment['score']:.1f})")
    st.markdown(f"**Escalation:** {escalate} ({reason})")
    st.markdown("---")
    st.markdown("**Agent response**")
    st.write(resp["answer"])

    if escalate:
        summary = make_handoff_summary(
            persona,
            query,
            [query],
            formatted_sources,
            [],
            "A human agent should review this issue and follow the provided sources.",
            escalation_reason=reason,
        )
        st.markdown("**Handoff Summary**")
        st.json(summary)
    elif reason == "greeting":
        st.info("No escalation is required for this greeting. Please describe your issue in more detail.")
    elif not formatted_sources:
        st.warning("No matching source documents were found. Try adding more context or a clearer description.")

    feedback = st.text_area("Optional feedback for this interaction", height=80)
    if st.button("Submit feedback"):
        if feedback.strip():
            log_feedback(query, persona, confidence, max(scores) if scores else 0.0, sentiment, escalate, reason, feedback)
            st.success("Thank you! Feedback recorded.")
        else:
            st.warning("Please enter feedback before submitting.")
