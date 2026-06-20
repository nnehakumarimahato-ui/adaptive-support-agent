Persona-Adaptive Customer Support Agent
======================================

Minimal demo for the Adsparkx AI assignment: a persona-aware customer support agent that uses a local RAG pipeline, persona detection, adaptive response generation, and configurable escalation with a human handoff summary.

**Quick Start**
- **Create venv & install deps:**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

- **Ingest knowledge base (first run):**

```bash
python -m src.agent.cli --ingest
```

- **Run CLI chat:**

```bash
python -m src.agent.cli
```

- **Run examples script:**

```bash
python examples.py
```

- **Streamlit UI (optional):**

```bash
streamlit run app.py
```

**Project Deliverables**
- **Source code:** repository containing the `src/agent` package and helpers.
- **Knowledge base:** `data/docs` (10–20 articles including `password_reset_guide.pdf`).
- **Demo script:** `demo_script.md` for recording the 3–8 minute walkthrough.

**Tech Stack**
- **Language:** Python 3.11+
- **Libraries:**
	- **sentence-transformers:** 2.2.2
	- **faiss-cpu:** 1.7.4
	- **PyPDF2:** 3.0.1
	- **openai:** 0.27.8 (optional; enable via `OPENAI_API_KEY`)
	- **rich:** >=13.3.6
	- **python-dotenv:** 1.0.0
	- **streamlit:** 1.30.0 (optional UI)
	- **langchain:** 0.0.334 (optional helpers)

**Architecture**
- Flow: User Query → Persona Detection → Retrieval (RAG) → Response Generation → Escalation Check → Human Handoff

```mermaid
flowchart LR
	U[User Query] --> P[Persona Detection]
	P --> R[Retriever (FAISS + embeddings)]
	R --> G[Response Generator (LLM or template)]
	G --> E[Escalation Logic]
	E --> H[Human Handoff Summary]
```

**Persona Detection Strategy**
- **Classification method:** Lightweight two-stage approach.
	- Heuristic fallback in `src/agent/persona.py` (keyword / rule-based) for offline usage.
	- Optional LLM classifier in `src/agent/classifier.py` using the `OPENAI_API_KEY` to call ChatCompletion and return `{persona, confidence, reasoning}` when available.
- **Prompt design:** System instructs the LLM to return JSON with keys `persona`, `confidence` (0–1), and `reasoning`.
- **Rules used:** Keyword lists for `Technical Expert`, `Frustrated User`, and `Business Executive`; fallbacks use message length and presence of polite/urgent tokens.

**RAG Pipeline Design**
- **Document loading:** `src/agent/ingest.py` reads `data/docs` (PDFs via PyPDF2, Markdown/TXT files directly).
- **Chunking strategy:** Default chunk size ~800 chars with 100-char overlap; falls back to a simple splitter if LangChain splitter is not available.
- **Embedding model:** `sentence-transformers` model `all-MiniLM-L6-v2` (configurable in `ingest.build_index`).
- **Vector DB:** FAISS (local `data/index.faiss`), using cosine-similarity (normalized vectors).
- **Retrieval:** Top-k nearest neighbors (default k=4). Chunk metadata stores `source` and `page`.

**Adaptive Response Generation**
- Responses are generated in `src/agent/rag.py`.
- **Persona instructions:**
	- Technical Expert: detailed, root-cause analysis, step-by-step.
	- Frustrated User: empathetic, simple, reassuring, action-oriented.
	- Business Executive: concise, impact-focused, minimal jargon, estimated guidance.
- If `OPENAI_API_KEY` is present, the agent composes a strict prompt telling the LLM to answer ONLY from the retrieved sources and to recommend escalation when the KB lacks detail. Otherwise, a template-based summary is used that embeds retrieved content.

**Escalation Logic**
- Implemented in `src/agent/escalation.py` with configurable thresholds.
- **Triggers:**
	- No retrieval results → escalate (`no_retrieval`).
	- Top retrieval score below `ESCALATION_MIN_SCORE` → escalate (`low_confidence`).
	- Presence of sensitive keywords (`billing, legal, refund, account locked, password, security`) → escalate (`sensitive_issue`).
	- Urgent language with low confidence → escalate (`urgent_low_confidence`).
	- Repeated interactions exceeding `ESCALATION_MAX_TURNS` → escalate (`repeat_interactions`).
- **Configurable via env vars:** `ESCALATION_MIN_SCORE`, `ESCALATION_MAX_TURNS`, `ESCALATION_SENSITIVE_KEYWORDS`.

**Human Handoff Summary**
- When escalation occurs the agent produces a JSON summary via `make_handoff_summary` including:
	- `persona`
	- `issue_summary` (user message)
	- `conversation_history`
	- `documents_used` (sources)
	- `attempted_steps`
	- `recommendation`
	- `escalation_reason`

Example handoff snippet produced by the system:

```json
{
	"persona": "Frustrated User",
	"issue_summary": "Unable to reset password",
	"documents_used": ["password_reset_guide.pdf"],
	"attempted_steps": ["Password reset", "Cleared browser cache"],
	"recommendation": "Investigate account lock status",
	"escalation_reason": "sensitive_issue"
}
```

**Setup Instructions**
- Create and activate venv, then install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

- Ingest docs and build vector index:

```bash
python -m src.agent.cli --ingest
```

- Start command-line chat:

```bash
python -m src.agent.cli
```

- (Optional) Run the Streamlit UI:

```bash
streamlit run app.py
```

**Environment Variables**
- **`OPENAI_API_KEY`**: Optional. If set, enables LLM-based persona classification and response synthesis. Do NOT commit this key.
- **`ESCALATION_MIN_SCORE`**: Optional float; default `0.24`.
- **`ESCALATION_MAX_TURNS`**: Optional int; default `3`.
- **`ESCALATION_SENSITIVE_KEYWORDS`**: Optional comma-separated words; default includes `billing,charge,refund,legal,account locked,password,security,downtime`.

**Example Queries**
- "Can you explain the API authentication failure and provide error details?" — expect `Technical Expert`.
- "I've tried everything and nothing works! The dashboard keeps failing." — expect `Frustrated User`.
- "How does this outage impact our SLAs and when will it be resolved?" — expect `Business Executive`.
- "My billing statement shows duplicate charges, I need a refund now." — triggers escalation.
- "Where do I find the password reset guide?" — retrieves `password_reset_guide.pdf`.

**Known Limitations & Future Work**
- Current persona classifier uses heuristics as a fallback; LLM classification is optional and requires `OPENAI_API_KEY`.
- No persistent multi-turn memory beyond the running session; adding a small DB (SQLite) would enable richer conversation state.
- Response grounding relies on the KB coverage — expand `data/docs` for better accuracy.
- Optional improvements: sentiment scoring, LangGraph workflows, feedback collection, and a human approval workflow for escalations.

**Files of Interest**
- `src/agent/ingest.py` — document ingestion, chunking, and index creation
- `src/agent/rag.py` — retrieval and response generation
- `src/agent/classifier.py` — persona classification (LLM + heuristic fallback)
- `src/agent/escalation.py` — escalation rules and handoff summary
- `data/docs` — knowledge base documents (includes at least one PDF)

---

If you'd like, I can now run the interactive CLI (`python -m src.agent.cli`) and capture a short session for your demo recording.
