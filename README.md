Persona-Adaptive Customer Support Agent
======================================

Minimal demo for Adsparkx AI assignment: persona-aware support agent using a local RAG pipeline.

Quick start
-----------

1. Create a Python 3.11 venv and install deps:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Ingest knowledge base (first run):

```bash
python -m src.agent.cli --ingest
```

3. Start CLI chat:

```bash
python -m src.agent.cli
```

Streamlit UI
-----------

Start the web UI with:

```bash
streamlit run app.py
```

Run examples
------------

To run the five example queries used in the assignment:

```bash
python examples.py
```

Recording guidance
------------------

Follow `demo_script.md` to create the required 3–5 minute screen recording.

Environment variables
---------------------
- `OPENAI_API_KEY`: Optional. If present, the agent will use the OpenAI ChatCompletion API for response synthesis and persona classification.
- `ESCALATION_MIN_SCORE`: Optional. Float threshold for vector retrieval confidence below which the agent escalates. Defaults to `0.24`.
- `ESCALATION_MAX_TURNS`: Optional. Integer number of conversational turns after which the agent escalates by default. Defaults to `3`.
- `ESCALATION_SENSITIVE_KEYWORDS`: Optional. Comma-separated words such as `billing,refund,legal` that force escalation when detected.

Notes
-----
- Add an `OPENAI_API_KEY` environment variable to enable LLM-based answer synthesis. Without it, the agent uses a template-based response generator that still relies on retrieved content.
- Fill `data/docs` with additional realistic support documentation for stronger retrieval quality.
