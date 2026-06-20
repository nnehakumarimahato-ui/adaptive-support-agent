# Demo Recording Checklist and Sample Commands

Follow these steps while recording a 3–8 minute demo of the project. Use the commands below in a terminal.

1. Project structure overview (15-25s)
   - Show repository files and `data/docs` contents.

2. Ingest knowledge base (30-45s)
   - Command:

```bash
python -m src.agent.cli --ingest
```

   - Explain chunking and index creation. Verify `data/index.faiss` exists.

3. Run scripted demo (45-60s)
   - Command:

```bash
python scripts/demo_full_run.py
```

   - Show printed transcript and `artifacts/demo_transcript.json`.

4. Persona detection examples (30-40s)
   - Use three example queries in CLI (or `examples.py`):

```bash
python examples.py
```

5. Retrieval process (30s)
   - Show retrieved sources and scores printed by the CLI or Streamlit UI.

6. Responses for each persona (40-60s)
   - Demonstrate differences in tone for `Technical Expert`, `Frustrated User`, and `Business Executive`.

7. Escalation scenario (30-40s)
   - Send a billing/legal-sensitive query and show handoff JSON printed by the CLI or saved in `artifacts`.

8. Human handoff summary (20-30s)
   - Open the JSON handoff summary and explain each field.

9. Explain a technical design decision (20-30s)
   - Suggest talking points: chunk size choice, embedding model selection, or why FAISS was chosen.

10. Wrap-up (10-20s)
    - Mention known limitations and next steps.

**Helpful file locations**
- `src/agent/ingest.py` — ingestion and chunking
- `src/agent/rag.py` — retrieval and response generation
- `src/agent/classifier.py` — persona classification
- `src/agent/escalation.py` — escalation rules and handoff summary
- `data/docs` — knowledge base
- `artifacts/demo_transcript.json` — generated demo transcript

**Notes**
- If you want LLM-based responses, set `OPENAI_API_KEY` in your environment before running ingestion or chat.
- Do not commit API keys to Git.
