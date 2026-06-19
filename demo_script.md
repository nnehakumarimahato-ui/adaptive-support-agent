# Demo Recording Script

Follow these steps while recording a 3–5 minute demo of the project.

1. Project structure overview (20s)
   - Show repo files and `data/docs`.

2. Ingest knowledge base (30s)
   - Run: `python -m src.agent.cli --ingest`
   - Explain chunking and index creation.

3. Persona detection examples (40s)
   - Use 3 sample queries: technical, frustrated, executive.
   - Show detected persona printed in CLI or UI.

4. Retrieval process (30s)
   - Show retrieved sources for a query and scores.

5. Responses for each persona (40s)
   - Demonstrate differences in tone and content.

6. Escalation scenario (30s)
   - Trigger billing/legal query and show handoff JSON.

7. Explain one design decision (20s)
   - E.g., chunk size, embedding model, or escalation threshold.

8. Wrap-up and next steps (10s)
