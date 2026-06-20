import os
import pickle
from typing import Dict, List, Tuple

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_VECTOR = True
except Exception:
    HAS_VECTOR = False

from .classifier import classify_persona_llm
from .escalation import is_simple_interaction
from .ingest import load_documents, chunk_text

OPENAI_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
PERSONA_INSTRUCTIONS = {
    "Technical Expert": "Provide detailed technical analysis, root-cause reasoning, and step-by-step troubleshooting instructions.",
    "Frustrated User": "Be empathetic, calm, simple, reassuring, and clearly outline actionable next steps.",
    "Business Executive": "Be concise, impact-focused, minimize technical jargon, and include estimated resolution guidance.",
}


class RAG:
    def __init__(self, index_path: str = "data/index.faiss"):
        self.index_path = index_path
        meta_path = index_path + ".meta.pkl"
        # Prefer vector index when available, otherwise build an in-memory simple index
        if HAS_VECTOR and os.path.exists(index_path) and os.path.exists(meta_path):
            self.is_simple = False
            self.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.metadatas = meta["metadatas"]
            self.texts = meta["texts"]
            self.model_name = meta.get("model_name", DEFAULT_MODEL)
            # Lazy-load the SentenceTransformer to avoid heavy imports at module load time.
            self.model = None
        else:
            # Simple in-memory retriever: load and chunk documents directly
            self.is_simple = True
            docs = load_documents("data/docs")
            texts = []
            metadatas = []
            for d in docs:
                for c in chunk_text(d["text"]):
                    texts.append(c)
                    metadatas.append({"source": d["source"], "page": d.get("page")})
            self.texts = texts
            self.metadatas = metadatas
            self.model = None

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[float, Dict]]:
        results: List[Tuple[float, Dict]] = []
        if not self.is_simple:
            # Load the embedding model on first use to speed up startup.
            if self.model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    self.model = SentenceTransformer(self.model_name)
                except Exception:
                    raise RuntimeError("Failed to load embedding model. Ensure sentence-transformers is installed.")

            q_emb = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            D, I = self.index.search(q_emb, k)
            for score, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                metadata = self.metadatas[idx].copy()
                metadata["text"] = self.texts[idx]
                results.append((float(score), metadata))
            return results

        # Simple overlap-based retriever: score by token overlap
        def _clean_tokens(s: str):
            return [t.lower().strip(".,!?()[]{}\"'`") for t in s.split() if len(t) > 2]

        q_tokens = set(_clean_tokens(query))
        scored = []
        for i, txt in enumerate(self.texts):
            tokens = set(_clean_tokens(txt))
            overlap = len(q_tokens & tokens)
            scored.append((overlap, i))
        scored.sort(reverse=True)
        for score, idx in scored[:k]:
            if score <= 0:
                continue
            metadata = self.metadatas[idx].copy()
            metadata["text"] = self.texts[idx]
            results.append((float(score), metadata))
        return results

    def _format_source_texts(self, retrieved: List[Tuple[float, Dict]]) -> Tuple[str, List[str]]:
        source_texts = []
        sources = []
        for score, chunk in retrieved:
            source = chunk.get("source", "unknown")
            page = chunk.get("page")
            sources.append(f"{source}{f' (page {page})' if page else ''}".strip())
            header = f"Source: {source}"
            if page:
                header += f" | Page: {page}"
            source_texts.append(f"{header}\n{chunk.get('text', '').strip()}")
        return "\n\n---\n\n".join(source_texts), list(dict.fromkeys(sources))

    def _fallback_answer(self, query: str) -> Dict:
        if is_simple_interaction(query):
            answer = (
                "Hello there! I didn't find any matching knowledge base articles for this short message. "
                "Please describe your issue in a few sentences so I can assist you further."
            )
            return {"persona": "", "answer": answer, "sources": []}

        answer = (
            "I couldn't find a matching article in the knowledge base for this issue. "
            "This should be escalated to a human support agent with a handoff summary."
        )
        return {"persona": "", "answer": answer, "sources": []}

    def _openai_answer(self, persona: str, query: str, persona_instruction: str, combined_text: str) -> str:
        import openai

        openai.api_key = os.getenv(OPENAI_KEY_ENV)
        system = (
            "You are a customer support assistant. Use ONLY the provided source text to answer. "
            "Do NOT hallucinate or invent details. If the answer is not present in the source text, say that the knowledge base does not contain enough details and recommend escalation. "
            "Keep the tone aligned to the detected persona."
        )
        prompt = (
            f"Persona: {persona}\n"
            f"Persona guidance: {persona_instruction}\n"
            f"User query: {query}\n\n"
            f"Sources:\n{combined_text}\n\n"
            "Answer strictly from sources and include a short sources list."
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def _default_answer(self, persona: str, combined_text: str) -> str:
        if persona == "Technical Expert":
            return (
                "I found relevant technical guidance in the knowledge base. Review the details below and follow the recommended steps:\n\n"
                + combined_text
            )
        if persona == "Frustrated User":
            return (
                "I understand how frustrating this is. Here is the information I found and the actions we can take next:\n\n"
                + combined_text
            )
        return (
            "Here is a concise summary of the relevant information and a recommended next step:\n\n"
            + combined_text
        )

    def generate_response(self, persona: str, query: str, retrieved: List[Tuple[float, Dict]]) -> Dict:
        if not retrieved:
            response = self._fallback_answer(query)
            return {"persona": persona, "answer": response["answer"], "sources": response["sources"]}

        combined_text, sources = self._format_source_texts(retrieved)
        persona_instruction = PERSONA_INSTRUCTIONS.get(persona, "Provide a helpful support response.")

        openai_key = os.getenv(OPENAI_KEY_ENV)
        answer = None
        if openai_key is not None:
            try:
                cls = classify_persona_llm(query)
                persona = cls.get("persona", persona)
            except Exception:
                pass

            try:
                answer = self._openai_answer(persona, query, persona_instruction, combined_text)
            except Exception:
                answer = (
                    "I found relevant documentation, but there was an issue generating a final summary. "
                    "Use the retrieved source text directly."
                )

        if answer is None:
            answer = self._default_answer(persona, combined_text)

        return {"persona": persona, "answer": answer, "sources": sources}
