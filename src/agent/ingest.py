import os
import pickle
from typing import List, Dict

# Delay importing heavy ML libraries until they're needed to avoid import-time failures
import faiss
from PyPDF2 import PdfReader
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None


def load_documents(folder: str) -> List[Dict]:
    docs = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if os.path.isdir(path):
            continue
        if fname.lower().endswith(".pdf"):
            try:
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    docs.append({"source": fname, "page": i, "text": text})
            except Exception:
                continue
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            docs.append({"source": fname, "page": None, "text": text})
    return docs


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    text = text.replace("\n", "\n")
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap)
        return splitter.split_text(text)

    # Fallback simple splitter
    text = text.replace("\n", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap if end - overlap > start else end
    return chunks


def build_index(docs_folder: str, model_name: str = "all-MiniLM-L6-v2", out_index_path: str = "data/index.faiss"):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise RuntimeError("sentence-transformers is required to build the index. Install requirements and try again.")

    model = SentenceTransformer(model_name)
    docs = load_documents(docs_folder)
    texts = []
    metadatas = []
    for d in docs:
        for c in chunk_text(d["text"]):
            texts.append(c)
            metadatas.append({"source": d["source"], "page": d.get("page")})

    if not texts:
        raise RuntimeError("No documents found in docs folder")

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # normalize for cosine similarity
    import numpy as np
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    os.makedirs(os.path.dirname(out_index_path), exist_ok=True)
    faiss.write_index(index, out_index_path)
    with open(out_index_path + ".meta.pkl", "wb") as f:
        pickle.dump({"metadatas": metadatas, "texts": texts, "model_name": model_name}, f)
    return out_index_path
