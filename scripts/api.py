from fastapi import FastAPI
from pydantic import BaseModel

from scripts.answers import load_meta, load_faiss_index, retrieve, make_answer
from sentence_transformers import SentenceTransformer

INDEX_PATH = "corpus/index/faiss.index"
META_PATH = "corpus/index/chunk_meta.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="Women's Health RAG", version="0.1")

# Load once at startup (fast requests)
index = load_faiss_index(INDEX_PATH)
meta = load_meta(META_PATH)
embed_model = SentenceTransformer(MODEL_NAME)

class AnswerRequest(BaseModel):
    query: str
    k: int = 6

@app.post("/answer")
def answer(req: AnswerRequest):
    hits, debug = retrieve(index, meta, embed_model, req.query, req.k)
    ans = make_answer(req.query, hits)
    return {
        "query": req.query,
        "answer": ans,
        "citations": [
            {
                "title": h["title"],
                "url": h["url"],
                "chunk_id": h["chunk_id"],
                "doc_id": h["doc_id"],
                "score": h["score"],
            }
            for h in hits
        ],
        "n_hits": len(hits),
        "debug": debug,
    }
