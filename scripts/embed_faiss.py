#!/usr/bin/env python3
"""
embed_faiss.py
Build embeddings for corpus/chunks/chunks.jsonl and create a FAISS index.

Outputs:
  - corpus/index/faiss.index
  - corpus/index/chunk_meta.jsonl

Usage:
  python3 scripts/embed_faiss.py \
    corpus/chunks/chunks.jsonl \
    corpus/index/faiss.index \
    corpus/index/chunk_meta.jsonl
"""

import os, sys, json
import numpy as np

import faiss  # pip install faiss-cpu
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64

def read_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main(in_chunks_jsonl: str, out_faiss: str, out_meta_jsonl: str):
    os.makedirs(os.path.dirname(out_faiss) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_meta_jsonl) or ".", exist_ok=True)

    chunks = list(read_chunks(in_chunks_jsonl))
    if not chunks:
        print("No chunks found.")
        return 1

    texts = [c.get("text", "") for c in chunks]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(texts)} chunks (batch={BATCH_SIZE}) ...")
    emb = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    emb = np.asarray(emb, dtype="float32")
    dim = emb.shape[1]

    print(f"Building FAISS index (cosine via inner product), dim={dim}")
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    print(f"Writing index: {out_faiss}")
    faiss.write_index(index, out_faiss)

    print(f"Writing metadata: {out_meta_jsonl}")
    tmp = out_meta_jsonl + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for c in chunks:
            meta = {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "title": c.get("title"),
                "url_or_doi": c.get("url_or_doi"),
                "tags": c.get("tags", []),
                "text_path": c.get("text_path"),
                "n_tokens_est": c.get("n_tokens_est"),
                "preview": (c.get("text", "")[:400] + ("..." if len(c.get("text", "")) > 400 else ""))
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    os.replace(tmp, out_meta_jsonl)

    print("Done.")
    print("Index size:", index.ntotal)
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 scripts/embed_faiss.py <chunks.jsonl> <faiss.index> <chunk_meta.jsonl>")
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
