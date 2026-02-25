#!/usr/bin/env python3
"""
retrieve.py (improved + topic guardrails)

What it does:
- Load FAISS index + chunk metadata
- Embed the query (SentenceTransformers)
- Retrieve a larger candidate pool from FAISS
- Infer context (pregnant/postpartum/none) and filter using ctx:* tags
- Apply topic guardrails (avoid the huge STI guideline doc unless query is STI-like)
- Enforce diversity (max per doc_id, max per source)
- Print top-k results with URLs + chunk_id (acts as citations)

Usage:
  python3 scripts/retrieve.py \
    corpus/index/faiss.index \
    corpus/index/chunk_meta.jsonl \
    "pelvic pain heavy bleeding painful periods" \
    8
"""

import sys
import json
from collections import defaultdict

import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# Retrieval config
# -----------------------------
CANDIDATES = 80                    # pull this many then filter down
MAX_PER_DOC = 1                    # diversity: max chunks per doc_id
MAX_PER_SOURCE = 4                 # diversity: max chunks per source (NHS/Journal/etc.)
EXCLUDE_PREGNANCY_IF_NOT_MENTIONED = True

# Your corpus is dominated by this doc (hundreds of chunks)
DOMINANT_STI_DOC_ID = "sexually_transmitted_infections_treatment_guidelines_2021"

# Guardrail switches
SUPPRESS_DOMINANT_STI_DOC_IF_NOT_STI_QUERY = True
SUPPRESS_ALL_SEXUAL_HEALTH_IF_NOT_STI_QUERY = False  # set True if you want stricter behavior


def load_meta(meta_jsonl: str):
    meta = []
    with open(meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def get_source_from_tags(tags):
    for t in tags or []:
        if t.startswith("source:"):
            return t.split(":", 1)[1]
    return "unknown"


def infer_context(query: str):
    """
    Returns: "pregnant", "postpartum", or None
    """
    q = query.lower()

    pregnant_signals = [
        "pregnan", "trimester", "weeks pregnant", "week pregnant",
        "fetal", "foetal", "prenatal", "antenatal", "baby",
        "labour", "labor",
    ]
    postpartum_signals = ["postpartum", "after birth", "after delivery", "newborn"]

    if any(s in q for s in pregnant_signals):
        return "pregnant"
    if any(s in q for s in postpartum_signals):
        return "postpartum"
    return None


def context_ok(tags, ctx):
    tagset = set(tags or [])
    is_preg_chunk = "ctx:pregnant" in tagset
    is_post_chunk = "ctx:postpartum" in tagset

    # Query not pregnancy-related: optionally exclude pregnancy-tagged chunks
    if ctx is None:
        if EXCLUDE_PREGNANCY_IF_NOT_MENTIONED and is_preg_chunk:
            return False
        return True

    # Query pregnancy-related: allow pregnancy + neutral; exclude postpartum-only
    if ctx == "pregnant":
        return not is_post_chunk

    # Query postpartum-related: allow postpartum + neutral; exclude pregnancy-only
    if ctx == "postpartum":
        return not is_preg_chunk

    return True


def mentions_sexual_health(query: str) -> bool:
    """
    Decide if query is likely STI / sexual-health related.
    If False, we can suppress STI-heavy doc/chunks to reduce noise.
    """
    q = query.lower()
    signals = [
        "sti", "std", "sexually transmitted",
        "chlamydia", "gonorrhea", "gonorrhoea", "syphilis",
        "trich", "trichomon", "mycoplasma",
        "genital", "vaginal discharge", "discharge",
        "itch", "itching", "burning", "burning when peeing",
        "pain when peeing", "urethra", "cervicitis",
        "pid", "pelvic inflammatory",
        "unprotected", "new partner", "condom",
    ]
    return any(s in q for s in signals)


def is_sexual_health_chunk(tags) -> bool:
    tagset = set(tags or [])
    return ("topic:sexual_health" in tagset) or ("possible:sti" in tagset)


def main(index_path: str, meta_path: str, query: str, k_final: int):
    index = faiss.read_index(index_path)
    meta = load_meta(meta_path)

    ctx = infer_context(query)
    sti_query = mentions_sexual_health(query)

    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    # Retrieve a larger pool then filter
    scores, idxs = index.search(q_emb, max(k_final, CANDIDATES))
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    selected = []
    per_doc = defaultdict(int)
    per_source = defaultdict(int)

    for i, s in zip(idxs, scores):
        if i < 0 or i >= len(meta):
            continue

        m = meta[i]
        tags = m.get("tags", [])
        doc_id = m.get("doc_id") or ""
        source = get_source_from_tags(tags)

        # 1) Context filter
        if not context_ok(tags, ctx):
            continue

        # 2) Topic guardrails (fix corpus imbalance)
        if not sti_query:
            if SUPPRESS_DOMINANT_STI_DOC_IF_NOT_STI_QUERY and doc_id == DOMINANT_STI_DOC_ID:
                continue
            if SUPPRESS_ALL_SEXUAL_HEALTH_IF_NOT_STI_QUERY and is_sexual_health_chunk(tags):
                continue

        # 3) Diversity filters
        if per_doc[doc_id] >= MAX_PER_DOC:
            continue
        if per_source[source] >= MAX_PER_SOURCE:
            continue

        selected.append((m, s))
        per_doc[doc_id] += 1
        per_source[source] += 1

        if len(selected) >= k_final:
            break

    print("\nQUERY:", query)
    print("INFERRED_CONTEXT:", ctx)
    print("STI_QUERY:", sti_query)
    print("-" * 80)

    for rank, (m, s) in enumerate(selected, start=1):
        print(f"[{rank}] score={s:.4f}  {m.get('title')}")
        print(f"    url: {m.get('url_or_doi')}")
        print(f"    chunk_id: {m.get('chunk_id')}  doc_id: {m.get('doc_id')}")
        if m.get("tags"):
            print(f"    tags: {', '.join(m['tags'])}")
        print(f"    preview: {m.get('preview')}")
        print("-" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 scripts/retrieve.py <faiss.index> <chunk_meta.jsonl> <query> [k]")
        sys.exit(1)

    index_path = sys.argv[1]
    meta_path = sys.argv[2]
    query = sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) >= 5 else 8

    main(index_path, meta_path, query, k)
