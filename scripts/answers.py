#!/usr/bin/env python3
"""
answers.py
- Loads FAISS + metadata
- Retrieves top-k (context + topic guardrails + diversity)
- Builds a grounded prompt
- Uses a local LLM (Ollama) ONLY to rewrite/phrase using provided sources
- Outputs JSON: {query, answer, citations, debug}

CLI:
  python3 scripts/answers.py corpus/index/faiss.index corpus/index/chunk_meta.jsonl \
    --query "pelvic pain heavy bleeding painful periods for 3 months" \
    --k 6 \
    --out corpus/answers/answers.jsonl
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss  # type: ignore
import requests
from sentence_transformers import SentenceTransformer

MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.0"))


# -----------------------------
# Embedding model
# -----------------------------
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Retrieval config
# -----------------------------
CANDIDATES = int(os.getenv("RAG_CANDIDATES", "80"))
MAX_PER_DOC = int(os.getenv("RAG_MAX_PER_DOC", "1"))
MAX_PER_SOURCE = int(os.getenv("RAG_MAX_PER_SOURCE", "4"))
EXCLUDE_PREGNANCY_IF_NOT_MENTIONED = os.getenv("RAG_EXCLUDE_PREG_IF_NOT_MENTIONED", "1") == "1"

DOMINANT_STI_DOC_ID = os.getenv(
    "RAG_DOMINANT_STI_DOC_ID", "sexually_transmitted_infections_treatment_guidelines_2021"
)
SUPPRESS_DOMINANT_STI_DOC_IF_NOT_STI_QUERY = os.getenv("RAG_SUPPRESS_STI_DOC", "1") == "1"
SUPPRESS_ALL_SEXUAL_HEALTH_IF_NOT_STI_QUERY = os.getenv("RAG_SUPPRESS_SEXUAL_HEALTH", "0") == "1"

# Optional: drop very weak hits
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.0"))

# -----------------------------
# LLM config (Ollama)
# -----------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "60"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

# -----------------------------
# Caches (important for API speed)
# -----------------------------
_EMBED_MODEL: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(MODEL_NAME)
    return _EMBED_MODEL


# -----------------------------
# Loaders
# -----------------------------
def load_meta(meta_jsonl: str) -> List[Dict[str, Any]]:
    meta: List[Dict[str, Any]] = []
    with open(meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return meta


def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)


def _get_source_from_tags(tags) -> str:
    for t in tags or []:
        if isinstance(t, str) and t.startswith("source:"):
            return t.split(":", 1)[1]
    return "unknown"


# -----------------------------
# Heuristics / filters
# -----------------------------
def infer_context(query: str) -> Optional[str]:
    q = query.lower()
    pregnant_signals = [
        "pregnan", "trimester", "weeks pregnant", "week pregnant",
        "fetal", "foetal", "prenatal", "antenatal", "baby", "labour", "labor"
    ]
    postpartum_signals = ["postpartum", "after birth", "after delivery", "newborn"]

    if any(s in q for s in pregnant_signals):
        return "pregnant"
    if any(s in q for s in postpartum_signals):
        return "postpartum"
    return None


def context_ok(tags, ctx: Optional[str]) -> bool:
    tagset = set(tags or [])
    is_preg_chunk = "ctx:pregnant" in tagset
    is_post_chunk = "ctx:postpartum" in tagset

    if ctx is None:
        if EXCLUDE_PREGNANCY_IF_NOT_MENTIONED and is_preg_chunk:
            return False
        return True

    if ctx == "pregnant":
        return not is_post_chunk
    if ctx == "postpartum":
        return not is_preg_chunk

    return True


def mentions_sexual_health(query: str) -> bool:
    q = query.lower()
    signals = [
        "sti", "std", "sexually transmitted",
        "chlamydia", "gonorrhea", "gonorrhoea", "syphilis",
        "trich", "trichomon", "mycoplasma",
        "genital", "vaginal discharge", "discharge",
        "itch", "itching",
        "burning", "burning when peeing", "pain when peeing",
        "urethra", "cervicitis",
        "pid", "pelvic inflammatory",
        "unprotected", "new partner", "condom",
    ]
    return any(s in q for s in signals)


def is_sexual_health_chunk(tags) -> bool:
    tagset = set(tags or [])
    return ("topic:sexual_health" in tagset) or ("possible:sti" in tagset)


# -----------------------------
# Snippet cleaning (improves LLM answers a lot)
# -----------------------------
_BOILERPLATE_PATTERNS = [
    r"Skip to main content",
    r"Skip navigation",
    r"Official websites use \.gov.*?Secure \.gov websites use HTTPS",
    r"Cookies.*?Accept cookies",  # generic
]

def _clean_snippet(text: str) -> str:
    t = (text or "").replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    for pat in _BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
        t = re.sub(r"\s+", " ", t)
    return t.strip()


def _snippet_from_meta(m: Dict[str, Any], limit: int = 900) -> str:
    txt = m.get("preview") or m.get("text") or ""
    txt = _clean_snippet(str(txt))
    return txt[:limit]


# -----------------------------
# Retrieval (exported)
# -----------------------------
def retrieve(
    index,
    meta: List[Dict[str, Any]],
    embed_model: SentenceTransformer,
    query: str,
    k_final: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ctx = infer_context(query)
    sti_query = mentions_sexual_health(query)

    q_emb = embed_model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, idxs = index.search(q_emb, max(k_final, CANDIDATES))
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    def _select(relax_sti_doc: bool, relax_sexual_topic: bool) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        per_doc = defaultdict(int)
        per_source = defaultdict(int)

        for i, s in zip(idxs, scores):
            if i < 0 or i >= len(meta):
                continue
            if MIN_SCORE and s < MIN_SCORE:
                continue

            m = meta[i]
            tags = m.get("tags", [])
            doc_id = m.get("doc_id") or ""
            source = _get_source_from_tags(tags)

            if not context_ok(tags, ctx):
                continue

            if not sti_query:
                if SUPPRESS_DOMINANT_STI_DOC_IF_NOT_STI_QUERY and not relax_sti_doc and doc_id == DOMINANT_STI_DOC_ID:
                    continue
                if SUPPRESS_ALL_SEXUAL_HEALTH_IF_NOT_STI_QUERY and not relax_sexual_topic and is_sexual_health_chunk(tags):
                    continue

            if per_doc[doc_id] >= MAX_PER_DOC:
                continue
            if per_source[source] >= MAX_PER_SOURCE:
                continue

            selected.append({
                "title": m.get("title"),
                "url": m.get("url_or_doi"),
                "chunk_id": m.get("chunk_id"),
                "doc_id": m.get("doc_id"),
                "score": float(s),
                "tags": tags,
                "snippet": _snippet_from_meta(m, limit=900),
            })

            per_doc[doc_id] += 1
            per_source[source] += 1

            if len(selected) >= k_final:
                break

        return selected

    hits = _select(relax_sti_doc=False, relax_sexual_topic=False)

    # fallback pass if too few hits (relax only the dominant STI doc suppression)
    if len(hits) < k_final and not mentions_sexual_health(query):
        more = _select(relax_sti_doc=True, relax_sexual_topic=False)
        seen = {h["chunk_id"] for h in hits}
        for h in more:
            if h["chunk_id"] not in seen:
                hits.append(h)
                seen.add(h["chunk_id"])
            if len(hits) >= k_final:
                break

    debug = {
        "ctx": ctx,
        "sti_query": sti_query,
        "n_hits": len(hits),
        "candidates": CANDIDATES,
        "max_per_doc": MAX_PER_DOC,
        "max_per_source": MAX_PER_SOURCE,
        "min_score": MIN_SCORE,
        "suppress_sti_doc": SUPPRESS_DOMINANT_STI_DOC_IF_NOT_STI_QUERY,
        "suppress_sexual_health": SUPPRESS_ALL_SEXUAL_HEALTH_IF_NOT_STI_QUERY,
    }
    return hits, debug


# -----------------------------
# Ollama health + call
# -----------------------------
def ollama_healthcheck() -> bool:
    # /api/tags returns 200 if server is up
    try:
        base = OLLAMA_URL.rsplit("/api/", 1)[0]
        r = requests.get(base + "/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# -----------------------------
# Grounded prompt + Ollama
# -----------------------------
def _build_grounded_prompt(query: str, hits: List[Dict[str, Any]]) -> str:
    sources_block = []
    for idx, h in enumerate(hits, start=1):
        sources_block.append(
            f"[{idx}] TITLE: {h.get('title')}\n"
            f"URL: {h.get('url')}\n"
            f"CHUNK_ID: {h.get('chunk_id')}\n"
            f"SNIPPET: {h.get('snippet')}\n"
        )

    sources_text = "\n".join(sources_block)

    # More constrained output format -> less hallucination
    return f"""
You are rewriting an educational answer using ONLY the SOURCES below.

Hard rules:
- Use ONLY facts stated in the sources.
- If a detail is not in the sources, write: "The sources do not specify."
- Do NOT invent diagnoses, causes, tests, or treatments.
- Do NOT add medication advice beyond what sources explicitly mention.
- Cite sources as [1], [2], etc. for every factual sentence.
- Keep it short and structured.

Output format:
1) Summary (2-4 sentences)
2) What it could relate to (bullets, non-diagnostic, grounded)
3) When to seek urgent care (bullets)
4) Sources used (list each citation number with URL and chunk_id)

User query:
{query}

SOURCES:
{sources_text}
""".strip()


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": OLLAMA_TEMPERATURE},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def make_answer(query: str, hits: List[Dict[str, Any]], model: str = OLLAMA_MODEL) -> str:
    if not hits:
        return "I couldn't find relevant sources for this query in the current index."

    prompt = _build_grounded_prompt(query, hits)

    if not ollama_healthcheck():
        # no hallucination fallback
        lines = [
            "### Answer (fallback)",
            "",
            "Ollama does not appear to be running. Here are the retrieved sources:",
        ]
        for i, h in enumerate(hits, start=1):
            lines.append(f"- [{i}] {h['title']} ({h['url']}) | chunk_id={h['chunk_id']}")
        lines.append("\nStart Ollama with: `ollama serve`")
        return "\n".join(lines)

    try:
        txt = _call_ollama(prompt, model=model)
        return txt if txt else "The model returned an empty response."
    except Exception as e:
        lines = [
            "### Answer (fallback)",
            "",
            "I found these sources, but the local LLM call failed:",
        ]
        for i, h in enumerate(hits, start=1):
            lines.append(f"- [{i}] {h['title']} ({h['url']}) | chunk_id={h['chunk_id']}")
        lines.append(f"\nLLM error: {e}")
        return "\n".join(lines)


# -----------------------------
# High-level function for API/CLI
# -----------------------------
def answer_query(index_path: str, meta_path: str, query: str, k: int, model: str = OLLAMA_MODEL) -> Dict[str, Any]:
    index = load_faiss_index(index_path)
    meta = load_meta(meta_path)
    embed_model = get_embed_model()  # cached

    hits, debug = retrieve(index, meta, embed_model, query, k)
    ans = make_answer(query, hits, model=model)

    return {
        "query": query,
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


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index_path")
    ap.add_argument("meta_path")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name (e.g. llama3.1:8b)")
    args = ap.parse_args()

    result = answer_query(args.index_path, args.meta_path, args.query, args.k, model=args.model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Wrote answers: {args.out}  (n=1)")


if __name__ == "__main__":
    main()