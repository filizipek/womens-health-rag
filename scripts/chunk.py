#!/usr/bin/env python3
"""
chunk.py
Recursive chunking (with overlap) for extracted texts.

Input:
  - corpus/metadata/text_index.jsonl (from extract_text.py)

Output:
  - corpus/chunks/chunks.jsonl (one JSON per chunk with metadata)

Usage:
  python3 scripts/chunk.py \
    corpus/metadata/text_index.jsonl \
    corpus/chunks/chunks.jsonl \
    1000 \
    150

Args:
  max_tokens: target chunk size (approx tokens)
  overlap_tokens: overlap between chunks (approx tokens)

Token counting:
  - Uses tiktoken if installed; otherwise approximates tokens as len(text)/4
"""

import os
import sys
import json
import re
from datetime import datetime, timezone

def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# Optional: real token counting if tiktoken exists
_ENCODER = None
def _get_encoder():
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    try:
        import tiktoken  # type: ignore
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
    return _ENCODER

def count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc is None:
        return max(1, len(text) // 4)  # rough but works
    return len(enc.encode(text))

def take_last_overlap(text: str, overlap_tokens: int) -> str:
    # overlap by approx characters if no encoder; otherwise token-based
    enc = _get_encoder()
    if enc is None:
        overlap_chars = overlap_tokens * 4
        return text[-overlap_chars:] if overlap_chars < len(text) else text
    toks = enc.encode(text)
    if len(toks) <= overlap_tokens:
        return text
    tail = toks[-overlap_tokens:]
    return enc.decode(tail)

def normalize(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

SEPARATORS = ["\n\n", "\n", ". ", " "]

def recursive_split(text: str, max_tokens: int) -> list[str]:
    """
    Split text into pieces <= max_tokens using a recursive separator strategy.
    """
    text = normalize(text)
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    # Try splitting with each separator from coarse->fine
    for sep in SEPARATORS:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""

        for p in parts:
            piece = p if sep == " " else (p + sep)
            if not buf:
                candidate = piece
            else:
                candidate = buf + piece

            if count_tokens(candidate) <= max_tokens:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf.strip())
                buf = piece

        if buf.strip():
            chunks.append(buf.strip())

        # If we actually reduced size, recurse on any still-too-large chunks
        if len(chunks) > 1:
            out = []
            for c in chunks:
                if count_tokens(c) > max_tokens and sep != " ":
                    out.extend(recursive_split(c, max_tokens))
                elif count_tokens(c) > max_tokens and sep == " ":
                    # Hard split fallback
                    out.extend(hard_split(c, max_tokens))
                else:
                    out.append(c)
            return out

    # If no separator worked, hard split
    return hard_split(text, max_tokens)

def hard_split(text: str, max_tokens: int) -> list[str]:
    enc = _get_encoder()
    text = normalize(text)
    if not text:
        return []
    if enc is None:
        # Approximate hard split by characters
        max_chars = max_tokens * 4
        return [text[i:i+max_chars].strip() for i in range(0, len(text), max_chars)]
    toks = enc.encode(text)
    out = []
    for i in range(0, len(toks), max_tokens):
        out.append(enc.decode(toks[i:i+max_tokens]).strip())
    return [x for x in out if x]

def add_overlap(chunks: list[str], overlap_tokens: int, max_tokens: int) -> list[str]:
    if not chunks or overlap_tokens <= 0:
        return chunks
    out = []
    prev = ""
    for c in chunks:
        if prev:
            ov = take_last_overlap(prev, overlap_tokens)
            merged = (ov + "\n" + c).strip()
            # Keep merged within budget; if too big, keep original c
            if count_tokens(merged) <= max_tokens + overlap_tokens:
                out.append(merged)
            else:
                out.append(c)
        else:
            out.append(c)
        prev = c
    return out

def main(text_index_jsonl: str, out_chunks_jsonl: str, max_tokens: int, overlap_tokens: int) -> int:
    os.makedirs(os.path.dirname(out_chunks_jsonl) or ".", exist_ok=True)

    chunk_count = 0
    doc_count = 0

    tmp_out = out_chunks_jsonl + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as out:
        with open(text_index_jsonl, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                d = json.loads(raw)
                if d.get("status") != "ok":
                    continue

                doc_id = d["doc_id"]
                title = d.get("title", "")
                url_or_doi = d.get("url_or_doi", "")
                tags = d.get("tags", [])
                tier = d.get("tier")
                source = d.get("source")
                typ = d.get("type")
                text_path = d.get("text_path")

                if not text_path or not os.path.exists(text_path):
                    continue

                with open(text_path, "r", encoding="utf-8") as tf:
                    text = tf.read()

                parts = recursive_split(text, max_tokens)
                parts = add_overlap(parts, overlap_tokens, max_tokens)

                doc_count += 1
                for i, chunk_text in enumerate(parts):
                    chunk_text = chunk_text.strip()
                    if not chunk_text:
                        continue
                    rec = {
                        "chunk_id": f"{doc_id}__{i:04d}",
                        "doc_id": doc_id,
                        "title": title,
                        "url_or_doi": url_or_doi,
                        "tier": tier,
                        "source": source,
                        "type": typ,
                        "tags": tags,
                        "text_path": text_path,
                        "text": chunk_text,
                        "n_chars": len(chunk_text),
                        "n_tokens_est": count_tokens(chunk_text),
                        "created_at": utc_now_z(),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    chunk_count += 1

    os.replace(tmp_out, out_chunks_jsonl)
    print(f"Done. Docs_chunked={doc_count}  Chunks_written={chunk_count}")
    print(f"Wrote: {out_chunks_jsonl}")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 scripts/chunk.py <text_index.jsonl> <out_chunks.jsonl> <max_tokens> <overlap_tokens>")
        sys.exit(1)

    sys.exit(main(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
    ))
