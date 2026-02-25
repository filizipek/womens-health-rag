#!/usr/bin/env python3
"""
extract_text.py
Extracts clean text from collected HTML/PDF docs and writes:

- corpus/text/{doc_id}.txt
- corpus/metadata/text_index.jsonl  (one JSON per doc)

Usage:
  python3 scripts/extract_text.py \
    corpus/metadata/documents.jsonl \
    corpus/text \
    corpus/metadata/text_index.jsonl

Tip:
  Use documents_ok_only.jsonl if you generated it; it should contain only OK docs.
"""

import os
import re
import sys
import json
import hashlib
from datetime import datetime, timezone

# HTML parsing
from bs4 import BeautifulSoup

# PDF parsing
from pypdf import PdfReader


def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_from_html(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()

    # Best-effort decode: try utf-8 then fallback latin-1
    try:
        html = raw.decode("utf-8", errors="ignore")
    except Exception:
        html = raw.decode("latin-1", errors="ignore")

    soup = BeautifulSoup(html, "lxml")

    # Remove junk
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = normalize_whitespace(text)

    # Remove very short lines that are mostly UI noise
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if len(ln) >= 2]
    return "\n".join(lines).strip()


def extract_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = t.strip()
        if t:
            parts.append(t)
    text = "\n\n".join(parts)
    text = normalize_whitespace(text)
    return text


def tags_to_fields(tags: list[str]) -> dict:
    # Optional helper: pull tier/source/type from tag prefixes
    out = {"tier": None, "source": None, "type": None}
    for t in tags or []:
        if t.startswith("tier:"):
            out["tier"] = t
        elif t.startswith("source:"):
            out["source"] = t
        elif t.startswith("type:"):
            out["type"] = t
    return out


def main(documents_jsonl: str, out_text_dir: str, out_index_jsonl: str) -> int:
    os.makedirs(out_text_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_index_jsonl) or ".", exist_ok=True)

    records = []
    total = 0
    ok = 0
    empty_text = 0

    with open(documents_jsonl, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except Exception:
                # If something weird slipped into JSONL, ignore safely
                continue

            total += 1
            if d.get("status") != "ok":
                continue

            doc_id = d.get("doc_id")
            title = d.get("title", "")
            url_or_doi = d.get("url_or_doi", "")
            tags = d.get("tags", [])
            local_path = d.get("local_path")

            if not doc_id or not local_path or not os.path.exists(local_path):
                continue

            ext = os.path.splitext(local_path)[1].lower()
            try:
                if ext == ".pdf":
                    text = extract_from_pdf(local_path)
                else:
                    # treat everything else as HTML-ish
                    text = extract_from_html(local_path)
            except Exception as e:
                # record extraction failure
                rec = {
                    "doc_id": doc_id,
                    "title": title,
                    "url_or_doi": url_or_doi,
                    "tags": tags,
                    "local_path": local_path,
                    "text_path": None,
                    "status": "extract_failed",
                    "reason": f"{type(e).__name__}:{e}",
                    "extracted_at": utc_now_z(),
                }
                records.append(rec)
                continue

            if not text or len(text) < 50:
                empty_text += 1

            out_txt = os.path.join(out_text_dir, f"{doc_id}.txt")
            tmp_txt = out_txt + ".tmp"
            with open(tmp_txt, "w", encoding="utf-8") as wf:
                wf.write(text + "\n")
            os.replace(tmp_txt, out_txt)

            fields = tags_to_fields(tags)

            rec = {
                "doc_id": doc_id,
                "title": title,
                "url_or_doi": url_or_doi,
                "tags": tags,
                "tier": fields["tier"],
                "source": fields["source"],
                "type": fields["type"],
                "local_path": local_path,
                "text_path": out_txt,
                "status": "ok",
                "text_sha256": sha256_text(text),
                "n_chars": len(text),
                "n_words": len(text.split()),
                "extracted_at": utc_now_z(),
            }
            records.append(rec)
            ok += 1

    # Write index JSONL atomically
    tmp_index = out_index_jsonl + ".tmp"
    with open(tmp_index, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_index, out_index_jsonl)

    print(f"Done. Parsed={total}  Extracted_OK={ok}  Very_short_or_empty={empty_text}")
    print(f"Wrote index: {out_index_jsonl}")
    print(f"Wrote texts: {out_text_dir}/<doc_id>.txt")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 scripts/extract_text.py <documents.jsonl> <out_text_dir> <out_text_index.jsonl>")
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
