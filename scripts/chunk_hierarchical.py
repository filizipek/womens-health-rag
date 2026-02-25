#!/usr/bin/env python3
"""
chunk_hierarchical.py (improved)
Hierarchical chunking: parent=section, child=sub-chunk.

Fixes vs previous:
- Much stricter heading detection (prevents "every line is a heading")
- Skips noisy UI lines as headings
- Guardrail: if too many sections are detected, fallback to single section

Usage:
  python3 scripts/chunk_hierarchical.py \
    corpus/metadata/text_index.jsonl \
    corpus/chunks/parents.jsonl \
    corpus/chunks/children.jsonl \
    1800 \
    900 \
    150
"""

import os, sys, json, re
from datetime import datetime, timezone
from urllib.parse import urlparse

def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# Optional token counting via tiktoken
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
        return max(1, len(text) // 4)
    return len(enc.encode(text))

def normalize(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

SEPARATORS = ["\n\n", "\n", ". ", " "]

def recursive_split(text: str, max_tokens: int) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    for sep in SEPARATORS:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            piece = p if sep == " " else (p + sep)
            candidate = piece if not buf else (buf + piece)
            if count_tokens(candidate) <= max_tokens:
                buf = candidate
            else:
                if buf.strip():
                    chunks.append(buf.strip())
                buf = piece
        if buf.strip():
            chunks.append(buf.strip())

        if len(chunks) > 1:
            out = []
            for c in chunks:
                if count_tokens(c) > max_tokens and sep != " ":
                    out.extend(recursive_split(c, max_tokens))
                elif count_tokens(c) > max_tokens and sep == " ":
                    out.extend(hard_split(c, max_tokens))
                else:
                    out.append(c)
            return out

    return hard_split(text, max_tokens)

def hard_split(text: str, max_tokens: int) -> list[str]:
    enc = _get_encoder()
    text = normalize(text)
    if not text:
        return []
    if enc is None:
        max_chars = max_tokens * 4
        return [text[i:i+max_chars].strip() for i in range(0, len(text), max_chars)]
    toks = enc.encode(text)
    out = []
    for i in range(0, len(toks), max_tokens):
        out.append(enc.decode(toks[i:i+max_tokens]).strip())
    return [x for x in out if x]

def take_last_overlap(text: str, overlap_tokens: int) -> str:
    enc = _get_encoder()
    if enc is None:
        overlap_chars = overlap_tokens * 4
        return text[-overlap_chars:] if overlap_chars < len(text) else text
    toks = enc.encode(text)
    if len(toks) <= overlap_tokens:
        return text
    return enc.decode(toks[-overlap_tokens:])

def add_overlap(chunks: list[str], overlap_tokens: int, child_max_tokens: int) -> list[str]:
    if not chunks or overlap_tokens <= 0:
        return chunks
    out = []
    prev = ""
    for c in chunks:
        if prev:
            ov = take_last_overlap(prev, overlap_tokens)
            merged = (ov + "\n" + c).strip()
            if count_tokens(merged) <= child_max_tokens + overlap_tokens:
                out.append(merged)
            else:
                out.append(c)
        else:
            out.append(c)
        prev = c
    return out

# ---------- Improved heading detection ----------

CUE_WORDS = [
    "symptom", "symptoms",
    "treatment", "treatments",
    "cause", "causes",
    "diagnosis", "diagnose",
    "when to", "urgent", "get help",
    "prevention", "risk", "risks",
    "complication", "complications",
    "tests", "test", "screening",
    "self-care", "management",
    "what happens", "how to", "overview"
]

NOISE_PREFIXES = (
    "skip to main content",
    "cookies",
    "search",
    "menu",
    "print",
    "share",
    "page last reviewed",
    "next review due",
)

def clean_heading_candidate(line: str) -> str:
    line = (line or "").strip()
    line = re.sub(r"^[•\-\–\—\*]+\s*", "", line)  # strip bullets/dashes
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()

def looks_like_heading(line: str) -> bool:
    raw = (line or "").strip()
    if not raw:
        return False

    low = raw.lower().strip()
    if any(low.startswith(p) for p in NOISE_PREFIXES):
        return True  # boundary marker

    cand = clean_heading_candidate(raw)
    lowc = cand.lower()

    # too short headings are usually noise (except ALL CAPS)
    if len(cand) < 5:
        return cand.isupper()

    # ignore "Adenomyosis - NHS" style title repeats: still a boundary but not a real section
    # (we treat as heading but later guardrail will prevent tiny sections)
    # Require headings to be reasonably short
    if len(cand) > 90:
        return False

    # numbered headings: "1.", "2.3", "3 Introduction"
    if re.match(r"^\d+(\.\d+)*\s+.+", cand):
        return True

    # ends with ":" is very common for section headers
    if cand.endswith(":"):
        return True

    # ALL CAPS headings
    if cand.isupper() and 4 <= len(cand) <= 80:
        return True

    # must have cue word OR be very short and title-like (but not sentence-like)
    if any(cw in lowc for cw in CUE_WORDS):
        return True

    # avoid treating normal sentences as headings
    if cand.endswith("."):
        return False

    # very short phrase headings (2–6 words) are okay if no punctuation
    words = re.findall(r"[A-Za-z]+", cand)
    if 2 <= len(words) <= 6 and not re.search(r"[,:;()]", cand):
        # but reject if it's basically a sentence fragment with verbs like "is/are"
        if re.search(r"\b(is|are|was|were|have|has|can)\b", lowc):
            return False
        return True

    return False

def split_into_sections(text: str, fallback_title: str) -> list[tuple[str, str]]:
    """
    Returns list of (section_title, section_text).
    If headings are unreliable, returns one big section.
    """
    text = normalize(text)
    if not text:
        return []

    # keep blanks for better structure, but we’ll iterate safely
    raw_lines = text.split("\n")

    sections = []
    current_title = fallback_title or "Document"
    buf = []
    found_any_heading = False

    for ln in raw_lines:
        lns = ln.strip()
        if not lns:
            # preserve paragraph breaks inside section body
            if buf and buf[-1] != "":
                buf.append("")
            continue

        if looks_like_heading(lns):
            found_any_heading = True
            # flush previous section
            body = "\n".join(buf).strip()
            if body:
                sections.append((current_title, body))
            buf = []

            # Use cleaned heading as title
            ht = clean_heading_candidate(lns)
            # Avoid setting title to super-generic noise lines
            if any(ht.lower().startswith(p) for p in NOISE_PREFIXES):
                # keep the existing title; treat as boundary only
                # (so we don't create "Skip to main content" sections)
                continue
            current_title = ht
        else:
            buf.append(lns)

    body = "\n".join(buf).strip()
    if body:
        sections.append((current_title, body))

    # If we didn’t detect headings, just one big section
    if not found_any_heading:
        return [(fallback_title or "Document", text)]

    # Guardrail: if heading detector went crazy, fallback to single section
    if len(sections) > 120:
        return [(fallback_title or "Document", text)]

    # Merge tiny sections into previous
    cleaned = []
    for title, body in sections:
        if len(body) < 300 and cleaned:
            pt, pb = cleaned[-1]
            cleaned[-1] = (pt, (pb + "\n\n" + title + "\n" + body).strip())
        else:
            cleaned.append((title, body))

    return cleaned

# ---------- Main ----------

def main(text_index_jsonl: str, out_parents: str, out_children: str,
         parent_max_tokens: int, child_max_tokens: int, overlap_tokens: int) -> int:

    os.makedirs(os.path.dirname(out_parents) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_children) or ".", exist_ok=True)

    parent_count = 0
    child_count = 0
    docs = 0

    tmp_p = out_parents + ".tmp"
    tmp_c = out_children + ".tmp"

    with open(tmp_p, "w", encoding="utf-8") as parents_out, open(tmp_c, "w", encoding="utf-8") as children_out:
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

                sections = split_into_sections(text, fallback_title=title or doc_id)
                if not sections:
                    continue

                docs += 1

                for si, (sec_title, sec_body) in enumerate(sections):
                    sec_body = normalize(sec_body)
                    if not sec_body:
                        continue

                    parent_parts = recursive_split(sec_body, parent_max_tokens)
                    for pi, parent_text in enumerate(parent_parts):
                        parent_id = f"{doc_id}__sec{si:02d}__p{pi:02d}"
                        parent_rec = {
                            "parent_id": parent_id,
                            "doc_id": doc_id,
                            "doc_title": title,
                            "section_title": sec_title,
                            "url_or_doi": url_or_doi,
                            "tier": tier,
                            "source": source,
                            "type": typ,
                            "tags": tags,
                            "text": parent_text,
                            "n_tokens_est": count_tokens(parent_text),
                            "created_at": utc_now_z(),
                        }
                        parents_out.write(json.dumps(parent_rec, ensure_ascii=False) + "\n")
                        parent_count += 1

                        child_parts = recursive_split(parent_text, child_max_tokens)
                        child_parts = add_overlap(child_parts, overlap_tokens, child_max_tokens)

                        for ci, child_text in enumerate(child_parts):
                            child_text = child_text.strip()
                            if not child_text:
                                continue
                            child_id = f"{parent_id}__c{ci:04d}"
                            child_rec = {
                                "chunk_id": child_id,
                                "parent_id": parent_id,
                                "doc_id": doc_id,
                                "title": title,
                                "section_title": sec_title,
                                "url_or_doi": url_or_doi,
                                "tier": tier,
                                "source": source,
                                "type": typ,
                                "tags": tags,
                                "text": child_text,
                                "n_tokens_est": count_tokens(child_text),
                                "created_at": utc_now_z(),
                            }
                            children_out.write(json.dumps(child_rec, ensure_ascii=False) + "\n")
                            child_count += 1

    os.replace(tmp_p, out_parents)
    os.replace(tmp_c, out_children)

    print(f"Done. Docs={docs}  Parents={parent_count}  Children={child_count}")
    print(f"Wrote parents:  {out_parents}")
    print(f"Wrote children: {out_children}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python3 scripts/chunk_hierarchical.py <text_index.jsonl> <parents.jsonl> <children.jsonl> <parent_max_tokens> <child_max_tokens> <overlap_tokens>")
        sys.exit(1)
    sys.exit(main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
        int(sys.argv[5]),
        int(sys.argv[6]),
    ))
