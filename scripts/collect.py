#!/usr/bin/env python3
"""
collect.py
Semi-automated data collector for womens-health-rag corpus.

Inputs:
  - manifest.csv: curated list of sources + tags
Outputs:
  - documents.jsonl: one JSON per document with local_path, sha256, collected_at, tags, etc.
  - collector_report.json: summary + failures

Behavior:
  - Downloads PDF if URL endswith .pdf OR response looks like a PDF
  - Otherwise stores HTML snapshot
  - Keeps allowlist optional (edit ALLOWLIST_DOMAINS)

Notes:
  - Incremental: previously OK docs are preserved and not re-downloaded.
  - Robust JSONL: always writes one JSON object per line.
  - Atomic output: writes to temp file then replaces.
"""

import csv
import json
import os
import re
import sys
import time
import hashlib
from urllib.parse import urlparse
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Install with: pip install requests")
    sys.exit(1)

ALLOWLIST_DOMAINS = set()

USER_AGENT = "womens-health-rag-collector/0.2 (+local educational project)"
TIMEOUT = 30
SLEEP_SECONDS = 0.5


def utc_now_z() -> str:
    """Timezone-aware UTC timestamp ending with Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_slug(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:max_len] if s else "doc"


def download(url: str) -> tuple[bytes, str]:
    """Returns (content_bytes, content_type_header)."""
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()
    return r.content, ctype


def load_existing_ok_records(jsonl_path: str) -> dict:
    """
    Loads existing documents.jsonl and returns a dict doc_id -> record
    for records with status == "ok". This makes collection incremental.
    """
    existing: dict[str, dict] = {}
    if not os.path.exists(jsonl_path):
        return existing

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue

                # If multiple JSON objects got glued together, split them safely.
                parts = raw.replace("}{", "}\n{").splitlines()
                for p in parts:
                    p = p.strip()
                    if not p:
                        continue
                    try:
                        d = json.loads(p)
                        if d.get("status") == "ok" and d.get("doc_id"):
                            existing[d["doc_id"]] = d
                    except Exception:
                        # ignore malformed fragments
                        pass
    except Exception:
        pass

    return existing


def main(manifest_csv: str, out_jsonl: str, report_json: str):
    # Ensure report directory exists
    if os.path.dirname(report_json):
        os.makedirs(os.path.dirname(report_json), exist_ok=True)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(out_jsonl), ".."))
    raw_a = os.path.join(base_dir, "raw", "A_patient_info")
    raw_b = os.path.join(base_dir, "raw", "B_peer_reviewed")
    os.makedirs(raw_a, exist_ok=True)
    os.makedirs(raw_b, exist_ok=True)

    existing_ok = load_existing_ok_records(out_jsonl)

    with open(manifest_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    docs = []
    failures = []

    for row in rows:
        doc_id = (row.get("doc_id") or "").strip()
        title = (row.get("title") or "").strip()
        url = (row.get("url_or_doi") or "").strip()
        tags = (row.get("tags") or "").strip()
        notes = (row.get("notes") or "").strip()

        if not doc_id:
            doc_id = safe_slug(title) or f"doc_{len(docs)+1}"

        # If already collected OK, preserve it and move on (incremental behavior).
        if doc_id in existing_ok:
            docs.append(existing_ok[doc_id])
            continue

        local_path = None
        status = "skipped"
        reason = None

        if url.startswith("http://") or url.startswith("https://"):
            domain = urlparse(url).netloc

            if ALLOWLIST_DOMAINS and domain not in ALLOWLIST_DOMAINS:
                reason = f"domain_not_allowlisted:{domain}"
            else:
                try:
                    content, ctype = download(url)

                    # Detect PDF by URL, magic bytes, or content-type
                    is_pdf = (
                        url.lower().endswith(".pdf")
                        or content[:4] == b"%PDF"
                        or "application/pdf" in ctype
                    )

                    tier_tag = "tier:A_patient_info" if "tier:A_patient_info" in tags else "tier:B_peer_reviewed"
                    out_dir = raw_a if tier_tag == "tier:A_patient_info" else raw_b

                    ext = ".pdf" if is_pdf else ".html"
                    filename = f"{doc_id}{ext}"
                    local_path = os.path.join(out_dir, filename)

                    with open(local_path, "wb") as wf:
                        wf.write(content)

                    status = "ok"

                except Exception as e:
                    reason = f"download_failed:{type(e).__name__}:{e}"
        else:
            reason = "no_url"

        record = {
            "doc_id": doc_id,
            "title": title,
            "url_or_doi": url,
            "tags": [t for t in tags.split(";") if t.strip()],
            "notes": notes,
            "collected_at": utc_now_z(),
            "status": status,
            "local_path": local_path,
        }

        if status == "ok" and local_path and os.path.exists(local_path):
            record["sha256"] = sha256_file(local_path)
            record["bytes"] = os.path.getsize(local_path)
        else:
            record["reason"] = reason
            failures.append(record)

        docs.append(record)
        time.sleep(SLEEP_SECONDS)

    # Atomic write: write temp then replace
    tmp_jsonl = out_jsonl + ".tmp"
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, out_jsonl)

    report = {
        "generated_at": utc_now_z(),
        "manifest": os.path.abspath(manifest_csv),
        "out_jsonl": os.path.abspath(out_jsonl),
        "counts": {
            "total": len(docs),
            "ok": sum(1 for d in docs if d.get("status") == "ok"),
            "failed_or_skipped": sum(1 for d in docs if d.get("status") != "ok"),
        },
        "failures_preview": failures[:50],
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Done. OK={report['counts']['ok']}  Failed/Skipped={report['counts']['failed_or_skipped']}")
    print(f"Wrote: {out_jsonl}")
    print(f"Report: {report_json}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 scripts/collect.py <manifest.csv> <documents.jsonl> <collector_report.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
