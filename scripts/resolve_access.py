#!/usr/bin/env python3
"""
resolve_access.py
Auto-resolves DOI URLs to accessible equivalents in this order:

1) PMC full text (PMCID)  -> https://pmc.ncbi.nlm.nih.gov/articles/<PMCID>/
2) PubMed record (PMID)   -> https://pubmed.ncbi.nlm.nih.gov/<PMID>/
3) Unpaywall OA PDF       -> best_oa_location.url_for_pdf
4) Unpaywall OA landing   -> best_oa_location.url
5) PubMed fallback search by title (E-utilities esearch)
6) Otherwise mark: status:paywalled_or_unresolved

Writes an updated manifest, leaving originals untouched.

Usage:
  python3 scripts/resolve_access.py corpus/metadata/manifest.csv corpus/metadata/manifest_resolved.csv youremail@example.com
"""

import csv
import sys
import time
import requests
from urllib.parse import quote


UA = "womens-health-rag-resolver/0.2 (+educational)"
TIMEOUT = 25
SLEEP = 0.25

BLOCKED_PDF_HOSTS = {
    "onlinelibrary.wiley.com",
    "obgyn.onlinelibrary.wiley.com",
    "jamanetwork.com",
    "journals.lww.com",
    "www.fertstert.org",
    "www.gynecologiconcology-online.net",
}

from urllib.parse import urlparse

def host_is_blocked(u: str) -> bool:
    try:
        return urlparse(u).netloc in BLOCKED_PDF_HOSTS
    except Exception:
        return False

def is_doi_url(u: str) -> bool:
    u = (u or "").strip()
    return u.startswith("https://doi.org/") or u.startswith("http://doi.org/")


def extract_doi(u: str) -> str | None:
    u = (u or "").strip()
    if not u:
        return None
    if is_doi_url(u):
        return u.split("doi.org/", 1)[1].strip()
    if u.lower().startswith("10."):
        return u.strip()
    return None


def ncbi_idconv(doi: str, email: str) -> dict:
    """NCBI id converter can map DOI -> PMID/PMCID when possible."""
    url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    params = {"tool": "womens-health-rag", "email": email, "ids": doi, "format": "json"}
    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def unpaywall(doi: str, email: str) -> dict:
    """Unpaywall OA resolver (requires email)."""
    url = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": email}
    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def add_tag(tags: str, new_tag: str) -> str:
    parts = [t.strip() for t in (tags or "").split(";") if t.strip()]
    if new_tag not in parts:
        parts.append(new_tag)
    return ";".join(parts)


def pubmed_search_by_title(title: str, email: str) -> str | None:
    """
    PubMed fallback: use E-utilities esearch to find a PMID using title.
    This often works even when DOI->PMID mapping fails.

    Strategy:
      - Search for exact-ish title using [Title] field
      - Return first PMID if any
    """
    title = (title or "").strip()
    if not title:
        return None

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    # Quote title to reduce noise + use [Title] tag.
    # Example term: "Some Title Here"[Title]
    term = f"\"{title}\"[Title]"

    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": "5",
        "tool": "womens-health-rag",
        "email": email,
    }

    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    ids = (((data.get("esearchresult") or {}).get("idlist")) or [])
    return ids[0] if ids else None


def main(in_csv: str, out_csv: str, email: str):
    with open(in_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # ensure columns exist
    for r in rows:
        r.setdefault("notes", "")
        r.setdefault("tags", "")

    fixed = 0
    pmc = 0
    pubmed = 0
    oa = 0
    unchanged = 0
    failed = 0

    for r in rows:
        url = (r.get("url_or_doi") or "").strip()
        doi = extract_doi(url)

        # Only try to resolve DOI links
        if not doi:
            unchanged += 1
            continue

        try:
            # 1) NCBI mapping DOI -> PMCID/PMID
            data = ncbi_idconv(doi, email)
            recs = data.get("records") or []
            rec = recs[0] if recs else {}

            pmcid = rec.get("pmcid")
            pmid = rec.get("pmid")

            if pmcid:
                r["url_or_doi"] = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
                r["tags"] = add_tag(r["tags"], "access:pmc_fulltext")
                r["notes"] = (r["notes"] + " | resolved DOI->PMC").strip(" |")
                fixed += 1
                pmc += 1
                time.sleep(SLEEP)
                continue

            if pmid:
                r["url_or_doi"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                r["tags"] = add_tag(r["tags"], "access:pubmed_record")
                r["notes"] = (r["notes"] + " | resolved DOI->PubMed (idconv)").strip(" |")
                fixed += 1
                pubmed += 1
                time.sleep(SLEEP)
                continue

            # 2) Unpaywall for legal OA PDF/landing
            upw = unpaywall(doi, email)
            best = upw.get("best_oa_location") or {}
            pdf = best.get("url_for_pdf")
            landing = best.get("url")

            # 2) Unpaywall for legal OA PDF/landing
            upw = unpaywall(doi, email)
            best = upw.get("best_oa_location") or {}
            pdf = best.get("url_for_pdf")
            landing = best.get("url")

            # If OA link points to a host that returns 403 for bots, prefer PubMed fallback
            if pdf and host_is_blocked(pdf):
                pmid2 = pubmed_search_by_title(r.get("title", ""), email)
                if pmid2:
                    r["url_or_doi"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid2}/"
                    r["tags"] = add_tag(r["tags"], "access:pubmed_record")
                    r["notes"] = (r["notes"] + " | OA PDF host blocks bots; using PubMed").strip(" |")
                    fixed += 1
                    pubmed += 1
                    time.sleep(SLEEP)
                    continue

            if landing and host_is_blocked(landing):
                pmid2 = pubmed_search_by_title(r.get("title", ""), email)
                if pmid2:
                    r["url_or_doi"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid2}/"
                    r["tags"] = add_tag(r["tags"], "access:pubmed_record")
                    r["notes"] = (r["notes"] + " | OA landing host blocks bots; using PubMed").strip(" |")
                    fixed += 1
                    pubmed += 1
                    time.sleep(SLEEP)
                    continue

            # Otherwise keep OA PDF / landing
            if pdf:
                r["url_or_doi"] = pdf
                r["tags"] = add_tag(r["tags"], "access:oa_pdf")
                r["notes"] = (r["notes"] + " | resolved DOI->OA PDF (Unpaywall)").strip(" |")
                fixed += 1
                oa += 1
                time.sleep(SLEEP)
                continue

            if landing:
                r["url_or_doi"] = landing
                r["tags"] = add_tag(r["tags"], "access:oa_landing")
                r["notes"] = (r["notes"] + " | resolved DOI->OA landing (Unpaywall)").strip(" |")
                fixed += 1
                oa += 1
                time.sleep(SLEEP)
                continue


            # 3) PubMed fallback by title (automatic workaround for 403 publishers)
            pmid2 = pubmed_search_by_title(r.get("title", ""), email)
            if pmid2:
                r["url_or_doi"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid2}/"
                r["tags"] = add_tag(r["tags"], "access:pubmed_record")
                r["notes"] = (r["notes"] + " | resolved via PubMed title search").strip(" |")
                fixed += 1
                pubmed += 1
                time.sleep(SLEEP)
                continue

            # 4) Still nothing: mark unresolved
            r["tags"] = add_tag(r["tags"], "status:paywalled_or_unresolved")
            r["notes"] = (r["notes"] + " | unresolved DOI (likely paywalled)").strip(" |")
            failed += 1
            time.sleep(SLEEP)

        except Exception as e:
            r["tags"] = add_tag(r["tags"], "status:resolver_error")
            r["notes"] = (r["notes"] + f" | resolver error: {type(e).__name__}").strip(" |")
            failed += 1
            time.sleep(SLEEP)

    # Write updated manifest
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("Wrote:", out_csv)
    print("Resolved total:", fixed, "| PMC:", pmc, "| PubMed:", pubmed, "| OA:", oa)
    print("Unchanged:", unchanged, "| Unresolved/failed:", failed)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 scripts/resolve_access.py <in_manifest.csv> <out_manifest.csv> <email_for_unpaywall_ncbi>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
