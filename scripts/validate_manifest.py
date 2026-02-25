#!/usr/bin/env python3
"""""
validate_manifest.py
Validates manifest.csv against tag_vocab.yaml and basic required fields.

Usage:
  python scripts/validate_manifest.py corpus/metadata/manifest.csv corpus/metadata/tag_vocab.yaml

"""

import csv, sys, yaml

REQUIRED_COLUMNS = {"doc_id","title","url_or_doi","tags"}

def load_vocab(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        vocab = yaml.safe_load(f)
    all_tags = set()
    for _, tags in (vocab or {}).items():
        for t in tags:
            all_tags.add(t)
    return all_tags

def main(manifest_csv: str, vocab_yaml: str):
    vocab = load_vocab(vocab_yaml)

    with open(manifest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing_cols = REQUIRED_COLUMNS - cols
        if missing_cols:
            print(f"ERROR: manifest.csv missing columns: {sorted(missing_cols)}")
            sys.exit(1)

        bad_rows = 0
        unknown_tags = 0
        required_tag_errors = 0

        for i, row in enumerate(reader, start=2):  # header is line 1
            tags = [t.strip() for t in (row.get("tags") or "").split(";") if t.strip()]
            if not row.get("doc_id") or not row.get("title"):
                print(f"Line {i}: missing doc_id/title")
                bad_rows += 1

            # required tag families
            has_tier = any(t.startswith("tier:") for t in tags)
            has_source = any(t.startswith("source:") for t in tags)
            has_type = any(t.startswith("type:") for t in tags)
            has_topic = any(t.startswith("topic:") for t in tags)

            if not (has_tier and has_source and has_type and has_topic):
                print(f"Line {i}: missing required tag family (need tier/source/type/topic). Got: {tags}")
                required_tag_errors += 1

            for t in tags:
                if t not in vocab:
                    print(f"Line {i}: unknown tag: {t}")
                    unknown_tags += 1

        if bad_rows or unknown_tags or required_tag_errors:
            print("Validation failed.")
            print(f"- Rows with missing fields: {bad_rows}")
            print(f"- Required tag family errors: {required_tag_errors}")
            print(f"- Unknown tags: {unknown_tags}")
            sys.exit(1)

    print("Manifest validation OK.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/validate_manifest.py <manifest.csv> <tag_vocab.yaml>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
