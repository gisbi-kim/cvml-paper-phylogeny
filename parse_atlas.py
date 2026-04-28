"""
parse_atlas.py
==============
all_dblp.json → papers_parsed.json + titles_chronological.txt
"""
import json
import sys
from pathlib import Path

RAW = Path("data/raw/all_dblp.json")
OUT_JSON = Path("data/intermediate/papers_parsed.json")
OUT_TITLES = Path("data/intermediate/titles_chronological.txt")

def main():
    print(f"Loading {RAW} ...")
    with open(RAW, encoding="utf-8") as f:
        raw = json.load(f)

    papers = []
    skipped = 0
    for p in raw:
        title = (p.get("title") or "").strip().rstrip(".")
        if not title:
            skipped += 1
            continue
        papers.append({
            "title":   title,
            "venue":   p.get("venue", ""),
            "year":    str(p.get("year", "")),
            "authors": p.get("authors", ""),
            "doi":     p.get("doi", ""),
            "url":     p.get("ee", ""),
        })

    # sort chronologically
    papers.sort(key=lambda p: (p["year"], p["venue"], p["title"]))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    with open(OUT_TITLES, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(f"{p['year']} | {p['venue']} | {p['title']}\n")

    print(f"Done. {len(papers)} papers saved, {skipped} skipped.")
    print(f"  → {OUT_JSON}")
    print(f"  → {OUT_TITLES}")

if __name__ == "__main__":
    main()
