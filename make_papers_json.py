"""
make_papers_json.py
===================
papers_classified.json → papers.json  (web viewer용)

papers.json schema (per entry):
  phylum, class, order, genus,
  title, authors, venue, year, citations, doi, url
"""
import json
from pathlib import Path

IN_CLASSIFIED = Path("data/intermediate/papers_classified.json")
IN_ENRICHED   = Path("data/raw/all_enriched.json")   # optional: citations
OUT           = Path("papers.json")

def load_citations():
    """Build doi→citations map from all_enriched.json if available."""
    if not IN_ENRICHED.exists():
        print("  (all_enriched.json not found - citations will be 0)")
        return {}
    print("Loading enriched data for citations ...")
    with open(IN_ENRICHED, encoding="utf-8") as f:
        enriched = json.load(f)
    cit = {}
    for p in enriched:
        doi = (p.get("doi") or "").strip().lower()
        c   = p.get("citations") or p.get("citationCount") or 0
        if doi:
            cit[doi] = int(c)
    print(f"  {len(cit)} citation entries loaded.")
    return cit

def main():
    print(f"Loading {IN_CLASSIFIED} ...")
    with open(IN_CLASSIFIED, encoding="utf-8") as f:
        papers = json.load(f)

    citations = load_citations()

    out = []
    for p in papers:
        doi = (p.get("doi") or "").strip().lower()
        out.append({
            "phylum":    p.get("phylum", "Other"),
            "class":     p.get("class",  "Unclassified"),
            "order":     p.get("order",  "Unclassified"),
            "genus":     p.get("genus",  "(general)"),
            "title":     p.get("title",  ""),
            "authors":   p.get("authors", ""),
            "venue":     p.get("venue",  ""),
            "year":      p.get("year",   ""),
            "citations": citations.get(doi, 0),
            "doi":       p.get("doi",    ""),
            "url":       p.get("url",    ""),
        })

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Done. {len(out)} entries → {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
