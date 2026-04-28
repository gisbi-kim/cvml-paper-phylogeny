"""
run_pipeline.py
===============
Full pipeline: parse → classify → make outputs

Usage:
  python run_pipeline.py

Steps:
  1. parse_atlas.py        → data/intermediate/papers_parsed.json
  2. classify.py           → data/intermediate/papers_classified.json
  3. make_papers_json.py   → papers.json  (web viewer)
  4. make_tree_data.py     → tree_data.json (D3 hierarchy)
  5. make_excel.py         → cvml_taxonomy.xlsx
"""
import subprocess
import sys
import time

STEPS = [
    ("Parse atlas data",   ["python", "parse_atlas.py"]),
    ("Run classifier",     ["python", "classify.py"]),
    ("Make papers.json",   ["python", "make_papers_json.py"]),
    ("Make tree_data",     ["python", "make_tree_data.py"]),
    ("Make Excel",         ["python", "make_excel.py"]),
]

def run(label, cmd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {label}  (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n[OK] {label}  ({elapsed:.1f}s)")

if __name__ == "__main__":
    t_total = time.time()
    for label, cmd in STEPS:
        run(label, cmd)
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {time.time()-t_total:.1f}s")
    print(f"{'='*60}")
