"""
make_tree_data.py
=================
papers_classified.json → tree_data.json  (D3 hierarchy용)

Output: nested JSON  { name, value, color, children: [...] }
  root → phylum → class → order → genus
"""
import json
from collections import defaultdict
from pathlib import Path

IN  = Path("data/intermediate/papers_classified.json")
OUT = Path("tree_data.json")

# Material Design 색상 (14 Phyla) — matches numbered phylum names from classify.py
PHYLUM_COLORS = {
    "1. Object Detection & Localization":  "#e53935",
    "2. Segmentation":                     "#e91e63",
    "3. 3D Vision & Reconstruction":       "#9c27b0",
    "4. Image Recognition & Retrieval":    "#3f51b5",
    "5. Video & Motion Understanding":     "#1976d2",
    "6. Generative Models & Synthesis":    "#0097a7",
    "7. Representation Learning":          "#388e3c",
    "8. Vision-Language & Multimodal":     "#f57c00",
    "9. Low-level Vision":                 "#795548",
    "10. Human-centric Vision":            "#ff5722",
    "11. Deep Learning Architecture":      "#607d8b",
    "12. Training & Learning Methods":     "#5c6bc0",
    "13. Efficient & Robust ML":           "#f44336",
    "14. Application Domains":             "#00897b",
    "Other":                               "#9e9e9e",
}

def main():
    print(f"Loading {IN} ...")
    with open(IN, encoding="utf-8") as f:
        papers = json.load(f)

    # Build count tree: phylum → class → order → genus → count
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    for p in papers:
        ph = p.get("phylum", "Other")
        cl = p.get("class",  "Unclassified")
        od = p.get("order",  "Unclassified")
        gn = p.get("genus",  "(general)")
        tree[ph][cl][od][gn] += 1

    root_children = []
    total = 0

    for ph, classes in sorted(tree.items(), key=lambda x: -sum(
            sum(sum(g.values()) for g in o.values()) for o in x[1].values())):
        ph_count = 0
        ph_children = []

        for cl, orders in sorted(classes.items(), key=lambda x: -sum(
                sum(g.values()) for g in x[1].values())):
            cl_count = 0
            cl_children = []

            for od, genera in sorted(orders.items(), key=lambda x: -sum(x[1].values())):
                od_count = 0
                od_children = []

                for gn, cnt in sorted(genera.items(), key=lambda x: -x[1]):
                    od_children.append({"name": gn, "value": cnt})
                    od_count += cnt

                # Internal nodes get no `value` — d3.hierarchy().sum()
                # rolls children up, so writing it here would double-count.
                cl_children.append({
                    "name": od,
                    "children": od_children,
                })
                cl_count += od_count

            ph_children.append({
                "name": cl,
                "children": cl_children,
            })
            ph_count += cl_count

        color = PHYLUM_COLORS.get(ph, "#9e9e9e")
        root_children.append({
            "name":  ph,
            "color": color,
            "children": ph_children,
        })
        total += ph_count

    root = {
        "name":     "CV+ML",
        "color":    "#1a73e8",
        "children": root_children,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)

    print(f"Done. {total} papers → {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    print("\nPhylum distribution:")
    def _sum(node):
        return node["value"] if "value" in node else sum(_sum(c) for c in node.get("children", []))
    for ch in root["children"]:
        cnt = _sum(ch)
        pct = cnt / total * 100
        print(f"  {ch['name']:<45} {cnt:>6}  {pct:5.1f}%")

if __name__ == "__main__":
    main()
