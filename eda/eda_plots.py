"""
EDA plot suite (cvml-paper-phylogeny).

Outputs (plotly only — both HTML and PNG):
  figures/01_phylum_stack.png           — Phylum × year stacked area
  figures/02_phylum_small_multiples.png — per-Phylum top-8 Class breakdown
  figures/03_class_heatmap.png          — every Class × 3-yr bucket heatmap
  figures/04_top_classes_drill.png      — top-12 Classes Order drill-down
  interactive/01_phylum_stack.html
  interactive/02_phylum_small_multiples.html
  interactive/03_class_heatmap.html
  interactive/04_top_classes_drill.html
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figures"
INTERACTIVE = ROOT / "interactive"
FIGS.mkdir(exist_ok=True)
INTERACTIVE.mkdir(exist_ok=True)

DATA_PATH = ROOT.parent / "data" / "intermediate" / "papers_classified.json"

# ---------------- Load ----------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

# Normalize: split 'Other' phylum into Editorial / Unclassified pseudo-phyla,
# coerce year to int.
papers = []
for p in raw:
    ph = p["phylum"]
    cl = p["class"]
    if ph == "Other":
        ph = "Other / Editorial" if cl == "Editorial" else "Other / Unclassified"
    papers.append({**p, "phylum": ph, "year": int(p["year"])})

print(f"Loaded {len(papers)} papers")

# Phylum order (must match make_tree_data.py + classify.py numbering)
PHYLA = [
    "1. Object Detection & Localization",
    "2. Segmentation",
    "3. 3D Vision & Reconstruction",
    "4. Image Recognition & Retrieval",
    "5. Video & Motion Understanding",
    "6. Generative Models & Synthesis",
    "7. Representation Learning",
    "8. Vision-Language & Multimodal",
    "9. Low-level Vision",
    "10. Human-centric Vision",
    "11. Deep Learning Architecture",
    "12. Training Strategies",
    "13. Optimization & Learning Theory",
    "14. Reinforcement Learning & Decision Making",
    "15. Efficient & Robust ML",
    "16. Application Domains",
    "Other / Editorial",
    "Other / Unclassified",
]

# Material Design palette — must match make_tree_data.py PHYLUM_COLORS
PHY_COLORS = {
    "1. Object Detection & Localization":          "#e53935",
    "2. Segmentation":                             "#e91e63",
    "3. 3D Vision & Reconstruction":               "#9c27b0",
    "4. Image Recognition & Retrieval":            "#3f51b5",
    "5. Video & Motion Understanding":             "#1976d2",
    "6. Generative Models & Synthesis":            "#0097a7",
    "7. Representation Learning":                  "#388e3c",
    "8. Vision-Language & Multimodal":             "#f57c00",
    "9. Low-level Vision":                         "#795548",
    "10. Human-centric Vision":                    "#ff5722",
    "11. Deep Learning Architecture":              "#607d8b",
    "12. Training Strategies":                     "#5c6bc0",
    "13. Optimization & Learning Theory":          "#7e57c2",
    "14. Reinforcement Learning & Decision Making":"#26a69a",
    "15. Efficient & Robust ML":                   "#f44336",
    "16. Application Domains":                     "#00897b",
    "Other / Editorial":                           "#bcbd22",
    "Other / Unclassified":                        "#7f7f7f",
}

YEARS = list(range(1987, 2026))


# ============================================================
# Plot 1: Phylum stack chart over time
# ============================================================
def plot_1_phylum_stack():
    print("\n[1/4] Phylum stack chart…")

    counts = defaultdict(lambda: defaultdict(int))
    for p in papers:
        counts[p["year"]][p["phylum"]] += 1

    matrix = np.zeros((len(PHYLA), len(YEARS)), dtype=int)
    for j, y in enumerate(YEARS):
        for i, phy in enumerate(PHYLA):
            matrix[i, j] = counts[y].get(phy, 0)

    title = (f"CV+ML papers per year, stacked by Phylum "
             f"(CVPR / NeurIPS / ICML / ICCV / ICLR / ECCV / 3DV — "
             f"{len(papers):,} papers)")

    fig = go.Figure()
    for i, phy in enumerate(PHYLA):
        fig.add_trace(go.Scatter(
            x=YEARS,
            y=matrix[i].tolist(),
            mode="lines",
            name=phy,
            stackgroup="one",
            fillcolor=PHY_COLORS[phy],
            line=dict(width=0.5, color="white"),
            hovertemplate=f"<b>{phy}</b><br>Year: %{{x}}<br>Papers: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Year",
        yaxis_title="Papers per year",
        hovermode="x unified",
        template="plotly_white",
        height=620,
        legend=dict(orientation="v", font=dict(size=10)),
        margin=dict(t=70, l=60, r=20, b=50),
    )
    fig.update_xaxes(range=[1987, 2025], dtick=2)

    out_html = INTERACTIVE / "01_phylum_stack.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"  → {out_html}")

    out_png = FIGS / "01_phylum_stack.png"
    fig.write_image(out_png, width=1500, height=720, scale=2)
    print(f"  → {out_png}")


# ============================================================
# Plot 2: Per-Phylum small multiples (Class breakdown, top 8)
# ============================================================
def plot_2_phylum_small_multiples():
    print("\n[2/4] Per-Phylum small multiples…")

    phylum_class_year = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for p in papers:
        phylum_class_year[p["phylum"]][p["class"]][p["year"]] += 1

    plot_phyla = [p for p in PHYLA if p != "Other / Unclassified"]
    nrows_p, ncols_p = 6, 3
    titles = list(plot_phyla)
    while len(titles) < nrows_p * ncols_p:
        titles.append("")

    fig = make_subplots(
        rows=nrows_p, cols=ncols_p,
        subplot_titles=titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
    )

    palette = (px.colors.qualitative.Pastel
               + px.colors.qualitative.Set2
               + px.colors.qualitative.Pastel1)

    for ax_idx, phy in enumerate(plot_phyla):
        r = ax_idx // ncols_p + 1
        c = ax_idx % ncols_p + 1
        classes = phylum_class_year[phy]
        cls_totals = {cls: sum(yc.values()) for cls, yc in classes.items()}
        top_classes = sorted(cls_totals.items(), key=lambda x: -x[1])[:8]

        for i, (cls, _) in enumerate(top_classes):
            yvals = [classes[cls].get(y, 0) for y in YEARS]
            fig.add_trace(
                go.Scatter(
                    x=YEARS, y=yvals,
                    mode="lines",
                    name=cls,
                    legendgroup=phy,
                    showlegend=False,
                    stackgroup=f"stk_{ax_idx}",
                    fillcolor=palette[i % len(palette)],
                    line=dict(width=0.5, color="white"),
                    hovertemplate=(f"<b>{phy}</b><br>{cls}<br>"
                                   "Year: %{x}<br>Papers: %{y}<extra></extra>"),
                ),
                row=r, col=c,
            )

    # Tint subplot titles by Phylum colour
    for i, ann in enumerate(fig["layout"]["annotations"]):
        if i < len(plot_phyla):
            phy = plot_phyla[i]
            ann["font"] = dict(size=11, color=PHY_COLORS[phy], family="Google Sans")

    fig.update_layout(
        title=dict(
            text="Per-Phylum breakdown by Class over time (top 8 Classes per Phylum)",
            x=0.5, xanchor="center"),
        height=1300,
        showlegend=False,
        template="plotly_white",
        margin=dict(t=80, l=40, r=20, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", range=[1987, 2025])
    fig.update_yaxes(showgrid=True, gridcolor="#eee")

    out_html = INTERACTIVE / "02_phylum_small_multiples.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"  → {out_html}")

    out_png = FIGS / "02_phylum_small_multiples.png"
    fig.write_image(out_png, width=1500, height=1500, scale=2)
    print(f"  → {out_png}")


# ============================================================
# Plot 3: Class heatmap (every Class × 3-yr bucket, log colour)
# ============================================================
def plot_3_class_heatmap():
    print("\n[3/4] Class heatmap…")

    # 3-year buckets (1987-2025)
    bucket_edges = list(range(1987, 2027, 3))
    buckets = []
    for i in range(len(bucket_edges) - 1):
        buckets.append((bucket_edges[i], bucket_edges[i + 1] - 1))
    bucket_labels = [f"{lo}-{hi}" for lo, hi in buckets]

    def yidx(y):
        for i, (lo, hi) in enumerate(buckets):
            if lo <= y <= hi:
                return i
        return -1

    # All (phylum, class) pairs sorted by phylum index, then count desc.
    pc_counts = Counter((p["phylum"], p["class"]) for p in papers)
    pc_sorted = []
    for phy in PHYLA:
        if phy == "Other / Unclassified":
            continue
        in_phy = [(pc, c) for pc, c in pc_counts.items() if pc[0] == phy]
        in_phy.sort(key=lambda x: -x[1])
        pc_sorted.extend(in_phy)

    rows = [pc[0] for pc in pc_sorted]
    nrows, ncols = len(rows), len(buckets)
    matrix = np.zeros((nrows, ncols), dtype=int)
    row_index = {r: i for i, r in enumerate(rows)}
    for p in papers:
        if p["phylum"] == "Other / Unclassified":
            continue
        i = row_index.get((p["phylum"], p["class"]))
        if i is None:
            continue
        j = yidx(p["year"])
        if j >= 0:
            matrix[i, j] += 1

    plot_matrix = np.log10(matrix + 1)
    ylabels = [cls for (_phy, cls) in rows]
    hover = [
        [f"{cls}<br>{phy}<br>{bucket_labels[j]}<br>{matrix[i,j]:,} papers"
         for j in range(ncols)]
        for i, (phy, cls) in enumerate(rows)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=plot_matrix,
        x=bucket_labels,
        y=ylabels,
        colorscale="YlOrRd",
        text=hover,
        hoverinfo="text",
        colorbar=dict(title="log10(1 + papers)"),
    ))

    # Phylum color bands on y axis: use coloured rectangle annotations.
    # Cleaner alternative: just rely on the hover text + group separators.
    # We add a thin colored strip via shapes.
    shapes = []
    cur = 0
    for phy in PHYLA:
        if phy == "Other / Unclassified":
            continue
        cnt = sum(1 for r in rows if r[0] == phy)
        if cnt == 0:
            continue
        shapes.append(dict(
            type="rect", xref="paper", yref="y",
            x0=-0.018, x1=-0.005,
            y0=cur - 0.5, y1=cur + cnt - 0.5,
            fillcolor=PHY_COLORS[phy],
            line=dict(width=0),
            layer="above",
        ))
        cur += cnt

    fig.update_layout(
        title=dict(
            text=(f"Class × year heatmap "
                  f"({nrows} Classes, color = log10(1+count); "
                  "left strip = Phylum)"),
            x=0.5, xanchor="center"),
        xaxis_title="Year (3-year bucket)",
        yaxis_title="Class (grouped by Phylum, ordered by count within Phylum)",
        height=max(900, nrows * 14),
        template="plotly_white",
        shapes=shapes,
        margin=dict(t=70, l=240, r=40, b=60),
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=9))
    fig.update_xaxes(tickangle=-40)

    out_html = INTERACTIVE / "03_class_heatmap.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"  → {out_html}")

    out_png = FIGS / "03_class_heatmap.png"
    fig.write_image(out_png, width=1400, height=max(900, nrows * 14), scale=2)
    print(f"  → {out_png}")


# ============================================================
# Plot 4: Top-12 Classes — Order drill-down
# ============================================================
def plot_4_top_classes_drill():
    print("\n[4/4] Top-12 Classes drill-down…")

    cls_counts = Counter()
    for p in papers:
        if p["phylum"].startswith("Other"):
            continue
        cls_counts[(p["phylum"], p["class"])] += 1
    top12 = cls_counts.most_common(12)

    nrows_p, ncols_p = 4, 3
    titles = [f"{phy} > {cls}<br><sub>{total:,} papers</sub>"
              for ((phy, cls), total) in top12]
    fig = make_subplots(
        rows=nrows_p, cols=ncols_p,
        subplot_titles=titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )

    palette = (px.colors.qualitative.Set3
               + px.colors.qualitative.Pastel2
               + px.colors.qualitative.Pastel1)

    for ax_idx, ((phy, cls), total) in enumerate(top12):
        r = ax_idx // ncols_p + 1
        c = ax_idx % ncols_p + 1
        order_year = defaultdict(lambda: defaultdict(int))
        for p in papers:
            if p["phylum"] == phy and p["class"] == cls:
                order_year[p["order"]][p["year"]] += 1
        ord_totals = {o: sum(yc.values()) for o, yc in order_year.items()}
        top_orders = sorted(ord_totals.items(), key=lambda x: -x[1])[:6]
        for i, (ord_, _) in enumerate(top_orders):
            yvals = [order_year[ord_].get(y, 0) for y in YEARS]
            fig.add_trace(
                go.Scatter(
                    x=YEARS, y=yvals,
                    mode="lines",
                    name=ord_,
                    showlegend=False,
                    stackgroup=f"stk4_{ax_idx}",
                    fillcolor=palette[i % len(palette)],
                    line=dict(width=0.5, color="white"),
                    hovertemplate=(f"<b>{phy} &gt; {cls}</b><br>{ord_}<br>"
                                   "Year: %{x}<br>Papers: %{y}<extra></extra>"),
                ),
                row=r, col=c,
            )

    for i, ann in enumerate(fig["layout"]["annotations"]):
        if i < len(top12):
            phy = top12[i][0][0]
            ann["font"] = dict(size=10, color=PHY_COLORS[phy], family="Google Sans")

    fig.update_layout(
        title=dict(
            text="Top 12 Classes — Order breakdown over time (top 6 Orders per Class)",
            x=0.5, xanchor="center"),
        height=1100,
        showlegend=False,
        template="plotly_white",
        margin=dict(t=80, l=40, r=20, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", range=[1987, 2025])
    fig.update_yaxes(showgrid=True, gridcolor="#eee")

    out_html = INTERACTIVE / "04_top_classes_drill.html"
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"  → {out_html}")

    out_png = FIGS / "04_top_classes_drill.png"
    fig.write_image(out_png, width=1500, height=1300, scale=2)
    print(f"  → {out_png}")


if __name__ == "__main__":
    plot_1_phylum_stack()
    plot_2_phylum_small_multiples()
    plot_3_class_heatmap()
    plot_4_top_classes_drill()
    print("\nDone.")
