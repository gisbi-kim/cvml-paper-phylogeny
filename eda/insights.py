"""
Quantitative insights extraction for cvml-paper-phylogeny.

Reads data/intermediate/papers_classified.json and computes:
1. Per-Phylum first-year, peak-year, 2016-20 → 2021-25 5-yr growth
2. Hot Class TOP 10 in 2020-2025 (by paper count)
3. "사라진" Classes — strong before 2015, near-zero after 2020
4. First-paper-year for emerging CV/ML categories (Diffusion, NeRF, VLM, …)

Outputs:
  eda/insights.json  — machine-readable
  eda/insights.md    — human-readable summary
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "intermediate" / "papers_classified.json"
OUT_JSON = ROOT / "insights.json"
OUT_MD = ROOT / "insights.md"

with open(DATA, "r", encoding="utf-8") as f:
    raw = json.load(f)

# Normalize: split 'Other' phylum into Editorial / Unclassified pseudo-phyla.
papers = []
for p in raw:
    ph = p["phylum"]
    cl = p["class"]
    if ph == "Other":
        ph = "Other / Editorial" if cl == "Editorial" else "Other / Unclassified"
    papers.append({**p, "phylum": ph, "year": int(p["year"])})

TOTAL = len(papers)
print(f"Loaded {TOTAL} papers")

PHYLA_ORDER = [
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
]

insights = {"total_papers": TOTAL}

# ============================================================
# 1) Per-Phylum: first year, peak year, recent-5y growth
# ============================================================
phy_year_counts = defaultdict(Counter)
for p in papers:
    phy_year_counts[p["phylum"]][p["year"]] += 1

phy_stats = {}
for phy in PHYLA_ORDER:
    yc = phy_year_counts[phy]
    if not yc:
        continue
    total = sum(yc.values())
    first_year = min(yc.keys())
    peak_year, peak_count = max(yc.items(), key=lambda x: x[1])
    recent = sum(yc.get(y, 0) for y in range(2021, 2026))
    prior = sum(yc.get(y, 0) for y in range(2016, 2021))
    growth = ((recent - prior) / prior * 100) if prior > 0 else None
    phy_stats[phy] = {
        "total": total,
        "first_year": first_year,
        "peak_year": peak_year,
        "peak_count": peak_count,
        "recent_5y_2021_2025": recent,
        "prior_5y_2016_2020": prior,
        "growth_pct": round(growth, 1) if growth is not None else None,
    }
insights["phylum_stats"] = phy_stats
print("\n=== Phylum stats ===")
for phy, s in phy_stats.items():
    g = f"{s['growth_pct']:+.1f}%" if s["growth_pct"] is not None else "—"
    print(f"  {phy}: peak={s['peak_year']} ({s['peak_count']}), growth={g}")

# ============================================================
# 2) Hot Class in 2020-2025
# ============================================================
recent_cls_counts = Counter()
for p in papers:
    if 2020 <= p["year"] <= 2025 and not p["phylum"].startswith("Other"):
        recent_cls_counts[(p["phylum"], p["class"])] += 1
top_recent = recent_cls_counts.most_common(10)
insights["hot_classes_2020_2025"] = [
    {"phylum": phy, "class": cls, "count": cnt}
    for (phy, cls), cnt in top_recent
]
print("\n=== Hot Classes 2020-2025 ===")
for (phy, cls), cnt in top_recent:
    print(f"  {cnt:>5,}  {phy} > {cls}")

# ============================================================
# 3) Declined Classes: pre-2015 ≥ 20 papers, post-2020 ≤ 10%
# ============================================================
cls_pre = Counter()
cls_post = Counter()
for p in papers:
    key = (p["phylum"], p["class"])
    if p["phylum"].startswith("Other"):
        continue
    if p["year"] < 2015:
        cls_pre[key] += 1
    elif p["year"] >= 2020:
        cls_post[key] += 1

declined = []
for key, pre_count in cls_pre.items():
    post = cls_post.get(key, 0)
    if pre_count >= 20 and post <= max(2, pre_count * 0.10):
        ratio = post / pre_count if pre_count > 0 else 0
        declined.append({
            "phylum": key[0],
            "class": key[1],
            "pre_2015_count": pre_count,
            "post_2020_count": post,
            "decline_ratio": round(ratio, 3),
        })
declined.sort(key=lambda x: x["pre_2015_count"], reverse=True)
insights["declined_classes"] = declined[:15]
print("\n=== Declined Classes (≥20 pre-2015 → ≤10% post-2020) ===")
for d in declined[:10]:
    print(f"  {d['phylum']} > {d['class']}: "
          f"{d['pre_2015_count']} → {d['post_2020_count']}")

# ============================================================
# 4) Emerging categories: first-paper year + cumulative
# ============================================================
emerging_keywords = {
    "Diffusion Models":             ["denoising diffusion", "score-based generative",
                                     "stable diffusion", "latent diffusion", "ddpm"],
    "Vision Transformer (ViT)":     ["vision transformer", " vit:", "vit-b",
                                     "vit-l", "swin transformer", "deit:"],
    "DETR / transformer detection": ["detr:", "deformable detr", "dino-detr"],
    "NeRF / Neural Radiance Field": ["neural radiance field", "nerf:", "nerf-",
                                     "instant ngp", "instant-ngp"],
    "3D Gaussian Splatting":        ["3d gaussian splatting", "gaussian splatting",
                                     "3dgs:", " 3dgs "],
    "Vision-Language Models (CLIP)":["clip:", "contrastive language-image",
                                     "vision-language model", "vision language model",
                                     "blip:", "blip-2", "llava"],
    "Segment Anything (SAM)":       ["segment anything", "sam-2", "sam2:"],
    "Foundation Models":            ["foundation model", "foundational model"],
    "Self-Supervised (modern)":     ["simclr", "moco:", "moco-v", "byol:", "barlow twins",
                                     "masked autoencoder", "masked image model",
                                     "ibot:", "dinov2"],
    "Mamba / State Space":          ["mamba:", "vision mamba", "selective state space",
                                     "vmamba"],
    "LoRA / PEFT":                  ["lora:", "low-rank adaptation",
                                     "parameter-efficient fine"],
    "ControlNet / Diffusion Edit":  ["controlnet", "instructpix2pix",
                                     "dreambooth", "textual inversion"],
}

# Cutoff year per category — anything before this is treated as a false positive
# (the keyword existed in a different sense before the paradigm).
CUTOFF = {
    "Diffusion Models":           2015,
    "Vision Transformer (ViT)":   2018,
    "DETR / transformer detection": 2018,
    "NeRF / Neural Radiance Field": 2018,
    "3D Gaussian Splatting":      2018,
    "Vision-Language Models (CLIP)": 2018,
    "Segment Anything (SAM)":     2020,
    "Foundation Models":          2018,
    "Self-Supervised (modern)":   2018,
    "Mamba / State Space":        2021,
    "LoRA / PEFT":                2018,
    "ControlNet / Diffusion Edit": 2020,
}

emerging = {}
for name, kws in emerging_keywords.items():
    cutoff = CUTOFF.get(name, 0)
    matches = [p for p in papers
               if p["year"] >= cutoff
               and any(kw in p["title"].lower() for kw in kws)]
    if not matches:
        continue
    matches.sort(key=lambda p: p["year"])
    first = matches[0]
    cum_5y = sum(1 for p in matches if 2021 <= p["year"] <= 2025)
    emerging[name] = {
        "first_year": first["year"],
        "first_title": first["title"],
        "first_venue": first.get("venue", ""),
        "total_matched": len(matches),
        "cum_2021_2025": cum_5y,
    }
insights["emerging_categories"] = emerging
print("\n=== Emerging categories ===")
for k, v in emerging.items():
    print(f"  {k}: first={v['first_year']}, total={v['total_matched']}, "
          f"cum 2021-25={v['cum_2021_2025']}")

# ============================================================
# Write outputs
# ============================================================
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(insights, f, ensure_ascii=False, indent=2)
print(f"\n→ {OUT_JSON}")


def md_section(title, body):
    return f"## {title}\n\n{body}\n\n"


lines = ["# Quantitative Insights\n"]
lines.append(f"_From {TOTAL:,} classified papers (1987-2025)._\n\n---\n")

body = ("| Phylum | Total | First | Peak (year, count) | "
        "2016-20 | 2021-25 | Growth |\n")
body += "|---|---:|---:|---|---:|---:|---:|\n"
for phy, s in phy_stats.items():
    g = f"{s['growth_pct']:+.1f}%" if s["growth_pct"] is not None else "—"
    body += (f"| {phy} | {s['total']:,} | {s['first_year']} | "
             f"{s['peak_year']} ({s['peak_count']:,}) | "
             f"{s['prior_5y_2016_2020']:,} | {s['recent_5y_2021_2025']:,} | "
             f"{g} |\n")
lines.append(md_section("1. Phylum별 출현·피크·성장률", body))

body = "**최근 5년 (2020-2025) 최다 논문 Class TOP 10:**\n\n"
for i, h in enumerate(insights["hot_classes_2020_2025"], 1):
    body += f"{i}. **{h['phylum']} > {h['class']}** — {h['count']:,} papers\n"
lines.append(md_section("2. Hot Classes 2020-2025", body))

body = "**Pre-2015에 활발했지만 Post-2020에 거의 사라진 Class (≥20 → ≤10%):**\n\n"
for d in insights["declined_classes"][:10]:
    body += (f"- {d['phylum']} > {d['class']}: "
             f"{d['pre_2015_count']} → {d['post_2020_count']} "
             f"(retain {d['decline_ratio']*100:.1f}%)\n")
lines.append(md_section("3. 사라진 분야", body))

body = "| Category | First year | First paper | Total | 2021-25 |\n"
body += "|---|---:|---|---:|---:|\n"
for k, v in emerging.items():
    title = v["first_title"][:70].replace("|", " ")
    body += (f"| {k} | {v['first_year']} | {title}… | "
             f"{v['total_matched']:,} | {v['cum_2021_2025']:,} |\n")
lines.append(md_section("4. 신생 카테고리 first paper", body))

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"→ {OUT_MD}")
