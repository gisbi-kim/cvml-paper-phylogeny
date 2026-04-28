**언어**: [English](README.en.md) | 한국어

CV+ML Paper Phylogenetic Taxonomy
===================================

🌐 **Live site**: https://gisbi-kim.github.io/cvml-paper-phylogeny/

> 112,183편의 컴퓨터비전·머신러닝 논문(CVPR / NeurIPS / ICML / ICCV / ICLR / ECCV / 3DV, 1987~2025)을 시맨틱하게 분류해서 생물 계통도(phylogenetic taxonomy)처럼 묶은 결과물.

원 데이터는 **CV+ML Paper Atlas**에서 발췌. **단순 TF-IDF가 아니라 시맨틱 동의어 클러스터** 기반으로 **4단계** (`Phylum > Class > Order > Genus`) 트리에 매핑.

- **모든 논문이 4-depth 라벨링**: Phylum / Class / Order 100%, Genus는 specific rule 매칭(약 50%) + 나머지는 `(general)`.

---

## EDA 시각화

`eda/` 폴더에 시각화 — 자세한 설명은 [eda/README.md](eda/README.md).

---

## 산출물 (Deliverables)

| 파일 | 설명 |
|-----|------|
| **`cvml_taxonomy.xlsx`** | 메인 결과 — 3 시트 (Papers / Taxonomy_Tree / Stats) |
| [`TAXONOMY.md`](TAXONOMY.md) | 14 Phylum × ~110 Class × ~380 Order 전체 트리 |
| [`TAXONOMY_CHANGES.md`](TAXONOMY_CHANGES.md) | 초안 vs 최종 비교 |
| [`classify.py`](classify.py) | 1~3 단계 분류기 (Phylum/Class/Order) |
| [`genus_rules.py`](genus_rules.py) | 4단계 분류기 (Genus) |
| [`parse_atlas.py`](parse_atlas.py) | 원본 데이터 파싱 |
| [`make_papers_json.py`](make_papers_json.py) | papers.json 생성 (웹 뷰어용) |
| [`make_tree_data.py`](make_tree_data.py) | tree_data.json 생성 (D3용) |
| [`make_excel.py`](make_excel.py) | 엑셀 변환 |
| [`run_pipeline.py`](run_pipeline.py) | 전체 파이프라인 한 번에 실행 |
| [`data/raw/all_dblp.json`](data/raw/) | 원본 (CV+ML Atlas dump, 112k papers) |

---

## 분류 분포 요약

총 112,183편. 14 Phylum.

| Phylum | 논문 수 | % |
|--------|---------|---|
| Object Detection & Localization | — | — |
| Segmentation | — | — |
| 3D Vision & Reconstruction | — | — |
| Image Recognition & Retrieval | — | — |
| Video & Motion Understanding | — | — |
| Generative Models & Synthesis | — | — |
| Representation Learning | — | — |
| Vision-Language & Multimodal | — | — |
| Low-level Vision | — | — |
| Human-centric Vision | — | — |
| Deep Learning Architecture | — | — |
| Training & Learning Methods | — | — |
| Efficient & Robust ML | — | — |
| Application Domains | — | — |

*← 분류 실행 후 자동 채워짐*

---

## 작업 흐름 (재현 가능)

```bash
# 원본 데이터를 data/raw/all_dblp.json 에 준비 후:

# 전체 파이프라인 한 번에
python run_pipeline.py

# 또는 단계별
python parse_atlas.py          # 데이터 파싱
python classify.py             # 분류 실행
python make_papers_json.py     # 웹 뷰어용 JSON
python make_tree_data.py       # D3 계층 JSON
python make_excel.py           # 엑셀 생성
```

의존성: `python3` + `openpyxl`만 있으면 됨.

---

## 분류 방법론

작업 지시 핵심 요건:

> "단순히 단어를 분리해서 tf idf 하라는게 아니야. 시맨틱하게 너가 어텐션타서 잘 판단하란 얘기야"
> "생물들 계통도처럼"

→ [`classify.py`](classify.py)는 **동의어 클러스터(synonym cluster)**를 직접 정의:

```python
NERF = ['nerf', 'neural radiance', 'neural implicit', ...]
DIFFUSION = ['diffusion model', 'denoising diffusion', 'ddpm', ...]

if has_any(t, NERF):
    return ('3D Vision & Reconstruction', 'Neural Implicit Representations', 'Neural Radiance Fields (NeRF)')
```

우선순위 규칙(specific → general)으로 cross-cutting 처리.

---

## 한계

- **단일 라벨**: 멀티-필드 논문은 가장 specific한 카테고리로 배정
- **제목만 사용**: abstract 없음 (all_dblp.json 기준)
- **미분류 ~5%**: `Other/Unclassified` 로 두고 검토 가능

---

## 관련 저장소

- [robotics-paper-phylogeny](https://github.com/gisbi-kim/robotics-paper-phylogeny) — 동일 방법론으로 만든 로봇공학 계통도 (7,477편)
- [cvmlpaper-atlas](https://github.com/gisbi-kim/cvmlpaper-atlas) — 원본 데이터 소스
