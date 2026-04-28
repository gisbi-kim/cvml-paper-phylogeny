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
| [`TAXONOMY.md`](TAXONOMY.md) | 16 Phylum × ~120 Class × ~400 Order 전체 트리 |
| [`TAXONOMY_CHANGES.md`](TAXONOMY_CHANGES.md) | 초안 vs 최종 비교 |
| [`REFRESH.md`](REFRESH.md) | **데이터/택소노미 갱신 가이드** — 1년 뒤 새 논문 들어오면 어떻게 할지 |
| [`slides/cvml_phylogeny.pdf`](slides/cvml_phylogeny.pdf) | **31-frame Beamer 슬라이드** (한국어, 38년 cohort wall + paradigm shift) |
| [`slides/build_slides.py`](slides/build_slides.py) | 슬라이드 LaTeX 자동 생성 스크립트 |
| [`classify.py`](classify.py) | 1~3 단계 분류기 (Phylum/Class/Order) |
| [`genus_rules.py`](genus_rules.py) | 4단계 분류기 (Genus) |
| [`parse_atlas.py`](parse_atlas.py) | 원본 데이터 파싱 |
| [`make_papers_json.py`](make_papers_json.py) | papers.json 생성 (웹 뷰어용) |
| [`make_tree_data.py`](make_tree_data.py) | tree_data.json 생성 (D3용) |
| [`make_excel.py`](make_excel.py) | 엑셀 변환 |
| [`run_pipeline.py`](run_pipeline.py) | 전체 파이프라인 한 번에 실행 |
| [`data/raw/all_dblp.json`](data/raw/) | 원본 (CV+ML Atlas dump, 112k papers) |

---

## 16 Phylum 번호 순서의 함의

번호(1~16)는 **느슨한 컨셉 그룹핑**이지 우선순위/논문 수/연대와 무관합니다. "픽셀 → 의미 → 학습 → 응용"이라는 연구 스택을 따라 내려가게 배치:

| 번호 | 그룹 | 의도 |
|---|---|---|
| **1~2** | Perception task | 픽셀에서 객체 단위로 잘라내는 가장 직접적인 인식: Detection, Segmentation |
| **3~5** | Visual understanding | 더 풍부한 시각 정보: 3D 기하, 카테고리 인식, 시간(영상) |
| **6~8** | Modeling & synthesis | 무엇을 만들고 무엇을 배울까: 생성, 표현, 비전-언어 |
| **9~10** | Specialized vision | 픽셀 레벨 처리(Low-level)와 사람 중심(Human) 특수화 |
| **11~15** | ML methodology | 어떻게 표현·학습·결정·배포할까: 아키텍처, 학습 전략, 최적화 이론, 강화학습, 효율·견고성 |
| **16** | Application | 의료/자율주행/원격탐사/문서 등 실세계 도메인 |

> 따라서 "1번이 가장 중요"가 아니라 "1번은 가장 입력에 가까운 task". 실제 논문 수로 정렬한 분포는 아래.

## 분류 분포 (실측, 112,183편)

| Phylum | 논문 수 | % |
|---|---:|---:|
| 15. Efficient & Robust ML | 11,993 | 10.7% |
| 12. Training Strategies | 11,404 | 10.2% |
| 3. 3D Vision & Reconstruction | 10,970 | 9.8% |
| 4. Image Recognition & Retrieval | 8,090 | 7.2% |
| 11. Deep Learning Architecture | 7,751 | 6.9% |
| 13. Optimization & Learning Theory | 6,819 | 6.1% |
| 7. Representation Learning | 6,619 | 5.9% |
| 6. Generative Models & Synthesis | 6,488 | 5.8% |
| 5. Video & Motion Understanding | 6,215 | 5.5% |
| 8. Vision-Language & Multimodal | 5,691 | 5.1% |
| 14. Reinforcement Learning & Decision Making | 5,634 | 5.0% |
| 10. Human-centric Vision | 4,506 | 4.0% |
| 1. Object Detection & Localization | 3,859 | 3.4% |
| 16. Application Domains | 3,655 | 3.3% |
| 2. Segmentation | 2,967 | 2.6% |
| 9. Low-level Vision | 2,900 | 2.6% |
| Other / Unclassified | 6,225 | 5.5% |
| Other / Editorial | 397 | 0.4% |
| **합계** | **112,183** | **100.0%** |

> 최대 phylum(15. Efficient & Robust ML)이 10.7% — 균형 잡힌 분포. 이전 버전에서 "12. Training & Learning Methods" 단일 phylum이 21.3%로 비대했는데, 학습 전략 / 최적화 이론 / 강화학습 셋으로 자연스럽게 분리됨.

## Hierarchy 정합성 (2025-04 audit)

전수조사로 같은 개념이 Order(3단)와 Genus(4단)에 중복 등장하던 케이스 28건 + 변형-Order 17건을 정리했습니다.

**주요 수정:**
- **Class==Order 자기충돌 제거 (27건)**: "Image Classification" Class 안에 "Image Classification" Order가 있는 식의 중복을 모두 `General <Class>` 형태로 개명. 예: `Image Classification` Order → `General Image Classification`. (남은 1건 `Unclassified`는 의도된 catchall이라 유지.)
- **NeRF / Gaussian Splatting 변형을 Genus로 강등**: 이전엔 `Dynamic NeRF`, `Human NeRF`, `Efficient NeRF`, `NeRF Editing`, `Large-scale NeRF` 등이 모두 Order로 흩어져 있었고, 같은 변형이 Genus 레벨에도 또 등장(중복). 이제 Order는 `Neural Radiance Fields` 하나로 평탄화하고 변형은 모두 Genus. Gaussian Splatting 동일 처리.
- **Diffusion Models 모달리티별 Order 재구성**: `Diffusion Models`(자기충돌), `Text-to-Image Generation`, `Diffusion-based Image Editing`을 묶어 Order = `Image Diffusion` 하나로 통합. 모달리티별 5개 Order로 정리: `Image Diffusion` / `Video Diffusion` / `3D Diffusion` / `Audio Diffusion` / `Medical Diffusion`. 텍스트 조건/편집/잠재 등은 Genus.
- **3D Scene Understanding Class 분할**: 4,308편 catchall이던 `3D Scene Understanding` Order를 3개로 분리 — `General 3D Vision`, `3D Shape Analysis`, `AR/VR Scene Understanding`.
- **소형 변형 Order → Genus 강등**: `Spatial-Temporal Graph Networks`, `Video Self-supervised Learning`, `Few-shot Action Recognition`, `Efficient Vision Transformers` 등 소수(< 50편) 변형 Order를 부모 Order의 Genus로 이동.

**검증:** 20/20 스모크 테스트 통과. 미분류율 5.9% 유지. Phylum 분포 변동 없음.

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
