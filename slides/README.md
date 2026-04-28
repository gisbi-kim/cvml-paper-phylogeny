# Slides — Beamer presentation

CV+ML 계통도 프로젝트의 31-frame Beamer 슬라이드.

다루는 내용:
- 4-depth taxonomy 소개 + methodology
- **38년 cohort wall** (8 × 5-year windows, 1987~2025)
- Paradigm shifts: AlexNet (2012), Transformer (2017), Diffusion (2020), Foundation Models (2023)
- 학회별 색깔 (CVPR · NeurIPS · ICML · ICCV · ICLR · ECCV · 3DV)
- 사라진 분야 (SIFT, BoW, DPM, 고전 CRF) — 데이터로 검증된 ``죽음''
- 교육적 활용 + Live tools 안내

데이터는 `../data/intermediate/papers_classified.json`에서 매번 새로 계산됩니다 — cohort 카운트, top class rank, emerging class delta 모두 현재 커밋된 스냅샷 기준.

## 빌드

컴파일된 PDF (`cvml_phylogeny.pdf`)가 이미 커밋되어 있으므로 **그냥 열어서 보면 됩니다**. 데이터 갱신 후 또는 `build_slides.py`를 수정한 경우에만 다시 빌드하세요.

### Option A — Docker (로컬 TeX 설치 불필요, 가장 권장)

```bash
# 1. (Optional) 현재 데이터셋으로 .tex 재생성
python3 slides/build_slides.py

# 2. texlive 이미지로 컴파일 (첫 pull은 ~8GB, 이후 캐시됨)
cd slides
docker run --rm -v "$(pwd):/work" -w /work texlive/texlive:latest \
  bash -c "xelatex -interaction=nonstopmode cvml_phylogeny.tex && \
           xelatex -interaction=nonstopmode cvml_phylogeny.tex"
```

Windows PowerShell에서:

```powershell
cd slides
docker run --rm -v "${PWD}:/work" -w /work texlive/texlive:latest `
  bash -c "xelatex -interaction=nonstopmode cvml_phylogeny.tex && xelatex -interaction=nonstopmode cvml_phylogeny.tex"
```

`texlive/texlive:latest-small`도 작동하지만 **Korean (kotex) 미포함**이므로 풀 `latest` 태그를 써야 합니다.

### Option B — 로컬 XeLaTeX

TeX 배포판(TeXLive / MacTeX / MiKTeX) + `kotex` + `cjk-ko` 패키지 필요.

```bash
python3 slides/build_slides.py    # optional, only if data changed
cd slides
xelatex cvml_phylogeny.tex        # TOC 해결을 위해 두 번 실행
xelatex cvml_phylogeny.tex
```

`lualatex`도 작동 (kotex이 엔진 자동 감지).

### Option C — Overleaf

`.tex` 파일을 새 프로젝트에 드래그 → Menu → Settings → Compiler를 **XeLaTeX**으로 설정 → Recompile.

## 편집

Narrative 음성은 `build_slides.py` 안에 있습니다:

- `COHORT_HEADLINES` — cohort별 제목 + tagline. 톤 변경 시 여기 편집.
- `make_paradigm_shifts_frame()` — AlexNet/Transformer/Foundation Models 등 deep-dive frame.
- `make_validation_frames()` — 학회별 색깔, rank race, 교육적 활용, Limitations.

숫자(논문 카운트, rank, dominant class)는 매번 live JSON에서 재계산되므로 손으로 갱신할 필요 없습니다.

## 컴파일 산출물

- `cvml_phylogeny.tex` — `build_slides.py`가 생성하는 LaTeX 소스 (커밋됨, 재생성 가능)
- `cvml_phylogeny.pdf` — 31 pages, 한국어 (커밋됨)

`.aux` `.log` `.nav` `.out` `.snm` `.toc` 등 보조 파일은 `.gitignore`로 제외.
