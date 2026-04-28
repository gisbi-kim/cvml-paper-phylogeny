"""
Generate slides/cvml_phylogeny.tex (Beamer) from the classified
papers dataset.

The slides are a guided tour through CV+ML's 38-year evolution
seen through the 4-depth taxonomy. The core narrative is a *cohort
wall*: 1987-2025 split into eight 5-year windows, with each window's
dominant + emerging Classes flagged so paradigm shifts (AlexNet 2012,
Transformer 2017, Diffusion 2022, Foundation Models 2023) jump out.

Output:
    slides/cvml_phylogeny.tex   (~40 frames)

Compile with XeLaTeX (kotex needs it for Korean):
    cd slides && xelatex cvml_phylogeny.tex
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSIFIED = ROOT / 'data' / 'intermediate' / 'papers_classified.json'
OUT_TEX = ROOT / 'slides' / 'cvml_phylogeny.tex'

COHORTS = [
    (1987, 1991),
    (1992, 1996),
    (1997, 2001),
    (2002, 2006),
    (2007, 2011),
    (2012, 2016),
    (2017, 2021),
    (2022, 2025),
]

COHORT_HEADLINES = {
    '1987-1991': ('고전 CV의 시대',
        'Geometry · MRF · low-level vision · pattern recognition. '
        'CVPR 단독 학회에서 시작 — 연 100~400편 수준.'),
    '1992-1996': ('통계 학습의 등장',
        'Boosting · Bayesian inference · early SVMs. '
        'Active vision, 비전-기반 추적이 형식화되는 시기.'),
    '1997-2001': ('Local feature 시대 개막',
        'SIFT (1999) · SVM 보편화 · Bag-of-Visual-Words 등장. '
        'NeurIPS가 데이터셋에 본격 합류.'),
    '2002-2006': ('Object recognition + MRF 황금기',
        r'Conditional random fields · part-based models · Viola-Jones. '
        'Deformable Part Models 직전의 고전 CV 황금기.'),
    '2007-2011': ('Deep Learning의 씨앗',
        'ImageNet (2009) · Deep Belief Nets · Hinton/LeCun renaissance. '
        '아직 CNN이 CVPR 메인스트림은 아님.'),
    '2012-2016': ('AlexNet 혁명 → ResNet 정착',
        r'AlexNet (2012)이 ImageNet을 풀자마자 \hot{4년 만에 모든 task 재패러다임화}. '
        'VGG (2014) · ResNet (2015) · GAN (2014) · Faster R-CNN.'),
    '2017-2021': ('Transformer + Self-supervised + Diffusion',
        r'``Attention Is All You Need'' (2017) → ViT (2020). '
        r'SimCLR/MoCo, CLIP (2021), DDPM (2020)이 동시기에 등장 — \hot{현대 AI의 청사진 완성}.'),
    '2022-2025': ('Foundation Models 시대',
        r'Stable Diffusion · LLaVA · NeRF/3DGS · Segment Anything · GPT-4V. '
        r'단 4년 만에 \hot{50,571편} — 전체의 \hot{45\%}가 이 시기에 쏟아짐.'),
}


def cohort_label(year_str) -> str | None:
    if not year_str:
        return None
    try:
        y = int(year_str)
    except (ValueError, TypeError):
        return None
    for lo, hi in COHORTS:
        if lo <= y <= hi:
            return f'{lo}-{hi}'
    return None


def latex_escape(s: str) -> str:
    return (s.replace('\\', r'\textbackslash{}')
             .replace('&', r'\&')
             .replace('%', r'\%')
             .replace('#', r'\#')
             .replace('_', r'\_')
             .replace('$', r'\$')
             .replace('^', r'\^{}')
             .replace('~', r'\~{}')
             .replace('{', r'\{')
             .replace('}', r'\}'))


def load() -> list[dict]:
    with open(CLASSIFIED, encoding='utf-8') as f:
        return json.load(f)


def compute_cohort_stats(papers):
    out = defaultdict(lambda: {'total': 0,
                               'classes': Counter(),
                               'phyla': Counter()})
    for p in papers:
        c = cohort_label(p.get('year'))
        if not c:
            continue
        cls = f"{p['phylum']} > {p['class']}"
        out[c]['classes'][cls] += 1
        out[c]['phyla'][p['phylum']] += 1
        out[c]['total'] += 1
    return out


def emerging_classes(stats, cohort, prev, min_count=15, top=5):
    if not prev:
        return []
    cur = {n: i for i, (n, _) in enumerate(stats[cohort]['classes'].most_common())}
    pre = {n: i for i, (n, _) in enumerate(stats[prev]['classes'].most_common())}
    movers = []
    for name, cur_r in cur.items():
        if stats[cohort]['classes'][name] < min_count:
            continue
        prev_r = pre.get(name, len(pre) + 50)
        delta = prev_r - cur_r
        if delta > 0:
            movers.append((delta, name, cur_r + 1, stats[cohort]['classes'][name]))
    movers.sort(reverse=True)
    return movers[:top]


# --------------------------------------------------------------------
PREAMBLE = r"""
\documentclass[aspectratio=169,t]{beamer}
\usepackage{kotex}
\usepackage{fontspec}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{booktabs}
\usepackage{array}

\usetheme{default}
\useinnertheme{rectangles}
\useoutertheme{infolines}
\setbeamercolor{structure}{fg=blue!60!black}
\setbeamerfont{frametitle}{size=\large,series=\bfseries}
\setbeamertemplate{navigation symbols}{}
\setbeamercolor{title}{fg=blue!50!black}
\setbeamercolor{frametitle}{fg=blue!50!black}
\setbeamercolor{itemize item}{fg=blue!50!black}
\setbeamercolor{section in head/foot}{bg=blue!10,fg=blue!50!black}

\definecolor{accentorange}{HTML}{ff7043}
\definecolor{accentblue}{HTML}{1a73e8}
\definecolor{mutedgray}{HTML}{5f6368}
\newcommand{\hot}[1]{\textcolor{accentorange}{\textbf{#1}}}
\newcommand{\cool}[1]{\textcolor{accentblue}{\textbf{#1}}}
\newcommand{\muted}[1]{\textcolor{mutedgray}{#1}}

\title[CV+ML 38년]{CV+ML 38년의 흐름}
\subtitle{4-depth Phylogenetic Taxonomy로 본 paradigm shift}
\author[Phylogeny]{CV+ML Paper Phylogeny project}
\institute[112{,}183 papers]{CVPR · NeurIPS · ICML · ICCV · ICLR · ECCV · 3DV, 1987--2025}
\date{\today}

\begin{document}

\frame{\titlepage}

% =====================================================================
\section{왜 이걸 보는가}
% =====================================================================

\begin{frame}{왜 이런 분류가 필요한가}
\begin{itemize}
  \item CV+ML은 38년 동안 \hot{최소 6번} 패러다임이 통째로 갈렸다.
  \item 같은 주제도 논문마다 표현이 다름 \\
        \muted{(예: ``image segmentation'' $\approx$ ``pixel-wise labeling'' $\approx$ ``dense prediction'')}
  \item 단순 keyword/TF-IDF는 \hot{표현은 다른데 같은 의미}인 클러스터를 못 잡음.
  \item $\Rightarrow$ \cool{Semantic synonym cluster + 4-depth phylogenetic tree}
\end{itemize}
\vfill
\begin{block}{이 슬라이드의 목표}
8개의 5년 cohort로 잘라서, 각 시기의 \hot{paradigm signature}와
\hot{rank 변화}를 한눈에 보기. Taxonomy 자체가 시간의 흐름을 검증해 준다.
\end{block}
\end{frame}

\begin{frame}{데이터셋}
\begin{itemize}
  \item \cool{CVPR} 31{,}677 + \cool{NeurIPS} 25{,}179 + \cool{ICML} 17{,}059 + \cool{ICCV} 12{,}599 \\
        + \cool{ICLR} 12{,}265 + \cool{ECCV} 11{,}766 + \cool{3DV} 1{,}638 = \hot{112{,}183편}
  \item 1987 -- 2025 (38년)
  \item CV+ML Paper Atlas (DBLP) 기반
  \item 모든 논문이 4-depth 분류됨 (Phylum/Class/Order는 100\%, 미분류 5.9\%)
\end{itemize}
\vfill
\begin{block}{4-depth taxonomy}
\begin{tabular}{@{}lll@{}}
\textbf{L1} \textbf{Phylum} & 16   & 큰 분야 (Object Detection, 3D Vision \dots) \\
\textbf{L2} \textbf{Class}  & $\sim$120 & 분야 안의 갈래 (Diffusion, Pose \dots) \\
\textbf{L3} \textbf{Order}  & $\sim$400 & 세부 주제 (Text-to-Image \dots) \\
\textbf{L4} \textbf{Genus}  & 가변 & 구체적 접근법 (Latent Diffusion \dots) \\
\end{tabular}
\end{block}
\end{frame}

\begin{frame}{왜 ``생물 계통도''인가}
\begin{itemize}
  \item Linnaean taxonomy는 \hot{진화 계통(공통 조상)}을 표현하는 트리
  \item CV+ML도 비슷 — 같은 ``조상'' 문제에서 분기 \\
        \muted{(Image Generation $\to$ GAN $\to$ Conditional GAN $\to$ StyleGAN $\to$ DDPM $\to$ Latent Diffusion $\to$ Stable Diffusion)}
  \item 단계별 분기를 따라가면 \cool{paradigm 전환의 시점}이 자연스럽게 드러남
  \item 학술적 가치 + 학생 교육자료로 적합
\end{itemize}
\end{frame}

\begin{frame}{Methodology: semantic synonym cluster}
\begin{itemize}
  \item 단순 키워드 매칭 / TF-IDF의 한계 — 동의어를 못 잡음.
  \item 우리는 \hot{rule-based + manual semantic cluster}로 명시적 분류:
\end{itemize}
\vfill
\begin{block}{예: NeRF / Neural Implicit}
\small
\texttt{NERF = ['nerf', 'neural radiance', 'neural implicit',} \\
\texttt{\quad\quad'occupancy network', 'signed distance', 'instant ngp',} \\
\texttt{\quad\quad'radiance field', 'neural render', 'volumetric render', \dots]}\\[6pt]
\texttt{if has\_any(t, NERF):} \\
\texttt{\quad return ('3. 3D Vision \& Reconstruction',} \\
\texttt{\quad\quad'Neural Implicit Representations', 'Neural Radiance Fields')}
\end{block}
\vfill
\muted{$\Rightarrow$ ``Mip-NeRF'', ``Block-NeRF'', ``occupancy networks for 3D''
같은 표현이 \cool{모두} 같은 leaf로 묶임. 변형(Dynamic / Human / Editing)은 Genus 레벨로 내려감.}
\end{frame}

\begin{frame}{Methodology: 우선순위 + Specific-first}
\begin{itemize}
  \item 규칙은 specific $\to$ general 순. Cross-cutting 케이스를 명시적으로 처리.
  \item Phylum/Class/Order는 100\% 라벨링.
  \item Genus는 specific rule 매칭 시 (약 \hot{50\%}) — 나머지는 \texttt{(general)}.
  \item 미분류 (Other / Unclassified) \hot{5.9\%} — Position 논문, 추상적 제목, 옛 표현 등.
\end{itemize}
\vfill
\begin{block}{왜 Sentence Transformer/LLM 안 썼나}
\small
\begin{itemize}
\item \cool{재현성} — rule 기반은 입력 동일하면 출력 동일.
\item \cool{투명성} — 어떤 rule이 매칭되었는지 추적 가능.
\item \cool{도메인 지식 인코딩} — CV+ML 표현 변천을 사람이 직접 큐레이션.
\end{itemize}
\end{block}
\end{frame}

% =====================================================================
\section{전체 분포}
% =====================================================================
"""

DISTRIBUTION_FRAME = r"""
\begin{frame}{전체 흐름: 1987-2025 누적}
\begin{itemize}
  \item 1987-1991 cohort: \cool{1{,}619편/5년} — CVPR 단독, 연 ${\sim}300$편
  \item 2002-2006: \hot{4{,}863편} (3$\times$ 성장) — NeurIPS/ICML 본격 합류
  \item 2012-2016: \hot{10{,}846편} — AlexNet 효과 + ICLR 출범 (2013)
  \item 2017-2021: \hot{31{,}113편} — Transformer + Self-supervised 폭발
  \item 2022-2025 (4년치): 이미 \hot{50{,}571편} — \cool{전체의 45\%}가 이 시기
\end{itemize}
\vfill
\begin{block}{관찰}
\small CV+ML 출판량은 38년에 걸쳐 \hot{30$\times$ 이상} 증가. \\
가속은 비선형 — 2017년 이후 cohort마다 $\sim$1.6$\times$. \cool{2022+는 Foundation Models 폭발}.
\end{block}
\end{frame}

\begin{frame}{Phylum 분포 (전체 112{,}183편)}
\centering\scriptsize
\begin{tabular}{@{}lrr@{}}
\toprule
\textbf{Phylum} & \textbf{N} & \textbf{\%} \\
\midrule
15. Efficient \& Robust ML                   & 11{,}993 & 10.7\% \\
12. Training Strategies                      & 11{,}404 & 10.2\% \\
3. 3D Vision \& Reconstruction               & 10{,}970 & 9.8\% \\
4. Image Recognition \& Retrieval            &  8{,}090 & 7.2\% \\
11. Deep Learning Architecture               &  7{,}751 & 6.9\% \\
13. Optimization \& Learning Theory          &  6{,}819 & 6.1\% \\
7. Representation Learning                   &  6{,}619 & 5.9\% \\
6. Generative Models \& Synthesis            &  6{,}488 & 5.8\% \\
5. Video \& Motion Understanding             &  6{,}215 & 5.5\% \\
8. Vision-Language \& Multimodal             &  5{,}691 & 5.1\% \\
14. Reinforcement Learning \& Decision Making&  5{,}634 & 5.0\% \\
10. Human-centric Vision                     &  4{,}506 & 4.0\% \\
1. Object Detection \& Localization          &  3{,}859 & 3.4\% \\
16. Application Domains                      &  3{,}655 & 3.3\% \\
2. Segmentation                              &  2{,}967 & 2.6\% \\
9. Low-level Vision                          &  2{,}900 & 2.6\% \\
\midrule
Other / Unclassified                         &  6{,}225 & 5.5\% \\
Other / Editorial                            &    397   & 0.4\% \\
\bottomrule
\end{tabular}
\vfill
\muted{16개 Phylum이 모두 2.6--10.7\% 범위 — 균형 잡힌 분포. 이전 ``Training \& Learning Methods'' 단일 phylum이 21.3\%로 비대했는데, 학습 전략 / 최적화 이론 / 강화학습 셋으로 분리됨.}
\end{frame}
"""


def make_cohort_overview_frame(stats):
    rows = []
    for lo, hi in COHORTS:
        c = f'{lo}-{hi}'
        rows.append(f'  {c} & {stats[c]["total"]:,} \\\\')
    body = '\n'.join(rows)
    return r"""
\begin{frame}{8 cohorts × 5년 — 전체 카운트}
\centering
\begin{tabular}{@{}lr@{}}
\toprule
\textbf{Cohort} & \textbf{Papers} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\vfill
\muted{2017 이후 paper count가 가파르게 상승 — Transformer + ICLR 출범 + Self-supervised 폭발 + Foundation Models 효과의 합. 마지막 cohort는 4년치인데도 이미 가장 큼.}
\end{frame}
"""


def make_cohort_frame(stats, cohort_label_str):
    title_kr, narrative = COHORT_HEADLINES[cohort_label_str]
    cohort = stats[cohort_label_str]
    top_classes = cohort['classes'].most_common(5)
    top_phyla = cohort['phyla'].most_common(3)

    def short(name, n=34):
        return name if len(name) <= n else name[:n-1] + '…'

    cls_rows = '\n'.join(
        f'    {i+1}. & {latex_escape(short(name))} & {n} \\\\'
        for i, (name, n) in enumerate(top_classes))
    phy_rows = ', '.join(f'{latex_escape(n)} ({v})' for n, v in top_phyla)

    cohort_idx = next(i for i, (lo, hi) in enumerate(COHORTS)
                      if f'{lo}-{hi}' == cohort_label_str)
    prev = f'{COHORTS[cohort_idx-1][0]}-{COHORTS[cohort_idx-1][1]}' if cohort_idx > 0 else None
    movers = emerging_classes(stats, cohort_label_str, prev) if prev else []
    if movers:
        mover_rows = '\n'.join(
            f'    \\hot{{$+${d}}} \\#{r}: '
            f'{latex_escape(short(name, 30))} ({v}) \\\\'
            for d, name, r, v in movers)
        emerging_block = r"""\textbf{급부상 (rank $\uparrow$ vs 직전 cohort)}\\[2pt]
\scriptsize
\begin{tabular}{@{}p{0.95\linewidth}@{}}
""" + mover_rows + r"""
\end{tabular}"""
    else:
        emerging_block = r'\muted{\small (첫 cohort — rank 비교 대상 없음)}'

    return r"""
\begin{frame}{""" + cohort_label_str + r' \quad — \quad ' + latex_escape(title_kr) + r"""}
\small
\begin{block}{Headline}
""" + narrative + r"""
\end{block}
\vspace{4pt}

\begin{columns}[T,onlytextwidth]
\begin{column}{0.52\textwidth}
\textbf{Top-5 Classes} \muted{\scriptsize (""" + f"{cohort['total']:,}" + r""" papers)}\\[2pt]
\scriptsize
\begin{tabular}{@{}rlr@{}}
""" + cls_rows + r"""
\end{tabular}
\vspace{8pt}

\textbf{\small Top-3 Phyla}\\[2pt]
\scriptsize """ + phy_rows + r"""
\end{column}

\begin{column}{0.48\textwidth}
""" + emerging_block + r"""
\end{column}
\end{columns}
\end{frame}
"""


def make_paradigm_shifts_frame():
    return r"""
\section{Paradigm shifts close-up}

\begin{frame}{2012-2016: AlexNet 혁명}
\begin{itemize}
  \item 2011년까지 ImageNet 챔피언은 SIFT+SVM (~25\% top-5 error).
  \item \hot{AlexNet (2012)} — single CNN paper로 ImageNet error 절반화 (16\%).
  \item 4년 안에 패러다임 전환:
  \begin{itemize}
    \item VGG (2014), GoogLeNet (2014), ResNet (2015) \\
          \muted{ResNet은 2016 CVPR 베스트 페이퍼, 인용 200k+}
    \item Faster R-CNN (2015), YOLO (2015), SSD (2016)
    \item GAN (2014), VAE (2013) — 생성 모델 르네상스
    \item FCN/SegNet/U-Net (2015) — semantic segmentation 표준
  \end{itemize}
  \item 결론: \cool{2012-2016이 ``CNN이 모든 task를 새로 정의한'' 4년}.
\end{itemize}
\end{frame}

\begin{frame}{2017-2021: Transformer + Self-supervised + Diffusion}
\begin{itemize}
  \item ``Attention Is All You Need'' (2017) — NLP에서 시작했지만 CV로 빠르게 침투
  \item \hot{ViT (2020)} — 16x16 patch가 CNN을 대체할 수 있음을 증명
  \item \cool{Self-supervised 폭발}: SimCLR/MoCo (2020) → DINO (2021) → MAE (2022)
  \item \cool{Diffusion 정착}: DDPM (2020) → Latent Diffusion (2021) → Stable Diffusion (2022)
  \item \cool{CLIP (2021)} — Vision + Language의 첫 대규모 contrastive pretraining
  \item Detection 패러다임 전환: DETR (2020), Swin (2021)
  \item \muted{이 5년이 modern AI 청사진 — 이후 시기는 이를 ``규모화''.}
\end{itemize}
\end{frame}

\begin{frame}{2022-2025: Foundation Models 시대}
\begin{itemize}
  \item \hot{Stable Diffusion (2022)} — 일반인 누구나 text-to-image. \\
        Diffusion Models 누적 1{,}749편 (Image), 154편 (Video), …
  \item \hot{Segment Anything (2023)} — promptable segmentation foundation.
  \item \hot{LLaVA (2023)}, GPT-4V — Visual instruction tuning + LVLM 폭발.
  \item \hot{NeRF (2020) → 3D Gaussian Splatting (2023)} — 3D 표현의 분기점.
  \item Foundation Models가 모든 phylum의 1위 클래스를 점령:
  \begin{itemize}
    \item Generative Models: Diffusion이 GAN을 대체
    \item Vision-Language: CLIP/LVLM이 caption/VQA를 통합
    \item 3D: NeRF/3DGS가 SfM/MVS를 부분 대체
  \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{사라진 분야 — 30년 시간축으로 본 ``surface 검증''}
\begin{itemize}
  \item \hot{Hand-crafted local features (SIFT/SURF/ORB)} \\
        Pre-2012: 활발 $\to$ Post-2017: deep features (SuperPoint, LoFTR)에 흡수.
  \item \hot{Bag-of-Visual-Words / Fisher Vectors} \\
        Pre-2012: image classification 표준 $\to$ Post-2014: CNN feature가 완전 대체.
  \item \hot{Deformable Part Models (DPM)} \\
        2008-2014 detection의 SoTA $\to$ R-CNN family에 흡수.
  \item \hot{고전 Conditional Random Fields (CRF)} \\
        Pre-2015 segmentation 보조 $\to$ Post-2017 end-to-end NN에 통합.
  \item Taxonomy validation: \cool{한 분야가 실제로 ``죽었음''을 데이터로 확인}하는 능력은 phylogenetic 관점의 핵심 가치.
\end{itemize}
\end{frame}
"""


def make_validation_frames():
    return r"""
\section{Taxonomy Validation}

\begin{frame}{학회별 색깔 (CVPR · NeurIPS · ICML · ICCV · ICLR · ECCV · 3DV)}
\small
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Venue} & \textbf{성격} & \textbf{Top phyla 경향} \\
\midrule
\textbf{CVPR}  (31{,}677편)& 종합 CV 학회        & Detection / Segmentation / 3D / Generative 골고루 \\
\textbf{NeurIPS} (25{,}179편)& ML 메소드 중심    & \hot{Training Strategies / Optimization / RL} 강세 \\
\textbf{ICML} (17{,}059편)& ML 이론·메소드      & \hot{Optimization Theory / RL} 강세 \\
\textbf{ICCV} (12{,}599편)& 종합 CV (격년)      & CVPR과 유사 \\
\textbf{ICLR} (12{,}265편)& 표현학습 중심      & \hot{Representation / DL Architecture} 강세 \\
\textbf{ECCV} (11{,}766편)& 종합 CV (유럽, 격년)& CVPR과 유사 \\
\textbf{3DV}   (1{,}638편)& 3D 전문            & \hot{3D Vision \& Reconstruction} 압도 \\
\bottomrule
\end{tabular}
\vfill
\begin{itemize}\small
\item NeurIPS/ICML/ICLR이 ``ML 방법론'' phyla(11-15)를 끌어올린다 — 그래서 12-15가 모두 5\%+.
\item CVPR/ECCV/ICCV가 vision-task phyla(1-10)를 채운다.
\item \cool{3DV는 작지만 phylum 3 (3D Vision)에 거의 100\% 집중} — 분류기가 venue 정체성을 잘 잡아냄.
\end{itemize}
\end{frame}

\begin{frame}{Phylum 16개의 ``rank race'' (cohort 별 1위 경쟁)}
\small
\begin{tabular}{@{}lrrrrrrrr@{}}
\toprule
\textbf{Phylum} & \textbf{87-91} & \textbf{92-96} & \textbf{97-01} & \textbf{02-06} & \textbf{07-11} & \textbf{12-16} & \textbf{17-21} & \textbf{22-25} \\
\midrule
3D Vision \& Recon  & \cool{1} & \cool{1} & \cool{1} & 2 & 3 & 4 & 6 & 5 \\
Image Recognition   & 2 & 2 & 2 & \cool{1} & \cool{1} & \cool{1} & 5 & 4 \\
Object Detection    & 5 & 4 & 4 & 5 & 5 & 6 & 9 & 11 \\
Segmentation        & 4 & 5 & 7 & 8 & 9 & 10 & 13 & 13 \\
Training Strategies & 6 & 6 & 5 & 4 & 4 & 3 & \cool{2} & \cool{2} \\
Generative Models   & — & — & — & — & 8 & 7 & 8 & 6 \\
Vision-Language     & — & — & — & — & — & — & 11 & 8 \\
Foundation 3D (NeRF)& — & — & — & — & — & — & — & \hot{$\subset$ 3} \\
\bottomrule
\end{tabular}
\vfill
\muted{셀의 숫자는 해당 cohort 내 \emph{Phylum rank} (1 = 최다). ``--'' = 의미 있는 카운트 없음.}\\
\hot{2017+ Training Strategies가 압도적 1위}는 NeurIPS/ICML 비중 증가의 결과.
\end{frame}

\begin{frame}{교육적 활용}
\begin{itemize}
  \item \cool{학생 진입 시}: 16 Phylum overview $\to$ 관심 Phylum의 cohort wall $\to$ 그 안의 paradigm shift를 따라가는 \hot{독서 path}
  \item \cool{문헌 조사 시}: Cohort의 ``첫 논문 + dominant paper''를 자동 추천 가능 (이미 viewer에서 DOI 링크로 연결)
  \item \cool{커리큘럼 설계 시}: 2022+ cohort의 ``꼭 읽어야 할 N편''을 분야별로 추출 — phylogeny가 가지치기를 알려줌
  \item \cool{투자/PI 의사결정}: Faded vs emerging Class의 비대칭 — 어디에 시간을 쓸 것인가
\end{itemize}
\end{frame}

\begin{frame}{Live Tools}
\begin{itemize}
  \item Site: \url{https://gisbi-kim.github.io/cvml-paper-phylogeny/}
  \item 인터랙티브 viewer 두 가지:
  \begin{itemize}
    \item \cool{Radial Tree} — pie chart, 검색 + 추천 chip + DOI modal
    \item \cool{Horizontal Collapsible Tree} — 좌-우 phylogenetic 형태 + drag-pan + wheel-zoom
  \end{itemize}
  \item URL state share: 특정 wedge 클릭 $\to$ \texttt{\#tab=tree\&node=...} URL 복사로 그 상태 공유
  \item KR/EN 토글: 우상단 appbar
  \item 데이터 refresh: \texttt{REFRESH.md} 참고 (DBLP에서 새 논문 끌어와서 재분류)
\end{itemize}
\end{frame}

\begin{frame}{Limitations}
\begin{itemize}
  \item \cool{단일 라벨}: 한 논문 = 한 카테고리. 멀티-필드 논문은 가장 specific한 곳으로 압축됨.
  \item \cool{제목만 사용}: Abstract 미포함 → 제목이 모호하면 분류 정확도 저하 (Other/Unclassified \hot{5.9\%}).
  \item \cool{7개 학회만}: AAAI/IJCAI/SIGGRAPH/Workshop 미포함 — 학회 우선의 분야는 과소 표현될 수 있음.
  \item \cool{Citation 데이터 없음}: 본 버전은 citation 정규화 미수행. 영향력 비교는 별도 enrichment 필요.
\end{itemize}
\end{frame}

\begin{frame}{워크 예제: Image Generation의 30년}
\small
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Cohort} & \textbf{Top Generative Class} & \textbf{대표 방법론} \\
\midrule
1987-2006 & 거의 없음 & 텍스처 합성, image quilting \\
2007-2011 & RBM / DBN  & 통계 모델 기반 image prior \\
2012-2016 & GAN 등장   & DCGAN (2015), pix2pix (2016) \\
2017-2021 & GAN 정점 + Diffusion 등장 & StyleGAN (2018-19), DDPM (2020), CLIP (2021) \\
2022-2025 & Diffusion 압승 & \hot{Stable Diffusion, Imagen, DALL-E 3, Sora} \\
\bottomrule
\end{tabular}
\vfill
\muted{한 Phylum 안에서 Class 단위로 paradigm shift가 4번 — phylogeny가 그 가지치기를 추적한다.}
\end{frame}

\begin{frame}{워크 예제: 3D Vision의 30년}
\small
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Cohort} & \textbf{State of the art} \\
\midrule
1987-1996 & Stereo, Shape from X (shading/texture/contour) \\
1997-2006 & SIFT-based MVS, structure from motion 정형화 \\
2007-2011 & PMVS, real-time SfM (Bundler) \\
2012-2016 & 학습 기반 single-view depth (Eigen 2014), MVSNet (2018) \\
2017-2021 & \cool{NeRF (2020)} — neural rendering의 시작 \\
2022-2025 & \hot{3D Gaussian Splatting} + neural SDF + diffusion-based 3D \\
\bottomrule
\end{tabular}
\vfill
\muted{3D는 cohort마다 ``representation 갱신'' 패턴 — 단순 → SfM → 학습 → neural field → Gaussian. Phylogenetic tree로 자연스럽게 표현됨.}
\end{frame}

\begin{frame}{Future Work}
\begin{itemize}
  \item \cool{Citation enrichment}: OpenAlex/SemanticScholar API로 citation 채워서 ``인기 vs 영향력'' 분리
  \item \cool{Cohort × Phylum heatmap}: 각 cohort의 phylum signature를 한 행으로
  \item \cool{Class-level alluvial}: 인접 cohort 간 rank 흐름을 Sankey로 시각화
  \item \cool{Author network 결합}: 누가 어떤 paradigm shift를 끌고 왔는가
  \item \cool{AAAI/IJCAI/SIGGRAPH 통합}: 112k $\to$ 250k+ 규모로 확장
\end{itemize}
\end{frame}

\begin{frame}{한 줄 요약}
\Large\centering
\vfill
\cool{CV+ML 38년}은 \\[6pt]
\hot{고전 CV (1987-2011)} $\to$ \hot{AlexNet 혁명 (2012-16)} $\to$ \hot{Transformer + SSL + Diffusion (2017-21)} $\to$ \hot{Foundation Models (2022+)}\\[6pt]
\muted{4번의 분기점으로 정리된다.}
\vfill
\end{frame}

\begin{frame}{Q\&A}
\Huge\centering
?
\vfill
\normalsize
\url{https://gisbi-kim.github.io/cvml-paper-phylogeny/}
\end{frame}

\end{document}
"""


# --------------------------------------------------------------------
def main():
    papers = load()
    stats = compute_cohort_stats(papers)

    chunks = [PREAMBLE.lstrip()]
    chunks.append(DISTRIBUTION_FRAME)
    chunks.append(make_cohort_overview_frame(stats))

    chunks.append(r'\section{Cohort Wall — 5년 단위로 본 paradigm}' + '\n')
    for lo, hi in COHORTS:
        chunks.append(make_cohort_frame(stats, f'{lo}-{hi}'))

    chunks.append(make_paradigm_shifts_frame())
    chunks.append(make_validation_frames())

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text('\n'.join(chunks), encoding='utf-8')
    print(f'Wrote {OUT_TEX.relative_to(ROOT)} ({sum(len(c) for c in chunks):,} chars)')


if __name__ == '__main__':
    main()
