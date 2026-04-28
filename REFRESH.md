# Refresh Guide — 데이터/택소노미 갱신 워크플로우

이 저장소는 **2025-04 시점 스냅샷**입니다. 시간이 지나 새 논문이 쌓이면 (혹은 분류 규칙을 다듬으면) 아래 절차를 따라 사이트 + 슬라이드 + 통계를 한 번에 새로 생성하세요.

---

## 1. 논문 데이터는 어디에서 오는가

| 항목 | 값 |
|------|-----|
| **원본 출처** | [github.com/gisbi-kim/cvmlpaper-atlas](https://github.com/gisbi-kim/cvmlpaper-atlas) (DBLP에서 자동 수집) |
| **포함 학회** | CVPR · NeurIPS · ICML · ICCV · ICLR · ECCV · 3DV |
| **제외 학회** | AAAI / IJCAI / SIGGRAPH / 워크샵 / 워크샵 페이퍼 / 저널 |
| **시간 범위** | 1987 ~ **2025** (스냅샷 시점) |
| **총 논문 수** | **112,183편** |
| **이 저장소의 dump 파일** | `data/raw/all_dblp.json` (39 MB, gitignore 제외 — 직접 커밋되어 있음) |

**중요**: `all_dblp.json`은 cvmlpaper-atlas 저장소의 한 시점 dump입니다. 학회는 매년 새 논문을 발표하므로 1년 뒤에는 약 35,000편 추가될 것으로 예상.

---

## 2. 갱신 시점 가이드

| 사용 상황 | 추천 갱신 주기 |
|----------|----------|
| 매 학회 시즌 직후 (CVPR 6월, NeurIPS 12월, ICLR 5월) | 분기별 |
| 논문 통계만 일치시키려고 | **연 1회** (1월) — 직전 해 학회 다 들어왔을 때 |
| 분류 규칙 개선 후 | 즉시 (코드 수정 시점에) |
| 신규 패러다임이 등장했을 때 | 새 키워드 추가 + 재분류 (예: 2026년에 새로운 paradigm이 보이면) |

---

## 3. 갱신 절차

### Step 1: 새 데이터 가져오기

cvmlpaper-atlas 저장소에서 최신 `all_dblp.json`을 가져옵니다.

```bash
# 옵션 A: cvmlpaper-atlas를 서브모듈로 두고 pull
git clone https://github.com/gisbi-kim/cvmlpaper-atlas.git /tmp/atlas
cp /tmp/atlas/output/all_dblp.json data/raw/all_dblp.json

# 옵션 B: 직접 다운로드 (releases 페이지에 있다면)
# curl -L https://github.com/gisbi-kim/cvmlpaper-atlas/releases/latest/download/all_dblp.json \
#   -o data/raw/all_dblp.json
```

새 논문이 정상적으로 들어왔는지 확인:

```bash
python3 -c "
import json
with open('data/raw/all_dblp.json', encoding='utf-8') as f:
    d = json.load(f)
print(f'Total: {len(d)} papers')
print(f'Year range: {min(p[\"year\"] for p in d)} - {max(p[\"year\"] for p in d)}')
"
```

### Step 2: 분류 규칙이 새 논문을 잘 잡는지 확인

```bash
# 미분류율 확인 (기준선: 5.9%)
python3 -c "
import json, sys
sys.path.insert(0, '.')
from classify import classify

with open('data/raw/all_dblp.json', encoding='utf-8') as f:
    papers = json.load(f)

other = sum(1 for p in papers if classify(p['title'])[0] == 'Other')
print(f'Unclassified: {other}/{len(papers)} = {other/len(papers)*100:.1f}%')
"
```

미분류율이 7% 이상이면 새로운 패러다임 키워드가 있다는 신호 — `classify.py`의 keyword cluster를 보강하세요. 어떤 패턴이 빠졌는지 확인:

```bash
python3 -c "
import json, sys, random
sys.path.insert(0, '.')
from classify import classify
with open('data/raw/all_dblp.json', encoding='utf-8') as f:
    papers = json.load(f)
unclass = [p for p in papers if classify(p['title'])[0] == 'Other']
random.seed(0)
for p in random.sample(unclass, 30):
    print(f'  [{p[\"year\"]}] {p[\"title\"][:90]}')
"
```

### Step 3: 전체 파이프라인 실행

```bash
python run_pipeline.py
```

순서대로 실행되는 단계:
1. `parse_atlas.py` → `data/intermediate/papers_parsed.json`
2. `classify.py` → 분류 (미분류율 출력)
3. `make_papers_json.py` → `papers.json` (웹 viewer용, ~40MB)
4. `make_tree_data.py` → `tree_data.json` (D3 hierarchy, ~110KB)
5. `make_excel.py` → `cvml_taxonomy.xlsx` (선택)

수동으로 단계별 실행해도 됨:
```bash
python parse_atlas.py
python classify.py            # 스모크 테스트 + sample 분류
# 위는 sample만 분류하므로, 실제 전수 분류는 inline:
python -c "
import json, sys
sys.path.insert(0, '.')
from classify import classify
from genus_rules import assign_genus
with open('data/raw/all_dblp.json', encoding='utf-8') as f: papers = json.load(f)
out = []
for p in papers:
    ph, cl, od = classify(p['title'])
    gn = assign_genus(ph, cl, od, p['title'])
    out.append({**p, 'phylum': ph, 'class': cl, 'order': od, 'genus': gn})
with open('data/intermediate/papers_classified.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False)
"
python make_papers_json.py
python make_tree_data.py
```

### Step 4: 슬라이드 재생성

```bash
python slides/build_slides.py
cd slides
docker run --rm -v "$(pwd):/work" -w /work texlive/texlive:latest \
  bash -c "xelatex -interaction=nonstopmode cvml_phylogeny.tex && \
           xelatex -interaction=nonstopmode cvml_phylogeny.tex"
cd ..
```

자세한 빌드 옵션은 [`slides/README.md`](slides/README.md) 참고.

### Step 5: README + index.html의 통계 갱신

다음 파일들의 **숫자**가 업데이트 필요:

| 파일 | 갱신 부분 | 자동/수동 |
|------|----------|-----------|
| `README.md` | "분류 분포" 테이블 (16-Phylum 카운트, %) | **수동** — 아래 스크립트로 새 표 생성 후 붙여넣기 |
| `index.html` | "Phylum 분포" 바 차트 (`<div class="phylum-bar">` 16개) | **수동** — 아래 스크립트로 새 마크업 생성 후 붙여넣기 |
| `index.html` | hero 영역 / 탭 설명의 `112,183` 같은 숫자 | **수동** |
| `slides/cvml_phylogeny.pdf` | 자동 (build_slides.py가 다 처리) | 자동 |

새 README 분포 테이블 + index.html 바 차트 마크업 자동 생성:

```bash
python3 -c "
import json
from collections import Counter
with open('data/intermediate/papers_classified.json', encoding='utf-8') as f:
    papers = json.load(f)
b = Counter()
for p in papers:
    ph, cl = p['phylum'], p['class']
    if ph == 'Other':
        b['Other / Editorial' if cl == 'Editorial' else 'Other / Unclassified'] += 1
    else:
        b[ph] += 1
total = sum(b.values())

# README 테이블
print('=== README.md 분포 테이블 ===')
for n, c in sorted(b.items(), key=lambda x: -x[1]):
    print(f'| {n} | {c:,} | {c/total*100:.1f}% |')
print(f'| **합계** | **{total:,}** | **100.0%** |')

# index.html 바 마크업
COLORS = {
    '1. Object Detection & Localization': '#e53935',
    '2. Segmentation': '#e91e63', '3. 3D Vision & Reconstruction': '#9c27b0',
    '4. Image Recognition & Retrieval': '#3f51b5', '5. Video & Motion Understanding': '#1976d2',
    '6. Generative Models & Synthesis': '#0097a7', '7. Representation Learning': '#388e3c',
    '8. Vision-Language & Multimodal': '#f57c00', '9. Low-level Vision': '#795548',
    '10. Human-centric Vision': '#ff5722', '11. Deep Learning Architecture': '#607d8b',
    '12. Training Strategies': '#5c6bc0', '13. Optimization & Learning Theory': '#7e57c2',
    '14. Reinforcement Learning & Decision Making': '#26a69a', '15. Efficient & Robust ML': '#f44336',
    '16. Application Domains': '#00897b', 'Other / Editorial': '#bcbd22', 'Other / Unclassified': '#7f7f7f',
}
top = max(b.values())
print()
print('=== index.html phylum-bar markup ===')
for n, c in sorted(b.items(), key=lambda x: -x[1]):
    color = COLORS.get(n, '#9e9e9e')
    safe = n.replace('&', '&amp;')
    w = c / top * 100
    print(f'  <div class=\"phylum-bar\"><span class=\"phylum-name\">{safe}</span><div class=\"phylum-bar-track\"><div class=\"phylum-bar-fill\" style=\"width:{w:.1f}%; background:{color};\"></div></div><span class=\"phylum-count\">{c:,} · {c/total*100:.1f}%</span></div>')
"
```

### Step 6: 커밋 + 푸시

```bash
git add data/raw/all_dblp.json papers.json tree_data.json \
        slides/cvml_phylogeny.tex slides/cvml_phylogeny.pdf \
        README.md index.html
git commit -m "Refresh data: <YYYY-MM> snapshot, +N papers"
git push
```

GitHub Pages는 master 브랜치 push 후 1~2분 안에 재배포됩니다.

---

## 4. 분류 규칙(택소노미)을 바꾸는 경우

택소노미가 변경되면 **연쇄적으로 갱신해야 할 파일**들:

| 변경 종류 | 영향받는 파일 |
|---|---|
| 키워드 추가/조정 (예: 새 paradigm 단어 추가) | `classify.py` + 재분류 + Step 5의 통계 갱신 |
| Phylum 추가/제거/이름 변경 | `classify.py`, `genus_rules.py`, `make_tree_data.py` (PHYLUM_COLORS), `TAXONOMY.md`, `README.md`, `index.html` (stat card / bar chart / 설명) |
| Class/Order 구조 변경 | `classify.py`, `genus_rules.py`, `TAXONOMY.md` |
| Genus 규칙 추가 | `genus_rules.py` |

**검증 절차** (Phylum 변경 후 반드시):

```bash
python classify.py    # 20/20 스모크 테스트가 모두 PASS여야 함
```

스모크 테스트가 깨지면 `classify.py` 맨 아래 `test_cases` 리스트도 새 phylum 이름에 맞게 갱신하세요.

---

## 5. 자기 일관성 체크리스트 (push 전 확인)

- [ ] `python classify.py` → 20/20 PASS
- [ ] 미분류율 < 10% (현재 5.9%)
- [ ] `papers.json`, `tree_data.json` 재생성 완료
- [ ] `slides/cvml_phylogeny.pdf` 재컴파일 완료
- [ ] `README.md` 분포 테이블 숫자 일치
- [ ] `index.html` Phylum 바 차트 16개 + Stat 카드 숫자 일치
- [ ] `TAXONOMY.md` 구조가 실제 분류 결과와 일치
- [ ] GitHub Pages가 빌드 후 정상 표시 (1~2분 후)

---

## 6. 추가 작업 (선택)

### Citation 데이터 enrichment

OpenAlex 또는 Semantic Scholar API로 각 논문의 인용 수를 채우면 "인기 vs 영향력" 분리 분석이 가능. 현재는 모든 citation 값이 0.

샘플 스크립트:
```python
# pseudo-code
import requests
for p in papers:
    if p['doi']:
        r = requests.get(f'https://api.openalex.org/works/doi:{p["doi"]}')
        p['citations'] = r.json().get('cited_by_count', 0)
```

대규모 처리 시 OpenAlex의 batch API 사용 권장 + rate limit 주의 (10 req/sec).

### 새 학회 통합

AAAI / IJCAI / SIGGRAPH 통합 시:
1. `parse_atlas.py`에 venue 필터 추가
2. cvmlpaper-atlas의 dump 범위 확장
3. 학회 특유 키워드(SIGGRAPH의 ``rendering'', ``simulation'') 클러스터 보강
4. 이번 가이드 Step 1~6 다시 실행

---

## 7. 문제 해결

| 증상 | 원인 / 해결 |
|------|------------|
| 미분류율이 갑자기 10% 넘어감 | 새 paradigm 키워드 누락 — sample 30개 확인 후 `classify.py`의 클러스터 보강 |
| `papers.json`이 너무 큼 (>50MB) | citation 데이터까지 포함되면 큼 — gzip 압축 또는 외부 호스팅 검토 |
| GitHub Pages 빌드 실패 | iframe 상대 경로 (`../../tree_data.json`) 깨졌는지 확인 |
| 슬라이드 한글 깨짐 | `kotex` 패키지 누락 — `texlive/texlive:latest` (`-small` 아님) 써야 함 |
| `xelatex` 한글 폰트 없음 | Docker 이미지에는 `UnDotum` 있음. 로컬 빌드면 시스템에 한글 폰트 설치 필요 |

---

## 8. 한 줄 요약

> **데이터 갱신 = atlas dump 교체 → `run_pipeline.py` 한 방 → `slides/build_slides.py` 한 방 → README/index.html 숫자 갱신 → 푸시**.
> 분류 규칙 변경 = `classify.py` / `genus_rules.py` 수정 → 위와 동일.
