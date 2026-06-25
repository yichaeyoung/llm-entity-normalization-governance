<div align="center">

# LLM-Based Corporate Governance & Family Shareholding Analysis

### DART 공시, 웹 검색 기반 RAG, LLM 추론을 결합한 기업 지배구조·친족 지분 자동 분석 시스템

`DART` · `Hybrid RAG` · `Ollama` · `Entity Classification` · `Family Graph`

</div>

---

## 1. 프로젝트 개요

본 프로젝트는 DART 공시에서 수집한 주주 관계 정보를 기반으로 기업의 기초 정보와 주주 간 친족 관계를 조사하고, 과거·현재 주주 명부의 변화를 비교하여 친족 지분과 지배구조를 자동 분석하는 파이프라인입니다.

단순히 `친족` 여부만 확인하는 데 그치지 않고, 웹 검색 기반 RAG와 LLM을 활용하여 배우자, 자녀, 부모, 형제자매 등 구체적인 관계를 추론합니다. 이후 논리 제약 검증을 거쳐 구조화된 CSV와 한국형 기업 가계도 이미지를 생성합니다.

> **주의:** LLM이 생성한 친족 관계는 자동 추론 결과이므로 사실 확정 자료가 아닙니다. 최종 분석에는 DART 공시, 기업 공식 홈페이지, 언론 보도 등 원문 근거를 통한 사람의 검증이 필요합니다.

---

## 2. 주요 기능

| 구분 | 기능 | 설명 |
|---|---|---|
| 1 | DART 기본 정보 로드 | `기존 자기자본(DART).xlsx` 및 기존 주주 관계 테이블을 불러옵니다. |
| 2 | 회사 정보 조사 | 기업명, 종목코드, 홈페이지, 대표자·회장 정보 등을 수집하고 `company_info.csv`로 통합합니다. |
| 3 | 주주 관계 조사 | 주주를 개인과 법인으로 구분하고, 친족 후보의 구체적인 관계를 RAG와 LLM으로 추론합니다. |
| 4 | 과거·현재 명부 비교 | 기존 주주 명부와 실행 시점의 최신 명부를 비교하여 신규 인물, 보유 주식 수, 관계 변화를 탐지합니다. |
| 5 | 친족 지분 계산 | 친족으로 분류된 주주의 보유 주식 수를 집계하여 가족 지분을 계산합니다. |
| 6 | 논리 제약 검증 | 동일 기준 인물에게 아버지 또는 어머니가 복수로 배정되는 등 비정상 관계를 재검증합니다. |
| 7 | 가계도 시각화 | 세대별 계층 구조와 직각 연결선을 적용한 기업 지배구조 가계도 PNG를 생성합니다. |
| 8 | 평가 지표 저장 | 기존 친족 라벨과 LLM 결과를 비교하여 평가 지표를 CSV로 저장합니다. |

---

## 3. 전체 처리 흐름

```text
DART 입력 데이터
    ↓
과거 주주 관계 테이블 생성 및 저장
    ↓
회사 기본 정보 수집
    ↓
실행 시점의 최신 주주 관계 테이블 생성
    ↓
과거·현재 주주 명부 비교
    ↓
변경 대상 또는 전체 대상 선별
    ↓
웹 검색 RAG + LLM 친족 관계 추론
    ↓
논리 제약 검증 및 재검증
    ↓
친족 지분 계산
    ↓
구조화 CSV · 평가 지표 · 가족관계도 PNG 저장
```

---

## 4. 세부 파이프라인

### 4.1 회사 정보 수집 파이프라인

`utils/get_company_info.py`는 입력 CSV의 각 기업을 순회하면서 웹 검색 결과와 기업 홈페이지 요약을 수집합니다. 수집된 정보를 LLM 프롬프트에 함께 제공하고, 기업의 대중적 명칭, 요약 정보, 검증 결과를 구조화합니다.

<p align="center">
  <img src="docs/images/company_info_pipeline.png" width="100%" alt="Company information extraction pipeline" />
</p>

<p align="center"><sub>Figure 1. Company information extraction pipeline</sub></p>

주요 처리 단계는 다음과 같습니다.

1. 환경 변수와 입력 CSV를 로드합니다.
2. 기업명, 종목코드, 홈페이지 등 기본 필드를 추출합니다.
3. DuckDuckGo 검색 결과와 기업 홈페이지 요약을 병렬로 수집합니다.
4. 수집 정보를 이용해 LLM 프롬프트를 구성합니다.
5. LLM 응답을 파싱·검증하고, 실패 시 오류 정보를 포함한 대체 결과를 생성합니다.
6. 전체 기업 결과를 `company_info.csv`에 저장합니다.

---

### 4.2 친족 세부 관계 추론 파이프라인

`utils/search_family_relationship.py`는 과거 주주 명부에서 `친족여부`가 `친족`으로 표시된 인물만 선별하여 세부 관계를 조사합니다.

기준 인물은 `company_info.csv`를 이용해 **회장 → 대표이사 → 대표자명** 순서로 결정합니다. 주주명이 기준 인물과 같으면 `본인`으로 처리하고, 그 외 인물은 회사명과 인물명을 결합한 웹 검색 결과를 LLM에 제공하여 관계를 추론합니다.

<p align="center">
  <img src="docs/images/family_detail_pipeline.png" width="76%" alt="Specific family relationship inference pipeline" />
</p>

<p align="center"><sub>Figure 2. Specific family relationship inference for shareholders labeled as relatives</sub></p>

대표 출력 정보는 다음과 같습니다.

- 대상 인물명
- 기준 인물명
- 구체적 가족 관계
- 관계 추론 근거
- 참고한 검색 요약
- 분석 상태 및 오류 정보

---

### 4.3 주주 명부 비교·RAG 분석·가계도 생성 파이프라인

`main.py`는 기존 주주 관계 테이블과 실행 시점에 생성된 최신 테이블을 비교합니다. 변경된 인물만 분석하거나, `recheck_all` 옵션을 통해 전체 기업과 전체 대상자를 다시 분석할 수 있습니다.

<p align="center">
  <img src="docs/images/full_pipeline.png" width="76%" alt="Shareholder diff, LLM RAG analysis and family graph pipeline" />
</p>

<p align="center"><sub>Figure 3. End-to-end shareholder difference analysis and family graph generation</sub></p>

핵심 로직은 다음과 같습니다.

1. 기업별 과거·현재 주주 명부를 로드합니다.
2. 신규 주주, 삭제된 주주, 주식 수 변화, 관계 변화를 탐지합니다.
3. 변경 대상 또는 전체 재검증 대상을 LLM 분석 목록으로 구성합니다.
4. 대표자와 동일한 인물은 `본인`으로 처리합니다.
5. 나머지 대상은 웹 검색 근거와 기업 정보를 결합하여 LLM이 관계를 추론합니다.
6. 개인·법인 여부, 관계, 상태를 구조화합니다.
7. 가족 관계 제약조건을 검사하고 필요한 경우 재질의합니다.
8. 친족 지분을 집계하고 기업별 가족관계도 PNG를 생성합니다.
9. 기존 친족 라벨과 비교한 평가 지표를 저장합니다.

---

## 5. 프로젝트 구조

```text
.
├── main.py
│
├── datas/
│   ├── shareholder_relationship_tables.py
│   ├── company_info.csv
│   ├── shareholder_relationship_tables/
│   │   └── <company>_주주관계.csv
│   ├── shareholder_relationship_tables_current/
│   │   └── <company>_주주관계.csv
│   ├── family_detail_results/
│   │   └── <company>_family_detail_llm.csv
│   ├── llm_diff_results/
│   │   └── <company>_llm_diff.csv
│   ├── family_graph/
│   │   └── <company>_family_graph.png
│   └── evaluation/
│       └── evaluation_metrics.csv
│
├── utils/
│   ├── get_company_info.py
│   ├── get_shareholder_relationship_tables.py
│   └── search_family_relationship.py
│
├── docs/
│   └── images/
│       ├── company_info_pipeline.png
│       ├── family_detail_pipeline.png
│       └── full_pipeline.png
│
├── requirements.txt
└── README.md
```

> 실제 저장소의 파일명이나 출력 폴더명이 위 구조와 다르다면, 코드에 정의된 경로 상수를 기준으로 README의 경로를 맞춰 주세요.

---

## 6. 입력 데이터

### 6.1 DART 기반 원본 파일

```text
기존 자기자본(DART).xlsx
```

기업별 주주 관계 테이블 생성에 필요한 기본 입력 파일입니다. 실제 코드가 요구하는 시트명과 컬럼명은 원본 파일 형식에 맞게 유지해야 합니다.

### 6.2 과거 주주 관계 테이블

```text
datas/shareholder_relationship_tables/
```

기준 시점의 주주 명부입니다. `datas/shareholder_relationship_tables.py`를 실행하면 코드 내부의 `OUTPUT_DIR_PATH`에 지정된 경로로 저장됩니다.

### 6.3 현재 주주 관계 테이블

```text
datas/shareholder_relationship_tables_current/
```

`main.py` 실행 시점에 DART에서 다시 수집한 최신 주주 명부입니다. 과거 명부와 비교하여 변경된 분석 대상을 찾는 데 사용합니다.

---

## 7. 출력 결과

### 7.1 회사 정보

```text
datas/company_info.csv
```

기업명, 종목코드, 홈페이지, 대표자·회장 정보, 검색 요약, 홈페이지 요약, 검증 상태 등을 통합 저장합니다.

### 7.2 친족 세부 관계 결과

```text
datas/family_detail_results/<company>_family_detail_llm.csv
```

기존 주주 명부에서 친족으로 라벨링된 인물의 구체적인 가족 관계와 추론 근거를 저장합니다.

### 7.3 정밀 관계 분석 결과

```text
datas/llm_diff_results/<company>_llm_diff.csv
```

과거·현재 주주 명부 비교 후 분석 대상으로 선별된 인물에 대해 다음 정보를 구조화합니다.

- 개인 또는 법인 구분
- 기준 인물과의 관계
- 친족 여부 및 분석 상태
- 보유 주식 수와 변동 정보
- LLM 추론 근거
- 웹 검색 요약 및 참고 정보
- 논리 제약 검증 결과

### 7.4 기업 지배구조 가계도

```text
datas/family_graph/<company>_family_graph.png
```

가계도는 다음 기준으로 구성합니다.

- 상단: 부모·창업주·선대 경영진
- 중단: 현재 회장·대표자 및 배우자
- 하단: 자녀·후계자·승계 대상
- 노드 정보: 인물명, 세부 관계, 보유 주식 수
- 연결 방식: 기업 분석 보고서에 적합한 직각형 계층 연결선

### 7.5 평가 지표

```text
datas/evaluation/evaluation_metrics.csv
```

기존 친족 라벨과 LLM 분석 결과를 비교한 기업별·전체 평가 결과를 저장합니다.

---

## 8. 실행 방법

### 8.1 환경 준비

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Ollama가 로컬 또는 원격 서버에서 실행 중이어야 합니다.

```bash
ollama serve
```

환경 변수 예시는 다음과 같습니다.

```env
OLLAMA_HOST=http://localhost:11434
```

사용 모델명, 입력 파일 경로, 출력 폴더 경로는 각 코드의 설정값 또는 환경 변수 구성에 맞게 지정합니다.

### 8.2 과거 주주 관계 테이블 생성

```bash
python datas/shareholder_relationship_tables.py
```

실행 시점의 DART 주주 관계 정보를 다음 경로에 저장합니다.

```text
datas/shareholder_relationship_tables/
```

### 8.3 회사 정보 수집

```bash
python utils/get_company_info.py
```

전체 기업 정보를 조사하여 다음 파일을 생성합니다.

```text
datas/company_info.csv
```

### 8.4 전체 파이프라인 실행

```bash
python main.py
```

전체 파이프라인은 다음 작업을 순차적으로 수행합니다.

1. 최신 주주 관계 테이블 수집
2. 과거·현재 주주 명부 비교
3. LLM 분석 대상 선별
4. RAG 기반 친족 관계 추론
5. 논리 제약 검증
6. 친족 지분 계산
7. 결과 CSV 및 가계도 저장
8. 평가 지표 산출

---

## 9. 분석 모드

### 변경된 대상만 분석

과거·현재 주주 명부의 차이가 있는 인물만 LLM 분석 대상으로 사용합니다. 실행 비용과 시간을 줄이는 데 적합합니다.

### 전체 대상 재검증

```python
run_full_pipeline(recheck_all=True)
```

기존 차이 여부와 관계없이 전체 기업 또는 전체 친족 후보를 다시 분석합니다. 모델, 프롬프트, 검색 전략이 변경된 후 전체 결과를 갱신할 때 사용합니다.

---

## 10. 분석 신뢰성 및 제한사항

1. 웹 검색 결과가 부족하거나 동명이인이 존재하면 잘못된 관계가 추론될 수 있습니다.
2. LLM 응답은 모델과 프롬프트, 검색 시점에 따라 달라질 수 있습니다.
3. `친족`이라는 사전 라벨을 전제로 세부 관계를 강제 선택하면 잘못된 기존 라벨이 그대로 증폭될 수 있습니다.
4. 언론 기사나 검색 요약만으로 가족 관계를 확정해서는 안 됩니다.
5. 친족 관계와 지분율은 공시 기준일에 따라 달라질 수 있으므로 분석 기준일을 결과에 함께 기록해야 합니다.
6. 자동 생성 결과를 외부에 공개하거나 평가 자료로 활용할 때에는 원문 출처, 검증자, 검증 일자를 함께 관리하는 것이 좋습니다.

---

## 11. 연구 활용

본 시스템은 다음 연구 주제에 활용할 수 있습니다.

- LLM 기반 기업 지배구조 분석
- 웹 검색 RAG를 활용한 인물 관계 추론
- 주주 엔터티 정규화와 개인·법인 분류
- 친족 지분 및 경영 승계 구조 분석
- 논리 제약을 적용한 LLM 자기검증
- 기업 관계 그래프 자동 생성

관련 문서:

```text
LLM 기반 주주 구조 분석 시스템 논문 초안 (KCI급).pdf
Su-Kwan Lee Draft
```

---

## 12. 결과 예시

```text
datas/llm_diff_results/삼성전자_llm_diff.csv
datas/family_graph/삼성전자_family_graph.png
```

구조화된 분석 결과와 시각화 결과를 함께 제공하므로, 기업별 지배구조 보고서 작성, 친족 지분 검토, 관계 변화 추적에 활용할 수 있습니다.
