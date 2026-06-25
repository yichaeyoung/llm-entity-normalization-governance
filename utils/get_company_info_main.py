import os
import time
import json
import re
import requests
import pandas as pd

from bs4 import BeautifulSoup
from ddgs import DDGS
from ollama import Client
from dotenv import load_dotenv
from typing import Optional


# =========================================================
# 0. 경로 상수
# =========================================================
INPUT_CSV  = "../datas/자기자본(DART).csv"
OUTPUT_CSV = "./datas/company_info.csv"


# =========================================================
# 1. LLM / 검색 유틸
# =========================================================
def query_llm(
    messages,
    model_name: str,
    host: str,
    temperature: float = 0.3,
) -> str:
    """
    Simple wrapper for Ollama client chat.
    """
    client = Client(host=host)
    res = client.chat(
        model=model_name,
        messages=messages,
        options={"temperature": temperature},
    )
    # res: dict {"message": {"content": "..."} , ...}
    if isinstance(res, dict):
        return res["message"]["content"]
    return res.message.content


def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Simple DuckDuckGo search (via ddgs) → joined text.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"{r.get('title','')} - {r.get('body','')}")
    return "\n".join(results)


def summarize_homepage(url: str) -> str:
    """
    Fetch homepage and return truncated plain text (for LLM prompt).
    """
    try:
        if not url or not isinstance(url, str):
            return "홈페이지 URL 없음"

        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:2000]
    except Exception as e:
        return f"홈페이지 접근 실패: {e}"


# =========================================================
# 2. 회사 1행에 대한 분석 함수
# =========================================================
def analyze_company_row(row: pd.Series, model_name: str, host: str) -> dict:
    """
    자본(DART).csv의 한 행(row)을 받아:
      - DuckDuckGo 검색
      - 홈페이지 텍스트 요약
      - LLM으로 대중적 이름 + 홈페이지 요약 + 검증결과 추출
    """
    company   = row["회사명"]
    code      = row.get("종목코드", "")
    industry  = row.get("업종", "")
    product   = row.get("주요제품", "")
    president = row.get("대표자명", "")
    homepage  = row.get("홈페이지", "")
    country   = row.get("지역", "")

    query = f"{company} {industry} {product} 어떤 회사인가요?"
    search_results = duckduckgo_search(query)

    homepage_text = summarize_homepage(homepage)

    prompt = f"""
아래는 회사의 기본 정보 및 검색 결과, 홈페이지에서 추출한 텍스트입니다. 

1. 이 회사는 사람들 사이에서 어떤 이름(브랜드, 제품명 등)으로 더 잘 알려져 있나요?
2. 홈페이지 내용을 2-3줄로 요약해주세요.
3. 위에서 제시한 '대중적이름(추정)'이 회사 정보(업종, 주요제품, 대표자명, 지역 등)와 검색·홈페이지 내용을 종합했을 때 적절하다고 판단하는지,
   '예' 또는 '아니오'로 답하고, 한 줄 근거를 써주세요.

### 회사 정보
회사명: {company}
종목코드: {code}
업종: {industry}
주요제품: {product}
대표자명 : {president}
홈페이지: {homepage}
기업 위치 지역 : {country}

### 검색 결과 요약
{search_results}

### 홈페이지 내용
{homepage_text}
    
답변은 아래 포맷을 따라주세요:
대중적이름(추정): ...
홈페이지정보(요약): ...
검증결과(예/아니오): ...
검증근거: ...
"""

    messages = [
        {"role": "system", "content": "기업 정보 분석 전문가"},
        {"role": "user", "content": prompt},
    ]

    try:
        answer = query_llm(messages, model_name=model_name, host=host)

        name_guess       = re.search(r"대중적이름\(추정\):(.+)", answer)
        homepage_summary = re.search(r"홈페이지정보\(요약\):(.+)", answer)
        verify_result    = re.search(r"검증결과\(예/아니오\):(.+)", answer)
        verify_reason    = re.search(r"검증근거:(.+)", answer)

        guessed_name  = name_guess.group(1).strip() if name_guess else str(company)
        homepage_info = homepage_summary.group(1).strip() if homepage_summary else "홈페이지 요약 실패"

        verdict = verify_result.group(1).strip() if verify_result else "판단실패"
        reason  = verify_reason.group(1).strip() if verify_reason else "검증근거 없음"

        return {
            "회사명": company,
            "종목코드": code,
            "업종": industry,
            "주요제품": product,
            "대표자명": president,
            "홈페이지": homepage,
            "지역": country,
            "대중적이름(추정)": guessed_name,
            "검색결과요약": search_results[:500],
            "홈페이지정보(요약)": homepage_info,
            "LLM_검증결과": verdict,    # '예' / '아니오' / '판단실패'
            "LLM_검증근거": reason,
        }

    except Exception as e:
        # 🔴 여기서 바로 fallback dict를 반환 (이중 except 제거 + None 방지)
        print(f"[ERROR] analyze_company_row 실패: 회사명={company}, 예외={repr(e)}")
        return {
            "회사명": company,
            "종목코드": code,
            "업종": industry,
            "주요제품": product,
            "대표자명": president,
            "홈페이지": homepage,
            "지역": country,
            "대중적이름(추정)": "오류",
            "검색결과요약": f"오류: {e}",
            "홈페이지정보(요약)": "오류",
            "LLM_검증결과": "오류",
            "LLM_검증근거": "LLM 호출 중 예외 발생",
        }


# =========================================================
# 3. 전체 실행: company_info.csv 생성
# =========================================================
def main_get_company_info(model_name: str = "gemma3:12b"):
    load_dotenv()
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    df = pd.read_csv(INPUT_CSV)

    required_cols = ["회사명", "종목코드", "경실련업종", "업종", "주요제품", "대표자명", "홈페이지", "지역"]
    df = df[[col for col in required_cols if col in df.columns]].dropna(subset=["회사명"])

    results = []
    for _, row in df.iterrows():
        info = analyze_company_row(row, model_name, host)
        # info는 항상 dict로 반환되도록 보장됨
        results.append(info)
        print(f"완료: {info['회사명']} → {info['대중적이름(추정)']}")

    out_df = pd.DataFrame(results)[
        [
            "회사명",
            "종목코드",
            "업종",
            "주요제품",
            "대표자명",
            "홈페이지",
            "지역",
            "대중적이름(추정)",
            "검색결과요약",
            "홈페이지정보(요약)",
            "LLM_검증결과",
            "LLM_검증근거",
        ]
    ]
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_CSV}")


# =========================================================
# 4. Step2에서 사용할: company_info 로더
# =========================================================
_company_info_df_cache: Optional[pd.DataFrame] = None


def load_company_info_df(csv_path: str = OUTPUT_CSV) -> pd.DataFrame:
    """
    Lazy-load company_info.csv into a global cache.
    """
    global _company_info_df_cache

    if _company_info_df_cache is not None:
        return _company_info_df_cache

    if not os.path.exists(csv_path):
        print(f"[WARN] company info CSV not found: {csv_path}")
        _company_info_df_cache = pd.DataFrame()
        return _company_info_df_cache

    df = pd.read_csv(csv_path)
    _company_info_df_cache = df
    return _company_info_df_cache


def get_company_info(company: str, csv_path: str = OUTPUT_CSV) -> dict:
    """
    Return a dict of company info for given 회사명.
    If not found, return {}.
    """
    df = load_company_info_df(csv_path=csv_path)

    if df.empty or "회사명" not in df.columns:
        return {}

    target = str(company).strip()
    rows = df[df["회사명"].astype(str).str.strip() == target]
    if rows.empty:
        print(f"[INFO] 회사 기본 정보 미발견: {company}")
        return {}

    return rows.iloc[0].to_dict()


if __name__ == "__main__":
    # 1) company_info.csv 생성 용도
    main_get_company_info()
    # 2) (테스트) 특정 회사 정보 불러오기
    # info = get_company_info("삼성전자")
    # print(info)