import os, time, json, re, requests
import pandas as pd
from bs4 import BeautifulSoup
from ddgs import DDGS
from ollama import Client
from dotenv import load_dotenv


# ========== LLM 질의 함수 ==========
def query_llm(
    messages,
    model_name: str,
    host: str,
    temperature: float = 0.3,
):
    client = Client(host=host)
    res = client.chat(
        model=model_name,
        messages=messages,
        options={"temperature": temperature},
    )
    return res["message"]["content"] if isinstance(res, dict) else res.message.content


# ========== DuckDuckGo 검색 함수 ==========
def duckduckgo_search(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(f"{r.get('title','')} - {r.get('body','')}")
    return "\n".join(results)

# ========== 홈페이지 요약 함수 ==========
def summarize_homepage(url):
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:2000] 
    except Exception as e:
        return f"홈페이지 접근 실패: {e}"

# ========== 대중적 이름 및 요약 추출 함수 ==========
def analyze_company_row(row, model_name, host):
    company   = row['회사명']
    code      = row['종목코드']
    industry  = row['업종']
    product   = row['주요제품']
    president = row['대표자명']
    homepage  = row['홈페이지']
    country   = row['지역']

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
        {"role": "user", "content": prompt}
    ]

    try:
        answer = query_llm(messages, model_name=model_name, host=host)

        name_guess       = re.search(r"대중적이름\(추정\):(.+)", answer)
        homepage_summary = re.search(r"홈페이지정보\(요약\):(.+)", answer)
        verify_result    = re.search(r"검증결과\(예/아니오\):(.+)", answer)
        verify_reason    = re.search(r"검증근거:(.+)", answer)

        guessed_name  = name_guess.group(1).strip() if name_guess else company
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

            
            "LLM_검증결과": verdict,   # '예' / '아니오' / '판단실패'
            "LLM_검증근거": reason,
        }

    except Exception as e:
        print(f"[ERROR] analyze_company_row 실패: 회사명={company}, 예외={repr(e)}")


    except Exception as e:
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

# ========== 전체 실행 ==========
def main_get_company_info(model_name = "gemma3:12b"):

    # ========== 환경 설정 ==========
    load_dotenv()
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    INPUT_CSV = "../datas/자기자본(DART).csv"
    OUTPUT_CSV = "../datas/company_info.csv"

    df = pd.read_csv(INPUT_CSV)
    required_cols = ['회사명', '종목코드', '경실련업종', '업종', '주요제품', '대표자명', '홈페이지', '지역']
    df = df[[col for col in required_cols if col in df.columns]].dropna(subset=['회사명'])

    results = []
    for _, row in df.iterrows():
        info = analyze_company_row(row, model_name, host)
        results.append(info)
        print(f"완료: {info['회사명']} → {info['대중적이름(추정)']}")

    out_df = pd.DataFrame(results)[[
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
    ]]
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n저장 완료: {OUTPUT_CSV}")

if __name__ == "__main__":
    main_get_company_info()