# step_family_detail_from_labeled.py
"""
1) datas/shareholder_relationship_tables/ 에 있는
   '<회사명>_주주관계.csv' 파일을 읽고
2) 친족여부 == '친족' 인 인물만 대상으로
3) company_info.py + get_company_info_main + 웹검색(RAG) + Ollama LLM을 통해
   '회장/대표자 기준 구체적인 가족관계'를 추론하는 파이프라인.

출력:
  datas/family_detail_results/<회사명>_family_detail_llm.csv
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import ollama
from ddgs import DDGS  # DuckDuckGo Search

from get_company_info_main import get_company_info


# ==============================
# 0. 경로 / 상수 설정
# ==============================

OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "gemma3:12b")
PAST_DIR            = "../datas/shareholder_relationship_tables"
SUMMARY_CSV         = "../datas/자기자본(DART).csv"
FAMILY_DETAIL_DIR   = "../datas/family_detail_results"
COMPANY_INFO_CSV    = "../datas/company_info.csv"

_company_info_df_cache: Optional[pd.DataFrame] = None

# 확실한 친족으로 간주할 값들
FAMILY_LABEL_VALUES = ["친족"]


# ==============================
# 1. 공통 유틸
# ==============================

def load_company_info_df(csv_path: str = COMPANY_INFO_CSV) -> pd.DataFrame:
    """
    company_info.csv 를 한 번만 읽어서 캐시해두는 함수.
    기대 컬럼(있으면 사용):
      - 회사명, 종목코드, 업종, 주요제품, 대표자명, 회장, 대표이사,
        홈페이지, 지역, 대중적이름(추정), 검색결과요약, 홈페이지정보(요약)
    """
    global _company_info_df_cache

    if _company_info_df_cache is not None:
        return _company_info_df_cache

    if not os.path.exists(csv_path):
        print(f"[WARN] company_info CSV not found: {csv_path}")
        _company_info_df_cache = pd.DataFrame()
        return _company_info_df_cache

    df = pd.read_csv(csv_path)
    if "회사명" in df.columns:
        df["회사명_norm"] = df["회사명"].astype(str).str.strip()
    _company_info_df_cache = df
    return _company_info_df_cache

def safe_company_name(company: str) -> str:
    """파일명용 회사명 정규화."""
    return re.sub(r"[^\w가-힣]", "_", str(company).strip())


def normalize_name_for_match(raw) -> str:
    """이름 비교용 간단 정규화."""
    if raw is None:
        return ""
    try:
        if pd.isna(raw):
            return ""
    except Exception:
        pass
    s = str(raw).strip()
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s)


def normalize_shareholder_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    주주관계 테이블 컬럼 정규화.
    - '성명', '관계', '주식수_판단', '친족여부' 만 남김.
    """
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace(" ", "").str.strip()

    # 성명 컬럼 통일
    for col in list(df.columns):
        if col in ["성명", "성명성명성명", "성명_성명_성명", "성명_", "_성명"]:
            df = df.rename(columns={col: "성명"})

    # 관계 컬럼 통일
    for col in list(df.columns):
        if col in ["관계", "관계관계관계", "관계_관계_관계", "관계_", "_관계"]:
            df = df.rename(columns={col: "관계"})

    # 주식수_판단이 없으면 '기말 주식수' 기준으로 생성
    if "주식수_판단" not in df.columns:
        stock_col = None
        cand = [c for c in df.columns if ("주식수" in c and "기말" in c)]
        if cand:
            stock_col = cand[0]
        else:
            cand = [c for c in df.columns if "주식수" in c]
            if cand:
                stock_col = cand[0]

        if stock_col:
            df["주식수_판단"] = (
                df[stock_col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("-", "0", regex=False)
            )
            df["주식수_판단"] = pd.to_numeric(df["주식수_판단"], errors="coerce")

    if "성명" in df.columns:
        df["성명"] = df["성명"].astype(str).str.strip()

    keep = []
    for c in ["성명", "관계", "주식수_판단", "친족여부"]:
        if c in df.columns:
            keep.append(c)

    if not keep:
        return pd.DataFrame()

    return df[keep].copy()


# ==============================
# 1-1. 회장/대표자 정보 가져오기
# ==============================

def get_company_leader_info(company: str) -> Tuple[str, Dict]:
    """
    회사별 리더(회장/대표자) 이름을 반환.

    1순위: company_info.csv
    2순위: utils.get_company_info_main.get_company_info(company)

    반환:
      (leader_name, company_info_dict)
    """
    company = str(company).strip()
    ci: Dict = {}
    leader_name = ""

    # ---------- 1) company_info.csv 에서 찾기 ----------
    df_info = load_company_info_df()
    if not df_info.empty and "회사명_norm" in df_info.columns:
        mask = df_info["회사명_norm"] == company
        if mask.any():
            row = df_info.loc[mask].iloc[0]

            # 회장 / 대표이사 / 대표자명 중 하나 선택
            leader_name = str(
                row.get("회장")
                or row.get("대표이사")
                or row.get("대표자명")
                or ""
            ).strip()

            # company_info dict 구성 (있으면 사용하는 필드만)
            for col in [
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
            ]:
                if col in row.index:
                    ci[col] = row.get(col, "")

    # ---------- 2) 그래도 비어 있으면 기존 get_company_info_main 사용 ----------
    ci_main = get_company_info(company)
    # leader_name 비어 있으면 여기서 보충
    if not leader_name:
        leader_name = str(
            ci_main.get("회장")
            or ci_main.get("대표자명")
            or ci_main.get("대표이사")
            or ""
        ).strip()

    # ci_main 에 있는 값도, ci 에 비어 있는 키에만 채워넣기
    for k, v in ci_main.items():
        if k not in ci or not ci[k]:
            ci[k] = v

    if "회사명" not in ci or not ci["회사명"]:
        ci["회사명"] = company

    return leader_name, ci


# ==============================
# 1-2. RAG (DuckDuckGo)
# ==============================

def web_search_snippets(company: str, person_name: str, max_results: int = 5) -> str:
    """
    DuckDuckGo 기반 간단 RAG 컨텍스트 생성.
    회사명 + 인물명으로 검색해서 title/body/url을 붙여 리턴.
    """
    query = f'"{company}" "{person_name}"'
    snippets: List[str] = []

    try:
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                if not title and not body:
                    continue
                snippet = f"[{i+1}] {title}\n{body}\nURL: {href}"
                snippets.append(snippet)
    except Exception as e:
        return f"(검색 실패: {e})"

    if not snippets:
        return "(유의미한 웹 검색 결과를 찾지 못했습니다.)"

    return "\n\n".join(snippets[:max_results])


# ==============================
# 1-3. LLM wrapper
# ==============================

def query_llm_ollama(messages, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Ollama chat 래퍼."""
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": float(temperature),
            "num_predict": max_tokens,
        },
    )
    msg = resp.get("message", {})
    content = msg.get("content", "")
    return content if isinstance(content, str) else str(content)


# ==============================
# 2. 회사별 LLM + RAG 처리
# ==============================

def run_family_detail_for_company(company: str, df_family: pd.DataFrame) -> pd.DataFrame:
    """
    특정 회사에 대해,
    친족여부 == '친족' 인 인물들을 대상으로
    '회장/대표자 기준 구체적인 가족관계'를 LLM+RAG 로 추론.
    """
    df_family = df_family.copy()
    df_family["회사명"] = company

    # 🔹 회사 정보 + 회장/대표자 이름
    leader_name, ci = get_company_leader_info(company)
    ci_company_name   = ci.get("회사명", company)
    ci_code           = ci.get("종목코드", "")
    ci_industry       = ci.get("업종", "")
    ci_product        = ci.get("주요제품", "")
    ci_homepage       = ci.get("홈페이지", "")
    ci_region         = ci.get("지역", "")
    ci_popular_name   = ci.get("대중적이름(추정)", "")
    ci_search_summary = ci.get("검색결과요약", "")
    ci_home_summary   = ci.get("홈페이지정보(요약)", "")

    norm_leader = normalize_name_for_match(leader_name)
    results = []

    for _, row in df_family.iterrows():
        name        = str(row.get("성명", "")).strip()
        base_rel    = str(row.get("관계", "")).strip()
        shares      = row.get("주식수_판단", None)
        family_flag = row.get("친족여부", "")

        if not name:
            continue

        norm_name = normalize_name_for_match(name)

        # 회장/대표자와 이름 동일 → 본인
        if norm_leader and norm_name and norm_leader == norm_name:
            results.append(
                {
                    "회사명": company,
                    "성명": name,
                    "기준인물": leader_name,
                    "원테이블_관계": base_rel,
                    "주식수_판단": shares,
                    "원_친족여부": family_flag,
                    "LLM_entity_type": "사람",
                    "LLM_세부관계": "본인",
                    "LLM_가족여부": "확실",
                    "LLM_근거": "company_info 기준 회장/대표자와 이름이 완전히 동일하여 본인으로 간주.",
                    "LLM_is_person": True,
                }
            )
            continue

        # 🔍 RAG 웹 검색
        rag_snip = web_search_snippets(ci_company_name, name, max_results=5)

        prompt = f"""
[전제]
- 아래 주주 '{name}'은 이미 회사 내부 자료에서 '친족'으로 확정된 인물이다.
- 당신의 임무는 오로지 이 인물이 '{leader_name}'(회장/대표자)과
  어떤 가족관계인지 판단하는 것이다.
- '가족이 아니다'라고 판단하는 것은 허용되지 않는다.
  판단이 어려우면 '판단불가'를 선택하라.

[회사 기본 정보]
회사명: {ci_company_name}
종목코드: {ci_code}
대표자(회장/CEO): {leader_name}
업종: {ci_industry}
주요제품: {ci_product}
홈페이지: {ci_homepage}
지역: {ci_region}
대중적이름(추정): {ci_popular_name}
검색결과요약: {ci_search_summary}
홈페이지정보(요약): {ci_home_summary}

[조사 대상 주주 정보]
주주 이름: {name}
주주 관계(표에 적힌 값): {base_rel}
보유 주식수: {shares}

[웹 검색 스니펫]
{rag_snip}

아래 목록에서 가장 타당한 하나를 선택하라:
- 배우자
- 아들
- 딸
- 아버지
- 어머니
- 형제
- 자매
- 형제자매
- 기타가족
- 판단불가

출력 형식은 정확히 다음과 같이 하라 (다른 말 금지):

세부관계: <배우자 / 아들 / 딸 / 아버지 / 어머니 / 형제 / 자매 / 형제자매 / 기타가족 / 판단불가>
근거요약: <한두 줄로 핵심 설명>
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 한국 상장사의 지배구조와 친족관계를 분석하는 전문가입니다. "
                    "회장/대표자 정보를 기준으로, 해당 주주가 어떤 가족관계에 있는지 신중히 추론하세요. "
                    "정보가 모호하면 '판단불가'를 사용하세요."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            answer = query_llm_ollama(messages, temperature=0.15)
        except Exception as e:
            results.append(
                {
                    "회사명": company,
                    "성명": name,
                    "기준인물": leader_name,
                    "원테이블_관계": base_rel,
                    "주식수_판단": shares,
                    "원_친족여부": family_flag,
                    "LLM_entity_type": "알수없음",
                    "LLM_세부관계": "판단불가",
                    "LLM_가족여부": "불확실",
                    "LLM_근거": f"LLM 호출 실패: {e}",
                    "LLM_is_person": True,
                }
            )
            continue

        detail_rel = "판단불가"
        reason = "판단 근거 부족"

        for line in (answer or "").splitlines():
            t = line.strip()
            if t.startswith("세부관계:"):
                detail_rel = t.replace("세부관계:", "").strip()
            elif t.startswith("근거요약:"):
                reason = t.replace("근거요약:", "").strip()

        results.append(
            {
                "회사명": company,
                "기준인물": leader_name,
                "성명": name,
                "원테이블_관계": base_rel,
                "주식수_판단": shares,
                "원_친족여부": family_flag,
                "LLM_세부관계": detail_rel,
                "LLM_근거": reason,
            }
        )

    return pd.DataFrame(results)


# ==============================
# 3. 메인 루프
# ==============================

def main():
    Path(FAMILY_DETAIL_DIR).mkdir(parents=True, exist_ok=True)

    df_summary = pd.read_csv(SUMMARY_CSV)
    if "회사명" not in df_summary.columns:
        raise KeyError(f"'회사명' column not found in {SUMMARY_CSV}")

    company_list = df_summary["회사명"].dropna().astype(str).tolist()

    for company in company_list:
        print(f"\n[DETAIL] 회사 처리: {company}")

        safe_name = safe_company_name(company)
        path = os.path.join(PAST_DIR, f"{safe_name}_주주관계.csv")
        if not os.path.exists(path):
            print("  ⚠️ 주주관계 테이블 없음, skip")
            continue

        try:
            df_raw = pd.read_csv(path)
        except Exception as e:
            print(f"  ❌ CSV 로드 실패: {e}")
            continue

        df_norm = normalize_shareholder_df(df_raw)
        if df_norm.empty:
            print("  ⚠️ 유효한 컬럼 없음, skip")
            continue

        if "친족여부" not in df_norm.columns:
            print("  ⚠️ '친족여부' 컬럼 없음, skip")
            continue

        df_family = df_norm[df_norm["친족여부"].isin(FAMILY_LABEL_VALUES)].copy()
        if df_family.empty:
            print("  ⚠️ 친족으로 라벨된 인물이 없음, skip")
            continue

        print(f"  🔍 친족 대상 인원 수: {len(df_family)}명")

        df_result = run_family_detail_for_company(company, df_family)
        if df_result is None or df_result.empty:
            print("  ⚠️ LLM 결과 없음")
            continue

        out_path = os.path.join(
            FAMILY_DETAIL_DIR,
            f"{safe_name}_family_detail_llm.csv",
        )
        df_result.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  📄 결과 저장: {out_path}")


if __name__ == "__main__":
    main()