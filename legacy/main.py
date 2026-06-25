# step2_diff_shareholders.py

import os
import re
from glob import glob
from typing import Callable, Optional
from utils.get_company_info_main import get_company_info
from utils.get_shareholder_relationship_tables import main_get_shareholder_current_tables
import pandas as pd
import shutil
from pathlib import Path
import ollama

# ==============================
# 0. 경로 / 상수 설정
# ==============================
# ✅ 필요에 따라 아래 경로만 프로젝트 구조에 맞게 조정해서 사용하세요.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
PAST_DIR        = "./datas/shareholder_relationship_tables"    # 과거: 친족여부 라벨 있는 테이블
CURRENT_DIR     = "./datas/shareholder_relationship_tables_current"        # 현재: 새로 뽑은 (이번에 DART에서 가져온) 테이블
SUMMARY_CSV     = "./datas/자기자본(DART).csv"                     # 회사명 리스트 기준
LLM_RESULT_DIR  = "./datas/llm_diff_results"                       # LLM 결과 저장 디렉토리
COMPANY_INFO_CSV = "./datas/company_info.csv"
OUTPUT_DIR_PATH = "./datas/shareholder_relationship_tables_current"        # 현재: 새로 뽑은 (이번에 DART에서 가져온) 테이블


# ==============================
# 1. 공통 유틸
# ==============================

def normalize_name_for_match(raw) -> str:
    """
    이름 비교를 위한 간단한 정규화 함수.
    - NaN / None → 빈 문자열
    - 앞뒤 공백 제거
    - 한글/영문/숫자 외 문자, 공백 제거
    예) '  홍  길 동 ' → '홍길동'
    """
    if raw is None:
        return ""

    # pandas NaN 대응
    try:
        import pandas as pd
        if pd.isna(raw):
            return ""
    except Exception:
        pass

    s = str(raw).strip()
    # 공백, 특수문자 제거 (한글/영문/숫자만 남기기)
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s

def refresh_current_tables(current_dir: str = CURRENT_DIR, clear_first: bool = True):
    """
    1단계: DART에서 '현재' 주주 테이블을 다시 추출하는 헬퍼.

    - clear_first=True 이면 current_dir 내 기존 CSV 파일을 먼저 삭제
    - 그 다음 main_get_shareholder_current_tables()을 호출해서
      새 table_002 CSV들을 current_dir 아래에 다시 생성
    """
    cur_path = Path(current_dir)
    cur_path.mkdir(parents=True, exist_ok=True)

    if clear_first:
        # remove only .csv files (other 파일은 유지)
        for csv_path in cur_path.glob("*.csv"):
            csv_path.unlink()
        print(f"[REFRESH] cleared existing CSVs in {cur_path}")

    print("[REFRESH] re-fetching current shareholder tables from DART...")
    # ⚠️ 여기 함수 이름/인자는 실제 step1 모듈에 맞춰 주세요.
    main_get_shareholder_current_tables(
        input_data_path=SUMMARY_CSV,  # ./datas/자기자본(DART).csv
        out_dir=current_dir,
    )
    print("[REFRESH] DART tables updated.")


def run_full_pipeline(
    recheck_all: bool = False,
    run_llm_if_changed: bool = True,
):
    """
    전체 파이프라인:
      1) CURRENT_DIR 비우고 1단계 다시 실행 → 최신 테이블 추출
      2) 2단계 diff + LLM 조사 실행
    """
    # 1) 항상 최신 테이블로 갱신
    refresh_current_tables(current_dir=CURRENT_DIR, clear_first=True)

    # 2) diff + LLM 단계
    step2_compare_and_optionally_run_llm(
        summary_csv=SUMMARY_CSV,
        past_dir=PAST_DIR,
        current_dir=CURRENT_DIR,
        run_llm_if_changed=run_llm_if_changed,
        llm_result_dir=LLM_RESULT_DIR,
        llm_handler=llm_handler_with_company_info,
        recheck_all=recheck_all,
    )


def query_llm_ollama(messages, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """
    Simple wrapper around Ollama chat API.

    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    returns: assistant text content (string)
    """
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": float(temperature),
            "num_predict": max_tokens,
        },
    )

    # resp: {"model": "...", "created_at": "...", "message": {"role": "assistant", "content": "..."} , ...}
    msg = resp.get("message", {})
    content = msg.get("content", "")
    return content if isinstance(content, str) else str(content)

def safe_company_name(company: str) -> str:
    """Normalize company name for file paths."""
    return re.sub(r"[^\w가-힣]", "_", str(company).strip())


def normalize_shareholder_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize shareholder table columns.
    - unify variants of name/relationship columns → '성명', '관계'
    - pick one stock column and convert to numeric '주식수_판단'
    - keep only relevant columns
    """
    df = df.copy()

    # normalize raw column names: remove spaces
    df.columns = df.columns.astype(str).str.replace(" ", "").str.strip()

    # --- 1) name column normalization ---
    name_aliases = {
        "성명", "성명성명성명", "성명_성명_성명",
        "성 명", "성명_", "_성명",
    }
    for col in list(df.columns):
        if col in name_aliases:
            df = df.rename(columns={col: "성명"})

    # --- 2) relation column normalization ---
    rel_aliases = {
        "관계", "관계관계관계", "관계_관계_관계",
        "관 계", "관계_", "_관계",
    }
    for col in list(df.columns):
        if col in rel_aliases:
            df = df.rename(columns={col: "관계"})

    # --- 3) detect stock column (contains "주식수") ---
    stock_col = None

    # 1순위: "기말" + "주식수" 둘 다 포함된 컬럼
    candidates = [c for c in df.columns if ("주식수" in c and "기말" in c)]
    if candidates:
        stock_col = candidates[0]
    else:
        # 2순위: "주식수"만 포함된 컬럼 중 첫 번째
        candidates = [c for c in df.columns if "주식수" in c]
        if candidates:
            stock_col = candidates[0]

    if stock_col:
        df["주식수_판단"] = (
            df[stock_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("-", "0", regex=False)
        )
        df["주식수_판단"] = pd.to_numeric(df["주식수_판단"], errors="coerce")

    # --- 4) normalize name text ---
    if "성명" in df.columns:
        df["성명"] = df["성명"].astype(str).str.strip()

    # --- 5) keep only useful columns if present ---
    keep_cols = []
    for col in ["성명", "관계", "주식수_판단", "친족여부"]:
        if col in df.columns:
            keep_cols.append(col)

    if not keep_cols:
        # nothing useful, just return empty frame with no columns
        return pd.DataFrame()

    return df[keep_cols].copy()


def find_past_table_path(company: str, past_dir: str = PAST_DIR) -> Optional[str]:
    """
    Return past table path for given company, or None if not found.

    Expected filename: "<회사명>_주주관계.csv"
    """
    safe_name = safe_company_name(company)
    path = os.path.join(past_dir, f"{safe_name}_주주관계.csv")
    return path if os.path.exists(path) else None


def find_current_table_path(company: str, current_dir: str = CURRENT_DIR) -> Optional[str]:
    """
    Return current table path for given company, or None if not found.

    Expected filename pattern: "<회사명>_*table_002.csv"
    """
    safe_name = safe_company_name(company)
    pattern = os.path.join(current_dir, f"{safe_name}_*table_002.csv")
    matches = glob(pattern)
    if not matches:
        return None
    # if multiple, just pick first (can refine later if needed)
    return matches[0]


# ==============================
# 2. 과거 vs 현재 diff 계산
# ==============================
def diff_shareholder_tables(df_past: pd.DataFrame, df_current: pd.DataFrame) -> dict:
    """
    Compare past/current shareholder tables.

    Returns:
        {
            "new_names": set,
            "removed_names": set,
            "changed_names": set,
            "unchanged_names": set,
            "past_df": DataFrame,
            "current_df": DataFrame,
        }
    """
    past = normalize_shareholder_df(df_past)
    curr = normalize_shareholder_df(df_current)

    if "성명" not in past.columns or "성명" not in curr.columns:
        # cannot compare without name column
        return {
            "new_names": set(),
            "removed_names": set(),
            "changed_names": set(),
            "unchanged_names": set(),
            "past_df": past,
            "current_df": curr,
        }

    past_names = set(past["성명"].dropna().astype(str))
    curr_names = set(curr["성명"].dropna().astype(str))

    new_names = curr_names - past_names
    removed_names = past_names - curr_names
    common_names = past_names & curr_names

    changed_names = set()
    unchanged_names = set()

    for name in common_names:
        row_p = past[past["성명"] == name].iloc[0]
        row_c = curr[curr["성명"] == name].iloc[0]

        rel_p = row_p.get("관계")
        rel_c = row_c.get("관계")

        stock_p = row_p.get("주식수_판단")
        stock_c = row_c.get("주식수_판단")

        changed = False

        # relationship changed
        if (rel_p != rel_c) and not (pd.isna(rel_p) and pd.isna(rel_c)):
            changed = True

        # stock changed (simple check)
        try:
            if pd.notna(stock_p) and pd.notna(stock_c):
                if float(stock_p) != float(stock_c):
                    changed = True
        except Exception:
            # if cast fails, ignore
            pass

        if changed:
            changed_names.add(name)
        else:
            unchanged_names.add(name)

    return {
        "new_names": new_names,
        "removed_names": removed_names,
        "changed_names": changed_names,
        "unchanged_names": unchanged_names,
        "past_df": past,
        "current_df": curr,
    }


def collect_llm_targets(diff_info: dict, recheck_all: bool = False) -> pd.DataFrame:
    """
    From diff result, select rows that should be sent to LLM.

    recheck_all:
      - True  → 현재 테이블에 있는 모든 인물을 LLM 대상으로 다시 조사
      - False → new_names ∪ changed_names 중 과거에 이미 '친족'이었던 인물은 제외
    """
    past = diff_info["past_df"]
    curr = diff_info["current_df"]

    if "성명" not in curr.columns:
        return pd.DataFrame()

    # ✅ recheck_all=True 이면 현재 테이블 전체를 LLM 대상으로 사용
    if recheck_all:
        keep_cols = [c for c in ["성명", "관계", "주식수_판단"] if c in curr.columns]
        if not keep_cols:
            return pd.DataFrame()
        return curr[keep_cols].copy()

    # ✅ recheck_all=False (기존 동작: 변경/신규 + 과거 '친족' 제외)
    new_names = diff_info["new_names"]
    changed_names = diff_info["changed_names"]

    # confirmed family in past table
    if "친족여부" in past.columns and "성명" in past.columns:
        confirmed_family = set(
            past[past["친족여부"] == "친족"]["성명"]
            .dropna()
            .astype(str)
        )
    else:
        confirmed_family = set()

    target_names = (new_names | changed_names) - confirmed_family
    if not target_names:
        return pd.DataFrame(columns=curr.columns)

    df_targets = curr[curr["성명"].isin(target_names)].copy()
    keep_cols = [c for c in ["성명", "관계", "주식수_판단"] if c in df_targets.columns]
    return df_targets[keep_cols]

# ==============================
# 4. Step2 메인
# ==============================
def step2_compare_and_optionally_run_llm(
    summary_csv: str = SUMMARY_CSV,
    past_dir: str = PAST_DIR,
    current_dir: str = CURRENT_DIR,
    run_llm_if_changed: bool = True,
    llm_result_dir: str = LLM_RESULT_DIR,
    llm_handler: Optional[Callable[[str, pd.DataFrame], pd.DataFrame]] = None,
    recheck_all: bool = False,   # ✅ 여기 추가
):
    """
    Step 2 main loop.

    For each company in summary_csv:
      1) Load past / current shareholder tables.
      2) Compute diff (new / changed / removed).
      3) If there is any diff and run_llm_if_changed is True and llm_handler is given:
           - select LLM targets (new or changed, excluding already '친족')
           - run llm_handler(company, df_targets)
           - save result to CSV
    """
    os.makedirs(llm_result_dir, exist_ok=True)

    df_summary = pd.read_csv(summary_csv)

    if "회사명" not in df_summary.columns:
        raise KeyError(f"'회사명' column not found in {summary_csv}")

    company_list = df_summary["회사명"].dropna().astype(str).tolist()

    for company in company_list:
        print(f"\n[STEP2] 회사 처리: {company}")

        past_path = find_past_table_path(company, past_dir=past_dir)
        curr_path = find_current_table_path(company, current_dir=current_dir)

        if not past_path:
            print("  ⚠️ 과거 테이블 없음, skip")
            continue
        if not curr_path:
            print("  ⚠️ 현재 테이블 없음, skip")
            continue

        try:
            df_past = pd.read_csv(past_path)
            df_curr = pd.read_csv(curr_path)
        except Exception as e:
            print(f"  ❌ 테이블 로드 실패: {e}")
            continue

        diff_info = diff_shareholder_tables(df_past, df_curr)

        new_cnt     = len(diff_info["new_names"])
        changed_cnt = len(diff_info["changed_names"])
        removed_cnt = len(diff_info["removed_names"])

        has_diff = bool(new_cnt or changed_cnt or removed_cnt)

        print(f"  ▷ new: {new_cnt}, changed: {changed_cnt}, removed: {removed_cnt}")

        # ✅ recheck_all=False 일 때만 diff 없으면 pass
        if not has_diff and not recheck_all:
            print("  ✅ 과거/현재 테이블 차이 없음 → pass (recheck_all=False)")
            continue
        elif not has_diff and recheck_all:
            print("  🔁 과거/현재 테이블 차이는 없지만, recheck_all=True 이므로 전체 재조사 진행")

        if not run_llm_if_changed:
            print("  ⚪ 옵션(run_llm_if_changed=False)에 의해 LLM 호출 생략")
            continue

        if llm_handler is None:
            print("  ⚠️ llm_handler 가 설정되지 않아 LLM 호출을 건너뜁니다.")
            continue

        # select LLM targets
        df_targets = collect_llm_targets(diff_info, recheck_all=recheck_all)
        if df_targets.empty:
            print("  ⚠️ LLM 대상으로 조사할 신규/변경 인물이 없음")
            continue

        print(f"  🔍 LLM 조사 대상 인원 수: {len(df_targets)}명")

        # call user-provided LLM handler
        try:
            df_llm = llm_handler(company, df_targets)
        except Exception as e:
            print(f"  ❌ llm_handler 호출 중 오류: {e}")
            continue

        if df_llm is None or df_llm.empty:
            print("  ⚠️ llm_handler 결과가 비어 있음")
            continue

        # save result
        out_path = os.path.join(llm_result_dir, f"{safe_company_name(company)}_llm_diff.csv")
        df_llm.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  📄 LLM diff 결과 저장: {out_path}")


def load_company_info_df(csv_path: str = COMPANY_INFO_CSV) -> pd.DataFrame:
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


# ==============================
# 5. 예시: 간단 더미 LLM 핸들러
# ==============================
def llm_handler_with_company_info(company: str, df_targets: pd.DataFrame) -> pd.DataFrame:
    """
    LLM handler that:
      1) Uses company_info.csv for background info (대중적이름, 주요제품, 검색요약, 홈페이지요약, 대표자명 등).
      2) If target name == 대표자명(본인) → LLM 호출 없이 '본인 / 확실'로 처리.
      3) Otherwise, call LLM once per person to judge:
           - 관계유형: <배우자 / 자녀 / 부모 / 기타가족 / 없음>
           - 가족여부: <확실 / 가족으로 예상됨 / 불확실>
           - 근거요약: ...
    Returns DataFrame with columns:
      ["회사명", "성명", "관계", "주식수_판단",
       "LLM_관계유형", "LLM_가족여부", "LLM_근거"]
    """
    df_targets = df_targets.copy()
    df_targets["회사명"] = company

    # 1) 회사 기본 정보 로드
    ci = get_company_info(company)
    ci_company_name   = ci.get("회사명", company)
    ci_code           = ci.get("종목코드", "")
    ci_industry       = ci.get("업종", "")
    ci_product        = ci.get("주요제품", "")
    ci_ceo            = ci.get("대표자명", "")  # 여기서 사실상 '회장/대표' 역할 (추측입니다)
    ci_homepage       = ci.get("홈페이지", "")
    ci_region         = ci.get("지역", "")
    ci_popular_name   = ci.get("대중적이름(추정)", "")
    ci_search_summary = ci.get("검색결과요약", "")
    ci_home_summary   = ci.get("홈페이지정보(요약)", "")

    norm_ceo = normalize_name_for_match(ci_ceo)

    results = []

    for _, row in df_targets.iterrows():
        name   = str(row.get("성명", "")).strip()
        rel    = str(row.get("관계", "")).strip() if "관계" in row else ""
        shares = row.get("주식수_판단", None)

        norm_name = normalize_name_for_match(name)

        # 2) 대표자명 == 인물명 → 본인으로 간주, LLM 스킵
        if norm_ceo and norm_name and norm_ceo == norm_name:
            results.append(
                {
                    "회사명": company,
                    "성명": name,
                    "관계": rel,
                    "주식수_판단": shares,
                    "LLM_관계유형": "본인",
                    "LLM_가족여부": "확실",
                    "LLM_근거": "회사 대표자(본인)와 이름이 일치하여 별도 조사 없이 본인으로 간주함.",
                }
            )
            continue

        # 3) LLM 프롬프트 구성
        prompt = f"""
[회사 기본 정보]
회사명: {ci_company_name}
종목코드: {ci_code}
업종: {ci_industry}
주요제품: {ci_product}
대표자명(회장/CEO): {ci_ceo}
홈페이지: {ci_homepage}
기업 위치 지역: {ci_region}
대중적이름(추정): {ci_popular_name}

[회사에 대한 검색/홈페이지 요약]
검색결과 요약: {ci_search_summary}
홈페이지 요약: {ci_home_summary}

[조사 대상 주주 정보]
주주 이름: {name}
주주 관계(표에 기재된 내용): {rel}
보유 주식수(추정): {shares}

위 정보를 바탕으로, 이 인물이 회사의 대표자(또는 오너 일가)와 어떤 가족 관계인지 판단해 주세요.

판단 기준:
- '관계유형'은 최소한 다음 중 하나로 선택:
    배우자 / 자녀 / 부모 / 기타가족 / 없음
- '가족여부'는:
    확실 / 가족으로 예상됨 / 불확실
  중에서 선택.

출력 형식은 반드시 아래와 같이 해주세요 (다른 말은 쓰지 마세요):

관계유형: <배우자 / 자녀 / 부모 / 기타가족 / 없음>
가족여부: <확실 / 가족으로 예상됨 / 불확실>
근거요약: <한두 줄로 핵심만 설명>
"""

        messages = [
            {
                "role": "system",
                "content": "당신은 기업 지배구조와 친족관계를 분석하는 전문가입니다. 지분 구조와 대표자 정보를 기준으로 가족 여부를 신중히 판단하세요.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            answer = query_llm_ollama(messages, temperature=0.2)
        except Exception as e:
            # LLM 호출 실패 시 fallback
            results.append(
                {
                    "회사명": company,
                    "성명": name,
                    "관계": rel,
                    "주식수_판단": shares,
                    "LLM_관계유형": "없음",
                    "LLM_가족여부": "불확실",
                    "LLM_근거": f"LLM 호출 실패: {e}",
                }
            )
            continue

        # 4) LLM 응답 파싱
        relation_type = "없음"
        family_status = "불확실"
        reason        = "판단 근거 부족"

        for line in (answer or "").splitlines():
            t = line.strip()
            if t.startswith("관계유형:"):
                relation_type = t.replace("관계유형:", "").strip()
            elif t.startswith("가족여부:"):
                family_status = t.replace("가족여부:", "").strip()
            elif t.startswith("근거요약:"):
                reason = t.replace("근거요약:", "").strip()

        results.append(
            {
                "회사명": company,
                "성명": name,
                "관계": rel,
                "주식수_판단": shares,
                "LLM_관계유형": relation_type,
                "LLM_가족여부": family_status,
                "LLM_근거": reason,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":

    run_full_pipeline(
        recheck_all=True,        # True면 전체 재조사, False면 변경/신규만
        run_llm_if_changed=True,
    )

    step2_compare_and_optionally_run_llm(
        summary_csv=SUMMARY_CSV,
        past_dir=PAST_DIR,
        current_dir=CURRENT_DIR,
        run_llm_if_changed=True,
        llm_result_dir=LLM_RESULT_DIR,
        llm_handler=llm_handler_with_company_info,
        recheck_all=True,   # ✅ 현재 테이블의 모든 인물을 다시 LLM에 태움
    )