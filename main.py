# step2_diff_shareholders.py

import os
import re
from glob import glob
from typing import Callable, Optional
from pathlib import Path

import pandas as pd
import ollama

# RAG 검색용
from ddgs import DDGS

# 그래프 + 시각화용
import matplotlib
matplotlib.use("Agg")  # headless 환경에서도 PNG 저장 가능하게
import matplotlib.pyplot as plt
import networkx as nx

from utils.get_company_info_main import get_company_info
from utils.get_shareholder_relationship_tables import main_get_shareholder_current_tables

from matplotlib import rcParams
import matplotlib as mpl
from matplotlib import font_manager    
import numpy as np                   


# ==============================
# 0. 경로 / 상수 설정
# ==============================

OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "gemma3:12b")
PAST_DIR           = "./datas/shareholder_relationship_tables"
CURRENT_DIR        = "./datas/shareholder_relationship_tables_current"
SUMMARY_CSV        = "./datas/자기자본(DART).csv"
LLM_RESULT_DIR     = "./datas/llm_diff_results"
COMPANY_INFO_CSV   = "./datas/company_info.csv"
OUTPUT_DIR_PATH    = "./datas/shareholder_relationship_tables_current"

FAMILY_GRAPH_DIR   = "./datas/family_graph"         # 엣지 CSV + PNG
EVAL_METRIC_CSV    = "./datas/llm_eval_metrics.csv" # 회사별 평가 결과

_company_info_df_cache = None  # load_company_info_df용 캐시

# 관계에 따른 y축 레벨 매핑 (위–중간–아래)
# 한 가정에서 1명 이하만 허용할 역할들
UNIQUE_ROLES = ["아버지", "어머니", "배우자"]
RELATION_LEVEL = {
    "아버지": 2,
    "어머니": 2,
    "부모": 2,
    "본인": 1,
    "배우자": 1,
    "형제": 1,
    "자매": 1,
    "형제자매": 1,
    "아들": 0,
    "딸": 0,
    "자녀": 0,
    "기타가족": 0,
    "가족아님": 0,
}

# 실제 존재하는 나눔 폰트 파일 경로로 바꿔주세요
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# 1) 폰트 파일을 fontManager에 등록
font_manager.fontManager.addfont(FONT_PATH)

# 2) Matplotlib이 인식하는 폰트 이름 얻기
font_prop = font_manager.FontProperties(fname=FONT_PATH)
font_name = font_prop.get_name()
print("[DEBUG] Using font:", font_name)

# 3) 전역 폰트 설정
mpl.rcParams["font.family"] = font_name
mpl.rcParams["font.sans-serif"] = [font_name]  # 👉 NetworkX가 쓰는 sans-serif도 덮어쓰기
mpl.rcParams["axes.unicode_minus"] = False

print("[DEBUG] rcParams font.family =", mpl.rcParams["font.family"])

# ✅ 여기에서 리스트 전체 대신 font_name(문자열) 사용
print(
    "[DEBUG] resolved font path   =",
    font_manager.findfont(font_name, fallback_to_default=True),
)


# ==============================
# 1. 공통 유틸
# ==============================

def normalize_name_for_match(raw) -> str:
    """
    Simple normalizer for name comparison.
    - NaN / None → ""
    - strip spaces
    - keep only Korean/English/digits
    """
    if raw is None:
        return ""

    try:
        if pd.isna(raw):
            return ""
    except Exception:
        pass

    s = str(raw).strip()
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    return s


def safe_company_name(company: str) -> str:
    """Normalize company name for file paths."""
    return re.sub(r"[^\w가-힣]", "_", str(company).strip())



def set_korean_font():
    FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    if not os.path.exists(FONT_PATH):
        print("[WARN] NanumGothic 폰트를 찾을 수 없습니다:", FONT_PATH)
        # 못 찾으면 기존 fallback 방식
        for fname in ["NanumGothic", "Malgun Gothic", "AppleGothic"]:
            rcParams["font.family"] = fname
            break
        rcParams["axes.unicode_minus"] = False
        return

    # 1) 폰트 등록
    font_manager.fontManager.addfont(FONT_PATH)

    # 2) 폰트 이름 얻기
    font_prop = font_manager.FontProperties(fname=FONT_PATH)
    font_name = font_prop.get_name()
    print("[DEBUG] Using font:", font_name)

    # 3) 전역 설정
    mpl.rcParams["font.family"] = font_name
    mpl.rcParams["font.sans-serif"] = [font_name]
    mpl.rcParams["axes.unicode_minus"] = False

    print("[DEBUG] rcParams font.family =", mpl.rcParams["font.family"])
    print(
        "[DEBUG] resolved font path   =",
        font_manager.findfont(font_name, fallback_to_default=True),
    )


def build_family_graph_df_for_company(df_llm: pd.DataFrame, ceo_name: str) -> pd.DataFrame:
    """
    Filter LLM result DataFrame for:
      - 사람만 (LLM_is_person == True)
      - 가족이거나 가족으로 예상됨
    그리고 세부관계가 비어 있으면 '기타가족'으로 보정.
    """
    df = df_llm.copy()

    if "LLM_is_person" in df.columns:
        df = df[df["LLM_is_person"] == True]

    # 가족 여부 필터링
    if "LLM_가족여부" in df.columns:
        df = df[df["LLM_가족여부"].isin(["확실", "가족으로 예상됨"])]

    # 세부관계 컬럼 보정
    if "LLM_세부관계" not in df.columns:
        df["LLM_세부관계"] = "기타가족"

    # 본인 노드 추가
    ceo_row = {
        "성명": ceo_name,
        "관계": "본인",
        "LLM_세부관계": "본인",
        "LLM_가족여부": "확실",
    }
    df_ceo = pd.DataFrame([ceo_row])

    # 중복 제거
    df = pd.concat([df_ceo, df], ignore_index=True)
    df = df.drop_duplicates(subset=["성명"], keep="first")

    return df


def compute_family_positions(df: pd.DataFrame, ceo_name: str):
    """
    Assign (x, y) positions by relation level.
    y:
      2: parents
      1: CEO, spouse, siblings
      0: children / 기타가족
    x: each level spread horizontally.
    """
    level_map = {}
    for _, row in df.iterrows():
        name = str(row.get("성명", "")).strip()
        rel  = str(row.get("LLM_세부관계", "")).strip()
        if name == ceo_name:
            level_map[name] = 1
            continue
        level_map[name] = RELATION_LEVEL.get(rel, 0)  # 기본은 0(아래)

    # level -> [names...]
    by_level = {}
    for name, lvl in level_map.items():
        by_level.setdefault(lvl, []).append(name)

    pos = {}
    for lvl, names in by_level.items():
        n = len(names)
        if n == 1:
            xs = [0.0]
        else:
            xs = list(np.linspace(-1.5, 1.5, n))
        y = float(lvl)
        for x, name in zip(xs, names):
            pos[name] = (x, y)

    return pos


def draw_family_graph(company: str, df_llm: pd.DataFrame, ci_ceo: str):
    """
    한국형 조직도 스타일 (직각 배선 + 사각형 노드) 시각화
    * 수정사항: ConnectionStyle을 쓰지 않고, 직접 좌표를 계산하여 선을 그림 (Crash 방지)
    """
    os.makedirs(FAMILY_GRAPH_DIR, exist_ok=True)
    set_korean_font()

    if not ci_ceo:
        return

    df_family = build_family_graph_df_for_company(df_llm, ci_ceo)
    if df_family.empty:
        return

    # 1. 그래프 객체 생성
    G = nx.DiGraph()

    # 2. 노드 데이터 추가
    for _, row in df_family.iterrows():
        name = str(row.get("성명", "")).strip()
        rel  = str(row.get("LLM_세부관계", "")).strip()
        shares = row.get("주식수_판단", 0)
        
        # 라벨: 이름 + 관계
        label_text = f"{name}\n({rel})"
        G.add_node(name, label=label_text, relation=rel, shares=shares)

    # 3. 엣지 데이터 추가
    ceo = ci_ceo
    for _, row in df_family.iterrows():
        name = str(row.get("성명", "")).strip()
        if name == ceo:
            continue
        G.add_edge(ceo, name)

    # 4. 레이아웃 계산
    pos = compute_family_positions(df_family, ceo_name=ceo)
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # --- (A) 엣지 그리기: 직접 좌표 계산 (Manhattan Style) ---
    # 오류 원인인 FancyArrowPatch의 connectionstyle을 버리고, plt.plot으로 직접 그립니다.
    for u, v in G.edges():
        start_x, start_y = pos[u]
        end_x, end_y = pos[v]
        
        # 조직도 스타일: 부모 밑으로 조금 내려왔다가 -> 옆으로 이동 -> 자녀 머리 위로 떨어짐
        # 중간 지점 (Y축) 계산
        mid_y = (start_y + end_y) / 2
        
        # 경로 좌표 생성: (시작) -> (시작_수직내림) -> (끝_수평이동) -> (끝)
        path_x = [start_x, start_x, end_x, end_x]
        path_y = [start_y, mid_y,   mid_y, end_y]
        
        # 선 그리기
        ax.plot(
            path_x, path_y, 
            color="gray", 
            linewidth=1.5, 
            zorder=-1  # 노드 뒤로 숨기기
        )

    # --- (B) 노드 그리기 (사각형 박스) ---
    for node, (x, y) in pos.items():
        rel = G.nodes[node].get("relation", "")
        lbl = G.nodes[node].get("label", "")
        
        # 색상 설정
        if node == ceo or rel == "본인":
            box_color = "#FFE082"  # 금색
            font_weight = 'bold'
        elif rel in ["배우자"]:
            box_color = "#FFCDD2"  # 분홍
            font_weight = 'normal'
        else:
            box_color = "#BBDEFB"  # 파랑
            font_weight = 'normal'

        # 텍스트 박스
        plt.text(
            x, y, 
            lbl,
            size=10, 
            color="black", 
            weight=font_weight,
            ha="center", 
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc=box_color, ec="black", lw=1.5)
        )

    # 5. 마무리 설정
    plt.title(f"{company} 지배구조(가족) 관계도", fontsize=15, pad=20)
    
    if pos:
        x_values, y_values = zip(*pos.values())
        # 여백을 좀 더 넉넉히 (노드 박스가 잘리지 않게)
        plt.xlim(min(x_values) - 1.0, max(x_values) + 1.0)
        plt.ylim(min(y_values) - 0.5, max(y_values) + 0.5)
    
    plt.axis("off")
    plt.tight_layout()

    png_path = os.path.join(FAMILY_GRAPH_DIR, f"{safe_company_name(company)}_family_graph.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  🖼 [한국형] 가족관계도 PNG 저장: {png_path}")

# ==============================
# 1-1. RAG: web search helpers
# ==============================

def web_search_snippets(company: str, person_name: str, max_results: int = 5) -> str:
    """
    Run DuckDuckGo search for (company + person_name) and return
    concatenated text snippets for RAG context.
    """
    query = f'"{company}" "{person_name}"'
    snippets = []

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
# 1-2. LLM wrapper
# ==============================


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

    msg = resp.get("message", {})
    content = msg.get("content", "")
    return content if isinstance(content, str) else str(content)


# ==============================
# 1-3. DART 테이블 갱신
# ==============================

def refresh_current_tables(current_dir: str = CURRENT_DIR, clear_first: bool = True):
    """
    Step1 helper: re-crawl 'current' shareholder tables from DART.

    - clear_first=True → remove existing CSVs in current_dir
    - then call main_get_shareholder_current_tables() to create new table_002 CSVs
    """
    cur_path = Path(current_dir)
    cur_path.mkdir(parents=True, exist_ok=True)

    if clear_first:
        for csv_path in cur_path.glob("*.csv"):
            csv_path.unlink()
        print(f"[REFRESH] cleared existing CSVs in {cur_path}")

    print("[REFRESH] re-fetching current shareholder tables from DART...")
    main_get_shareholder_current_tables(
        input_data_path=SUMMARY_CSV,
        out_dir=current_dir,
    )
    print("[REFRESH] DART tables updated.")


def run_full_pipeline(
    recheck_all: bool = False,
    run_llm_if_changed: bool = True,
):
    """
    Full pipeline:
      1) Refresh CURRENT_DIR from DART
      2) Run diff + LLM + graph + evaluation
    """

    # Check bottom of this code!
    # refresh_current_tables(current_dir=CURRENT_DIR, clear_first=True)

    step2_compare_and_optionally_run_llm(
        summary_csv=SUMMARY_CSV,
        past_dir=PAST_DIR,
        current_dir=CURRENT_DIR,
        run_llm_if_changed=run_llm_if_changed,
        llm_result_dir=LLM_RESULT_DIR,
        llm_handler=llm_handler_with_company_info,
        recheck_all=recheck_all,
    )


# ==============================
# 1-4. 테이블 정규화
# ==============================

def normalize_shareholder_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize shareholder table columns.
    - unify variants of name/relationship columns → '성명', '관계'
    - pick one stock column and convert to numeric '주식수_판단'
    - keep only relevant columns
    """
    df = df.copy()

    df.columns = df.columns.astype(str).str.replace(" ", "").str.strip()

    # 1) name column normalization
    name_aliases = {
        "성명", "성명성명성명", "성명_성명_성명",
        "성명_", "_성명",
    }
    for col in list(df.columns):
        if col in name_aliases:
            df = df.rename(columns={col: "성명"})

    # 2) relation column normalization
    rel_aliases = {
        "관계", "관계관계관계", "관계_관계_관계",
        "관계_", "_관계",
    }
    for col in list(df.columns):
        if col in rel_aliases:
            df = df.rename(columns={col: "관계"})

    # 3) detect stock column ("주식수")
    stock_col = None
    candidates = [c for c in df.columns if ("주식수" in c and "기말" in c)]
    if candidates:
        stock_col = candidates[0]
    else:
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

    # 4) normalize name text
    if "성명" in df.columns:
        df["성명"] = df["성명"].astype(str).str.strip()

    keep_cols = []
    for col in ["성명", "관계", "주식수_판단", "친족여부"]:
        if col in df.columns:
            keep_cols.append(col)

    if not keep_cols:
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
    return matches[0]


# ==============================
# 2. 과거 vs 현재 diff 계산
# ==============================

def diff_shareholder_tables(df_past: pd.DataFrame, df_current: pd.DataFrame) -> dict:
    """
    Compare past/current shareholder tables.
    """
    past = normalize_shareholder_df(df_past)
    curr = normalize_shareholder_df(df_current)

    if "성명" not in past.columns or "성명" not in curr.columns:
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

        if (rel_p != rel_c) and not (pd.isna(rel_p) and pd.isna(rel_c)):
            changed = True

        try:
            if pd.notna(stock_p) and pd.notna(stock_c):
                if float(stock_p) != float(stock_c):
                    changed = True
        except Exception:
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
    Select rows that should be sent to LLM.

    recheck_all:
      - True  → use all rows in current table
      - False → new_names ∪ changed_names, excluding already '친족' in past
    """
    past = diff_info["past_df"]
    curr = diff_info["current_df"]

    if "성명" not in curr.columns:
        return pd.DataFrame()

    if recheck_all:
        keep_cols = [c for c in ["성명", "관계", "주식수_판단"] if c in curr.columns]
        if not keep_cols:
            return pd.DataFrame()
        return curr[keep_cols].copy()

    new_names = diff_info["new_names"]
    changed_names = diff_info["changed_names"]

    if "친족여부" in past.columns and "성명" in past.columns:
        confirmed_family = set(
            past[past["친족여부"] == "친족"]["성명"].dropna().astype(str)
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
# 3. 평가 & 가족 그래프
# ==============================

def binarize_llm_family_flag(family_status: str) -> int:
    """
    Map LLM_가족여부 → binary label (1 = family, 0 = not/uncertain).
    """
    s = (family_status or "").strip()
    if s in ["확실", "가족으로 예상됨"]:
        return 1
    return 0


def evaluate_llm_vs_past(company: str, df_llm: pd.DataFrame, past_df: pd.DataFrame):
    """
    Compare LLM+RAG family prediction with past '친족여부' labels.

    Returns metrics dict or None.
    """

    if df_llm is None or df_llm.empty:
        return None

    # ✅ 사람만 평가에 사용
    df_eval = df_llm.copy()
    if "LLM_is_person" in df_eval.columns:
        df_eval = df_eval[df_eval["LLM_is_person"] == True]
    if df_eval.empty:
        return None

    if df_llm is None or df_llm.empty:
        return None

    p = past_df.copy()
    if "성명" not in p.columns or "친족여부" not in p.columns:
        return None

    p["성명"] = p["성명"].astype(str).str.strip()
    p["gt_family"] = (p["친족여부"] == "친족").astype(int)

    l = df_llm.copy()
    l["성명"] = l["성명"].astype(str).str.strip()
    l["pred_family"] = l["LLM_가족여부"].apply(binarize_llm_family_flag)

    merged = pd.merge(l, p[["성명", "gt_family"]], on="성명", how="left")
    merged = merged.dropna(subset=["gt_family"])
    if merged.empty:
        print(f"  ⚠️ 평가용 GT 라벨이 없음: {company}")
        return None

    merged["gt_family"] = merged["gt_family"].astype(int)

    y_true = merged["gt_family"].values
    y_pred = merged["pred_family"].values

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    metrics = {
        "회사명": company,
        "n_targets": int(len(df_llm)),
        "n_with_label": int(len(merged)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }
    return metrics



# ==============================
# 4. Step2 메인 루프
# ==============================

def step2_compare_and_optionally_run_llm(
    summary_csv: str = SUMMARY_CSV,
    past_dir: str = PAST_DIR,
    current_dir: str = CURRENT_DIR,
    run_llm_if_changed: bool = True,
    llm_result_dir: str = LLM_RESULT_DIR,
    llm_handler: Optional[Callable[[str, pd.DataFrame], pd.DataFrame]] = None,
    recheck_all: bool = False,
):
    """
    Step 2 main loop.

    For each company in summary_csv:
      1) Load past / current shareholder tables.
      2) Compute diff.
      3) If needed, select LLM targets.
      4) Call llm_handler → save LLM result CSV.
      5) Build family graph PNG.
      6) Evaluate vs past '친족여부'.
    """
    os.makedirs(llm_result_dir, exist_ok=True)

    df_summary = pd.read_csv(summary_csv)
    if "회사명" not in df_summary.columns:
        raise KeyError(f"'회사명' column not found in {summary_csv}")

    company_list = df_summary["회사명"].dropna().astype(str).tolist()
    eval_metrics = []

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

        if not has_diff and not recheck_all:
            print("  ✅ 과거/현재 테이블 차이 없음 → pass (recheck_all=False)")
            continue
        elif not has_diff and recheck_all:
            print("  🔁 차이는 없지만 recheck_all=True → 전체 재조사 진행")

        if not run_llm_if_changed:
            print("  ⚪ run_llm_if_changed=False → LLM 호출 생략")
            continue

        if llm_handler is None:
            print("  ⚠️ llm_handler 미설정 → LLM 호출 스킵")
            continue

        df_targets = collect_llm_targets(diff_info, recheck_all=recheck_all)
        if df_targets.empty:
            print("  ⚠️ LLM 대상으로 조사할 인물이 없음")
            continue

        print(f"  🔍 LLM 조사 대상 인원 수: {len(df_targets)}명")

        try:
            df_llm = llm_handler(company, df_targets)
        except Exception as e:
            print(f"  ❌ llm_handler 호출 중 오류: {e}")
            continue

        if df_llm is None or df_llm.empty:
            print("  ⚠️ llm_handler 결과가 비어 있음")
            continue

        out_path = os.path.join(llm_result_dir, f"{safe_company_name(company)}_llm_diff.csv")
        df_llm.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  📄 LLM diff 결과 저장: {out_path}")

        # 평가
        metrics = evaluate_llm_vs_past(company, df_llm, diff_info["past_df"])
        if metrics:
            eval_metrics.append(metrics)
            print(
                f"  📊 평가 - acc={metrics['accuracy']:.3f}, "
                f"prec={metrics['precision']:.3f}, rec={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
            )

    if eval_metrics:
        df_eval = pd.DataFrame(eval_metrics)
        df_eval.to_csv(EVAL_METRIC_CSV, index=False, encoding="utf-8-sig")
        print(f"\n✅ 전체 LLM 평가 결과 저장: {EVAL_METRIC_CSV}")


# ==============================
# 5. company_info.csv 로더
# ==============================

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
# 6. LLM 핸들러 (RAG + 가족관계도 저장)
# ==============================
def apply_family_constraints_with_llm(
    company: str,
    ci: dict,
    df_result: pd.DataFrame,
) -> pd.DataFrame:
    """
    1차 LLM 결과(df_result)에 대해
    - UNIQUE_ROLES(아버지/어머니/배우자)는 1명 이하만 허용
    - 여러 명이면 LLM에게 다시 물어보고, 나머지는 '기타가족' 또는 '가족아님'으로 조정
    """
    df = df_result.copy()

    # 사람만 대상으로 제약 적용
    if "LLM_is_person" in df.columns:
        df_person = df[df["LLM_is_person"] == True].copy()
    else:
        df_person = df.copy()

    for role in UNIQUE_ROLES:
        mask_role = (
            (df_person["LLM_세부관계"] == role)
            & df_person["LLM_가족여부"].isin(["확실", "가족으로 예상됨"])
        )
        idxs = df_person[mask_role].index.tolist()

        # 0명 또는 1명이면 제약 적용 필요 없음
        if len(idxs) <= 1:
            continue

        # 후보들만 모아서 LLM에게 다시 물어보기
        candidates = df_person.loc[idxs]
        selected_name = reverify_unique_role_with_llm(
            company, ci, role, candidates
        )

        # 파싱 실패 → 일단 그대로 두고 넘어감
        if selected_name is None:
            continue

        # '없음'이면 모두 기타가족 처리
        if selected_name == "없음":
            for i in idxs:
                df.loc[i, "LLM_세부관계"] = "기타가족"
                # 가족 여부는 애매하므로 '가족으로 예상됨' 또는 '불확실'로 낮춤
                if df.loc[i, "LLM_가족여부"] == "확실":
                    df.loc[i, "LLM_가족여부"] = "가족으로 예상됨"
            continue

        # 선택된 한 명만 role 유지, 나머지는 다운그레이드
        for i in idxs:
            name_i = str(df.loc[i, "성명"]).strip()
            if name_i == selected_name:
                # 선택된 사람: 가족여부를 최소 '확실'로 승격 가능
                if df.loc[i, "LLM_가족여부"] == "가족으로 예상됨":
                    df.loc[i, "LLM_가족여부"] = "확실"
                continue

            # 나머지 후보들: 상식 제약에 의해 역할 변경
            df.loc[i, "LLM_세부관계"] = "기타가족"
            if df.loc[i, "LLM_가족여부"] == "확실":
                df.loc[i, "LLM_가족여부"] = "가족으로 예상됨"
            # 근거에 제약 적용 사실 메모
            old_reason = str(df.loc[i, "LLM_근거"])
            df.loc[i, "LLM_근거"] = (
                old_reason
                + f" / 상식 제약: '{role}' 역할은 한 명만 가능하여 다른 후보로 조정됨."
            )

    return df

def reverify_unique_role_with_llm(
    company: str,
    ci: dict,
    role: str,
    df_candidates: pd.DataFrame,
) -> Optional[str]:
    """
    특정 역할(role: '아버지', '어머니', '배우자' 등)에 대해
    여러 명이 동시에 표시된 경우,
    LLM에게 '가장 그럴듯한 사람 한 명만' 다시 골라달라고 요청.

    반환:
      - 선택된 이름 (문자열)
      - 또는 '없음'
      - 파싱 실패 시 None
    """
    if df_candidates.empty:
        return None

    ci_company_name = ci.get("회사명", company)
    ci_ceo         = ci.get("대표자명", "")

    # 후보 리스트를 문자열로 정리
    lines = []
    for i, (_, r) in enumerate(df_candidates.iterrows(), start=1):
        nm   = str(r.get("성명", "")).strip()
        rel0 = str(r.get("관계", "")).strip()
        rel1 = str(r.get("LLM_세부관계", "")).strip()
        fam  = str(r.get("LLM_가족여부", "")).strip()
        sh   = r.get("주식수_판단", "")
        reason = str(r.get("LLM_근거", "")).strip()

        lines.append(
            f"{i}) 이름: {nm}, 표기관계(원 테이블): {rel0}, "
            f"1차판단 세부관계: {rel1}, 1차 가족여부: {fam}, "
            f"주식수: {sh}, 1차 근거: {reason}"
        )

    candidates_block = "\n".join(lines)

    prompt = f"""
[회사 정보]
회사명: {ci_company_name}
대표자명(회장/CEO): {ci_ceo}

[역할 설명]
우리는 이 회사의 대표자(또는 오너)의 가족 관계를 조사하고 있습니다.
지금 논의하는 역할은 '{role}' 입니다.
한 가정에서 '{role}' 역할을 담당하는 사람은 보통 1명을 넘지 않습니다.

[후보 목록]
{candidates_block}

위 후보들 중에서, 상식과 지분 구조, 설명을 모두 고려했을 때
이 회사 대표자의 '{role}'일 가능성이 가장 높은 사람을
0명 또는 1명만 골라주세요.

반드시 아래 형식으로만 출력하세요 (다른 말은 쓰지 마세요):

선택된_이름: <없음 또는 위 후보 중 한 사람의 이름 그대로>
이유: <한 줄로 핵심만 설명>
"""

    messages = [
        {
            "role": "system",
            "content": (
                "당신은 기업 지배구조와 친족관계를 분석하는 전문가입니다. "
                "역할은 한 사람만 가질 수 있다는 상식을 고려해 후보 중 가장 타당한 사람을 고르세요."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        answer = query_llm_ollama(messages, temperature=0.1)
    except Exception:
        return None

    selected = None
    for line in (answer or "").splitlines():
        t = line.strip()
        if t.startswith("선택된_이름:"):
            selected = t.replace("선택된_이름:", "").strip()
            break

    if not selected:
        return None

    return selected  # '없음'일 수도 있음

def llm_handler_with_company_info(company: str, df_targets: pd.DataFrame) -> pd.DataFrame:
    """
    LLM handler that:
      1) Uses company_info.csv for background info.
      2) For each target shareholder, runs web search (DuckDuckGo) for RAG.
      3) If target name == 대표자명(본인) → '본인 / 확실'로 처리.
      4) Otherwise, calls LLM to judge:
           - 관계유형: <배우자 / 자녀 / 부모 / 형제자매 / 사위/며느리 / 기타가족 / 없음>
           - 가족여부: <확실 / 가족으로 예상됨 / 불확실>
    At the end, saves family graph (CSV + PNG) based on results.
    """
    df_targets = df_targets.copy()
    df_targets["회사명"] = company

    ci = get_company_info(company)
    ci_company_name   = ci.get("회사명", company)
    ci_code           = ci.get("종목코드", "")
    ci_industry       = ci.get("업종", "")
    ci_product        = ci.get("주요제품", "")
    ci_ceo            = ci.get("대표자명", "")
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

        # 대표자 == 인물명 → 본인
        if norm_ceo and norm_name and norm_ceo == norm_name:
            results.append(
                {
                    "회사명": company,
                    "성명": name,
                    "관계": rel,
                    "주식수_판단": shares,
                    "LLM_관계유형": "본인",
                    "LLM_가족여부": "확실",
                    "LLM_근거": "회사 대표자(본인)과 이름이 일치하여 본인으로 간주함.",
                }
            )
            continue

        # RAG: 웹 검색
        rag_snippets = web_search_snippets(ci_company_name, name, max_results=5)

        # 3) LLM 프롬프트 구성 (관계 세분화 + 인물/법인 구분)
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


[웹 검색 결과(RAG 컨텍스트)]
{rag_snippets}


[조사 대상 주주 정보]
주주 이름: {name}
주주 관계(표에 기재된 내용): {rel}
보유 주식수(추정): {shares}

위 정보를 바탕으로 이 이름이 '사람'인지, 아니면 '회사/법인/재단/계' 같은 조직 이름인지 먼저 구분해 주세요.
사람인 경우에는 이 인물이 회사의 대표자(또는 오너 일가)와 어떤 가족 관계인지 최대한 구체적으로 판단해 주세요.

1) entity_type 은 다음 중 하나로만 선택:
   - 사람
   - 회사/법인
   - 재단/계
   - 기타조직

2) 사람이면서 가족일 가능성이 있는 경우, 세부 관계는 다음 중 하나로 선택:
   - 본인
   - 배우자
   - 아들
   - 딸
   - 아버지
   - 어머니
   - 형제
   - 자매
   - 형제자매
   - 기타가족
   - 가족아님

3) 가족여부는:
   - 확실
   - 가족으로 예상됨
   - 불확실
   중에서 선택.

출력 형식은 반드시 아래와 같이 해주세요 (다른 말은 쓰지 마세요):

entity_type: <사람 / 회사/법인 / 재단/계 / 기타조직>
세부관계: <본인 / 배우자 / 아들 / 딸 / 아버지 / 어머니 / 형제 / 자매 / 형제자매 / 기타가족 / 가족아님>
가족여부: <확실 / 가족으로 예상됨 / 불확실>
근거요약: <한두 줄로 핵심만 설명>
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 기업 지배구조와 친족관계를 분석하는 전문가입니다. "
                    "웹 검색 결과와 회사 정보를 함께 고려하여 가족 여부를 신중히 판단하세요. "
                    "추측이 필요하면 '가족으로 예상됨' 또는 '불확실'을 사용하세요."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            answer = query_llm_ollama(messages, temperature=0.1)
        except Exception as e:
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
        entity_type   = "사람"
        detail_rel    = "가족아님"
        family_status = "불확실"
        reason        = "판단 근거 부족"

        for line in (answer or "").splitlines():
            t = line.strip()
            if t.startswith("entity_type:"):
                entity_type = t.replace("entity_type:", "").strip()
            elif t.startswith("세부관계:"):
                detail_rel = t.replace("세부관계:", "").strip()
            elif t.startswith("가족여부:"):
                family_status = t.replace("가족여부:", "").strip()
            elif t.startswith("근거요약:"):
                reason = t.replace("근거요약:", "").strip()

        # 5) 회사명/법인/재단/계 등은 가족 그래프/정확도에서 제외하고 싶다면 여기서 처리
        is_non_person = (
            entity_type != "사람"
            or normalize_name_for_match(name) == normalize_name_for_match(ci_company_name)
        )

        results.append(
            {
                "회사명": company,
                "성명": name,
                "관계": rel,
                "주식수_판단": shares,
                "LLM_entity_type": entity_type,
                "LLM_세부관계": detail_rel,
                "LLM_가족여부": family_status,
                "LLM_근거": reason,
                "LLM_is_person": not is_non_person,
            }
        )

    df_result = pd.DataFrame(results)

    df_result = apply_family_constraints_with_llm(company, ci, df_result)

    # 가족관계도(엣지 + PNG) 저장
    draw_family_graph(company, df_result, ci_ceo)

    return df_result


# ==============================
# 7. main
# ==============================

if __name__ == "__main__":
    # 1) 항상 최신 테이블로 갱신 + LLM/RAG/그래프/평가
    run_full_pipeline(
        recheck_all=True,        # True: 현재 테이블 전체를 LLM에 태움
        run_llm_if_changed=True,
    )