# mcp_server.py
import glob
from typing import Dict, Any, Optional

import pandas as pd
from fastmcp.server import FastMCP

# git commit ex) mcp_server.py

# 전역 DF
DF: Optional[pd.DataFrame] = None

mcp = FastMCP(
    "MerchantSearchServer",
    instructions=(
        "data/*.csv를 모두 로드해 하나의 DataFrame으로 통합하고, "
        "가맹점명을 나타내는 컬럼을 자동 탐지하여 부분 일치(대소문자 무시) 검색을 제공합니다. "
        "검색 결과에는 CSV에 존재하는 가맹점 관련 컬럼들이 그대로 포함됩니다."
    ),
)

def _load_df() -> pd.DataFrame:
    """data/*.csv 모두 읽어 DF로 결합 (인코딩 자동 시도)"""
    global DF
    files = glob.glob("./data/*.csv")
    if not files:
        raise FileNotFoundError("data 폴더에 CSV가 없습니다.")

    dfs = []
    for f in files:
        df = None
        for enc in (None, "utf-8-sig", "cp949"):
            try:
                df = pd.read_csv(f, encoding=enc) if enc else pd.read_csv(f)
                break
            except Exception:
                continue
        if df is not None:
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("CSV 로드에 실패했습니다. 인코딩을 확인하세요.")

    DF = pd.concat(dfs, ignore_index=True)
    return DF

# 서버 시작 시 데이터 로드
_load_df()

def _find_merchant_col(df: pd.DataFrame) -> Optional[str]:
    """가맹점명에 해당하는 컬럼 자동 탐지"""
    cols = [str(c) for c in df.columns]

    # 1) 정확 매칭(한글)
    if "가맹점명" in cols:
        return "가맹점명"

    # 2) 한글 후보 정확 매칭
    for k in ["상호", "상호명", "업체명", "점포명", "매장명"]:
        if k in cols:
            return k

    # 3) 부분 포함(한글/영문 키워드)
    partial_keys = [
        "가맹점", "상호", "업체", "점포", "매장",      # 한글
        "mct_nm", "merchant", "store", "shop", "brand", "name"  # 영문
    ]
    low = {c.lower(): c for c in cols}
    for key in partial_keys:
        key = key.lower()
        for lc, orig in low.items():
            if key in lc:
                return orig

    return None

@mcp.tool()
def search_merchant(merchant_name: str) -> Dict[str, Any]:
    """
    자동 탐지한 가맹점명 컬럼으로 부분 일치(대소문자 무시) 검색
    """
    assert DF is not None, "DataFrame이 초기화되지 않았습니다."
    col = _find_merchant_col(DF)
    if not col:
        return {
            "found": False,
            "message": "가맹점명을 나타내는 컬럼을 찾지 못했습니다. 헤더명을 확인해 주세요.",
            "count": 0,
            "merchants": [],
            "columns": list(map(str, DF.columns))[:100],
        }

    needle = str(merchant_name).replace("*", "").strip().lower()
    series = DF[col].astype(str).str.replace("*", "", regex=False).str.lower()
    result = DF[series.str.contains(needle, na=False)]

    if result.empty:
        return {
            "found": False,
            "message": f"'{merchant_name}' 검색 결과가 없습니다. (사용 컬럼: {col})",
            "count": 0,
            "merchants": [],
        }

    merchants = result.to_dict(orient="records")
    return {
        "found": True,
        "message": f"'{merchant_name}' 검색 결과 {len(merchants)}건 (사용 컬럼: {col})",
        "count": len(merchants),
        "merchants": merchants,
    }

if __name__ == "__main__":
    mcp.run()
