# tasks/etl_load_stg.py
from .dbconnect import get_engine  # ← 이미 고친 상태 유지
import pandas as pd
from sqlalchemy import text
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# 🔹 인코딩을 몇 가지 시도해보는 도우미
def read_csv_kr(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except Exception as e:
            last_err = e
    # 전부 실패하면 에러 다시 던짐
    raise last_err

def load_csv_to_staging():
    eng = get_engine()

    # 1) 경로
    f1 = DATA_DIR / "big_data_set1_f.csv"  # 점포 기본정보
    f2 = DATA_DIR / "big_data_set2_f.csv"  # 매출/운영정보
    f3 = DATA_DIR / "big_data_set3_f.csv"  # 연령/고객정보

    # 2) CSV 읽기 (인코딩 자동 시도)
    df1 = read_csv_kr(f1)
    df2 = read_csv_kr(f2)
    df3 = read_csv_kr(f3)

    # 3) 기존 STAGING 비우기
    with eng.begin() as cx:
        cx.execute(text("TRUNCATE TABLE stg_store"))
        cx.execute(text("TRUNCATE TABLE stg_perf"))
        cx.execute(text("TRUNCATE TABLE stg_demo"))

    # 4) 적재
    df1.to_sql("stg_store", eng, if_exists="append", index=False)
    df2.to_sql("stg_perf",  eng, if_exists="append", index=False)
    df3.to_sql("stg_demo",  eng, if_exists="append", index=False)

    print("STAGING 적재 ")

if __name__ == "__main__":
    load_csv_to_staging()
