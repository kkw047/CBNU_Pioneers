# tasks/stats_association.py
import sys, pathlib
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sqlalchemy import text

# ---------- import plumbing (ROOT/tasks 경로 모두 보장) ----------
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]          # CBNU_DATA/
TASKS_DIR = HERE.parent         # CBNU_DATA/tasks/
for p in (str(ROOT), str(TASKS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    # 루트에 dbconnect.py가 있는 경우
    from dbconnect import get_engine, exec_sql
except ModuleNotFoundError:
    # tasks 폴더에 있는 경우(대비)
    from tasks.dbconnect import get_engine, exec_sql

# ---------- DDL: 없으면 생성 ----------
DDL = r"""
CREATE TABLE IF NOT EXISTS fx_age_bucket (
  mct_id VARCHAR(32),
  ym VARCHAR(8),
  age_bucket VARCHAR(8),
  PRIMARY KEY (mct_id, ym)
);

CREATE TABLE IF NOT EXISTS fx_aud_bucket (
  mct_id VARCHAR(32),
  ym VARCHAR(8),
  aud_bucket VARCHAR(8),
  PRIMARY KEY (mct_id, ym)
);

CREATE TABLE IF NOT EXISTS mat_assoc (
  pair_name VARCHAR(64),
  chi2 DOUBLE,
  dof INT,
  p_value DOUBLE,
  cramers_v DOUBLE,
  n INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (pair_name, created_at)
);
"""

# ---------- 파생 버킷 적재 ----------
SQL_AGE_BUCKET = r"""
INSERT INTO fx_age_bucket (mct_id, ym, age_bucket)
SELECT mct_id, ym,
       CASE
         WHEN age_10s>=GREATEST(age_30s,age_40s,age_50s,age_60p) THEN '10s'
         WHEN age_30s>=GREATEST(age_10s,age_40s,age_50s,age_60p) THEN '30s'
         WHEN age_40s>=GREATEST(age_10s,age_30s,age_50s,age_60p) THEN '40s'
         WHEN age_50s>=GREATEST(age_10s,age_30s,age_40s,age_60p) THEN '50s'
         ELSE '60p'
       END
FROM fx_age_mix
ON DUPLICATE KEY UPDATE age_bucket=VALUES(age_bucket);
"""

SQL_AUD_BUCKET = r"""
INSERT INTO fx_aud_bucket (mct_id, ym, aud_bucket)
SELECT mct_id, ym,
       CASE
         WHEN NEW>=GREATEST(REU,RES,WORK,FLOW) THEN 'NEW'
         WHEN REU>=GREATEST(NEW,RES,WORK,FLOW) THEN 'REU'
         WHEN RES>=GREATEST(NEW,REU,WORK,FLOW) THEN 'RES'
         WHEN WORK>=GREATEST(NEW,REU,RES,FLOW) THEN 'WORK'
         ELSE 'FLOW'
       END
FROM fx_audience_type
ON DUPLICATE KEY UPDATE aud_bucket=VALUES(aud_bucket);
"""

# ---------- 분석 데이터 뽑기 ----------
SQL_FETCH = r"""
SELECT a.mct_id, a.ym, ag.age_bucket, au.aud_bucket,
       f.cancel_high, f.delivery_high
FROM fx_age_bucket ag
JOIN fx_aud_bucket au ON au.mct_id=ag.mct_id AND au.ym=ag.ym
JOIN fx_age_mix a     ON a.mct_id=ag.mct_id AND a.ym=ag.ym
LEFT JOIN fx_perf_flags f ON f.mct_id=ag.mct_id AND f.ym=ag.ym
"""

def cramers_v_from_ct(ct: np.ndarray):
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.sum()
    k = min(ct.shape)
    V = np.sqrt((chi2 / n) / (k - 1)) if k > 1 and n > 0 else 0.0
    return chi2, dof, p, V, int(n)

def save_assoc(engine, pair_name, chi2, dof, p, V, n):
    with engine.begin() as cx:
        cx.execute(text("""
            INSERT INTO mat_assoc (pair_name, chi2, dof, p_value, cramers_v, n)
            VALUES (:p, :c, :d, :pv, :v, :n)
        """), {"p": pair_name, "c": float(chi2), "d": int(dof), "pv": float(p), "v": float(V), "n": int(n)})

def run():
    eng = get_engine()

    # DDL 실행
    for stmt in DDL.strip().split(";\n\n"):
        s = stmt.strip()
        if s:
            exec_sql(s + ";")


    exec_sql(SQL_AGE_BUCKET); print("• fx_age_bucket ok")
    exec_sql(SQL_AUD_BUCKET); print("• fx_aud_bucket ok")


    df = pd.read_sql(SQL_FETCH, eng)
    print(f"• dataset rows: {len(df)}")


    ct1 = pd.crosstab(df["age_bucket"], df["aud_bucket"]).values
    chi2, dof, p, V, n = cramers_v_from_ct(ct1)
    save_assoc(eng, "age_bucket~aud_bucket", chi2, dof, p, V, n)
    print(f"[age_bucket~aud_bucket] V={V:.3f}, p={p:.3g}, n={n}")


    d2 = df.dropna(subset=["cancel_high"]).copy()
    d2["cancel_high"] = d2["cancel_high"].astype(int)
    ct2 = pd.crosstab(d2["age_bucket"], d2["cancel_high"]).values
    chi2, dof, p, V, n = cramers_v_from_ct(ct2)
    save_assoc(eng, "age_bucket~cancel_high", chi2, dof, p, V, n)
    print(f"[age_bucket~cancel_high] V={V:.3f}, p={p:.3g}, n={n}")

    d3 = df.dropna(subset=["delivery_high"]).copy()
    d3["delivery_high"] = d3["delivery_high"].astype(int)
    ct3 = pd.crosstab(d3["aud_bucket"], d3["delivery_high"]).values
    chi2, dof, p, V, n = cramers_v_from_ct(ct3)
    save_assoc(eng, "aud_bucket~delivery_high", chi2, dof, p, V, n)
    print(f"[aud_bucket~delivery_high] V={V:.3f}, p={p:.3g}, n={n}")

    print("chi-square & Cramér’s V saved → mat_assoc")

if __name__ == "__main__":
    run()
