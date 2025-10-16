# tasks/etl_build_fx.py
import sys, pathlib
from sqlalchemy import text

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]           # CBNU_DATA/
TASKS_DIR = HERE.parent          # CBNU_DATA/tasks/

for p in (str(ROOT), str(TASKS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dbconnect import exec_sql
except ModuleNotFoundError:
    from tasks.dbconnect import exec_sql


# --------------------------------------------------
# dim_store : stg_store → 도로명 추출 포함
# --------------------------------------------------
SQL_DIM_STORE = r"""
INSERT INTO dim_store (mct_id, name, sector, biz_zone, sigungu, addr, road_name, brand_code, open_dt, close_dt)
SELECT 
    ENCODED_MCT                           AS mct_id,
    MCT_NM                                AS name,
    HPSN_MCT_ZCD_NM                       AS sector,
    HPSN_MCT_BZN_CD_NM                    AS biz_zone,
    MCT_SIGUNGU_NM                        AS sigungu,
    MCT_BSE_AR                            AS addr,
    REGEXP_SUBSTR(MCT_BSE_AR, '([가-힣A-Za-z0-9\\- ]+(로|길))') AS road_name,
    MCT_BRD_NUM                           AS brand_code,
    ARE_D                                  AS open_dt,
    MCT_ME_D                               AS close_dt
FROM stg_store
ON DUPLICATE KEY UPDATE
    name=VALUES(name),
    sector=VALUES(sector),
    biz_zone=VALUES(biz_zone),
    sigungu=VALUES(sigungu),
    addr=VALUES(addr),
    road_name=VALUES(road_name),
    brand_code=VALUES(brand_code),
    open_dt=VALUES(open_dt),
    close_dt=VALUES(close_dt);
"""

# --------------------------------------------------
# fx_age_mix : 남/녀 합을 신호로 쓰고, 5개 합=1로 정규화 (CTE/LATERAL 없이 호환)
# --------------------------------------------------
SQL_FX_AGE = r"""
INSERT INTO fx_age_mix (mct_id, ym, age_10s, age_30s, age_40s, age_50s, age_60p)
SELECT
  dn.mct_id,
  dn.ym,
  ROUND(IFNULL(dn.s10 / NULLIF(dn.total, 0), 0), 6) AS age_10s,
  ROUND(IFNULL(dn.s30 / NULLIF(dn.total, 0), 0), 6) AS age_30s,
  ROUND(IFNULL(dn.s40 / NULLIF(dn.total, 0), 0), 6) AS age_40s,
  ROUND(IFNULL(dn.s50 / NULLIF(dn.total, 0), 0), 6) AS age_50s,
  ROUND(IFNULL(dn.s60 / NULLIF(dn.total, 0), 0), 6) AS age_60p
FROM (
  SELECT
    base.mct_id,
    base.ym,
    base.s10, base.s30, base.s40, base.s50, base.s60,
    (base.s10 + base.s30 + base.s40 + base.s50 + base.s60) AS total
  FROM (
    SELECT
      ENCODED_MCT AS mct_id,
      TA_YM       AS ym,

      /* 0~100(%) 또는 센티넬(-999999.9) → 0~1로 정리 후 남/녀 합 */
      (IFNULL(NULLIF(LEAST(GREATEST(M12_MAL_1020_RAT,0),100),-999999.9)/100.0,0)
       + IFNULL(NULLIF(LEAST(GREATEST(M12_FME_1020_RAT,0),100),-999999.9)/100.0,0)) AS s10,

      (IFNULL(NULLIF(LEAST(GREATEST(M12_MAL_30_RAT ,0),100),-999999.9)/100.0,0)
       + IFNULL(NULLIF(LEAST(GREATEST(M12_FME_30_RAT ,0),100),-999999.9)/100.0,0)) AS s30,

      (IFNULL(NULLIF(LEAST(GREATEST(M12_MAL_40_RAT ,0),100),-999999.9)/100.0,0)
       + IFNULL(NULLIF(LEAST(GREATEST(M12_FME_40_RAT ,0),100),-999999.9)/100.0,0)) AS s40,

      (IFNULL(NULLIF(LEAST(GREATEST(M12_MAL_50_RAT ,0),100),-999999.9)/100.0,0)
       + IFNULL(NULLIF(LEAST(GREATEST(M12_FME_50_RAT ,0),100),-999999.9)/100.0,0)) AS s50,

      (IFNULL(NULLIF(LEAST(GREATEST(M12_MAL_60_RAT ,0),100),-999999.9)/100.0,0)
       + IFNULL(NULLIF(LEAST(GREATEST(M12_FME_60_RAT ,0),100),-999999.9)/100.0,0)) AS s60
    FROM stg_demo
  ) AS base
) AS dn
ON DUPLICATE KEY UPDATE
  age_10s = VALUES(age_10s),
  age_30s = VALUES(age_30s),
  age_40s = VALUES(age_40s),
  age_50s = VALUES(age_50s),
  age_60p = VALUES(age_60p);
"""

# --------------------------------------------------
# fx_audience_type : CTE 없이 서브쿼리로 변환(0~1 스케일)
# --------------------------------------------------
SQL_FX_AUD = r"""
INSERT INTO fx_audience_type (mct_id, ym, NEW, REU, RES, WORK, FLOW)
SELECT
  da.mct_id,
  da.ym,
  ROUND(COALESCE(da.v_new ,0), 4),
  ROUND(COALESCE(da.v_reu ,0), 4),
  ROUND(COALESCE(da.v_res ,0), 4),
  ROUND(COALESCE(da.v_work,0), 4),
  ROUND(COALESCE(da.v_flow,0), 4)
FROM (
  SELECT
    ENCODED_MCT AS mct_id,
    TA_YM       AS ym,
    NULLIF(LEAST(GREATEST((MCT_UE_CLN_NEW_RAT),0),100), -999999.9)/100.0  AS v_new,
    NULLIF(LEAST(GREATEST((MCT_UE_CLN_REU_RAT),0),100), -999999.9)/100.0  AS v_reu,
    NULLIF(LEAST(GREATEST((RC_M1_SHC_RSD_UE_CLN_RAT),0),100), -999999.9)/100.0 AS v_res,
    NULLIF(LEAST(GREATEST((RC_M1_SHC_WP_UE_CLN_RAT ),0),100), -999999.9)/100.0 AS v_work,
    NULLIF(LEAST(GREATEST((RC_M1_SHC_FLP_UE_CLN_RAT),0),100), -999999.9)/100.0 AS v_flow
  FROM stg_demo
) AS da
ON DUPLICATE KEY UPDATE
  NEW  = VALUES(NEW),
  REU  = VALUES(REU),
  RES  = VALUES(RES),
  WORK = VALUES(WORK),
  FLOW = VALUES(FLOW);
"""

# --------------------------------------------------
# fx_perf_flags : 취소율/배달 강도 플래그
# --------------------------------------------------
SQL_FX_FLAGS = r"""
INSERT INTO fx_perf_flags (mct_id, ym, cancel_high, delivery_high)
SELECT 
    p.ENCODED_MCT,
    p.TA_YM,
    CASE WHEN LOWER(COALESCE(p.APV_CE_RAT,'')) REGEXP '상|high' THEN 1 ELSE 0 END,
    CASE WHEN NULLIF(p.DLV_SAA_RAT, -999999.9) > 0.5 THEN 1 ELSE 0 END
FROM stg_perf p
ON DUPLICATE KEY UPDATE
    cancel_high=VALUES(cancel_high),
    delivery_high=VALUES(delivery_high);
"""

# --------------------------------------------------
# fx_region_age_avg : 업종×상권×월 평균 연령 믹스
# --------------------------------------------------
SQL_REGION_AVG = r"""
INSERT INTO fx_region_age_avg (sector, biz_zone, ym, age_10s, age_30s, age_40s, age_50s, age_60p)
SELECT 
    s.HPSN_MCT_ZCD_NM        AS sector,
    s.HPSN_MCT_BZN_CD_NM     AS biz_zone,
    a.ym,
    AVG(a.age_10s),
    AVG(a.age_30s),
    AVG(a.age_40s),
    AVG(a.age_50s),
    AVG(a.age_60p)
FROM fx_age_mix a
JOIN stg_store s 
  ON s.ENCODED_MCT = a.mct_id
WHERE s.HPSN_MCT_ZCD_NM    IS NOT NULL
  AND s.HPSN_MCT_BZN_CD_NM IS NOT NULL
GROUP BY s.HPSN_MCT_ZCD_NM, s.HPSN_MCT_BZN_CD_NM, a.ym
ON DUPLICATE KEY UPDATE
    age_10s=VALUES(age_10s),
    age_30s=VALUES(age_30s),
    age_40s=VALUES(age_40s),
    age_50s=VALUES(age_50s),
    age_60p=VALUES(age_60p);
"""

def run():
    exec_sql(SQL_DIM_STORE);   print("• dim_store ok")
    exec_sql(SQL_FX_AGE);      print("• fx_age_mix ok")
    exec_sql(SQL_FX_AUD);      print("• fx_audience_type ok")
    exec_sql(SQL_FX_FLAGS);    print("• fx_perf_flags ok")
    exec_sql(SQL_REGION_AVG);  print("• fx_region_age_avg ok")
    print("전처리(DIM/FX) 완료")

if __name__ == "__main__":
    run()
