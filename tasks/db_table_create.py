from .dbconnect import exec_sql_script

DDL = r"""
USE data_llm;

-- STAGING
DROP TABLE IF EXISTS stg_store;
CREATE TABLE stg_store (
  ENCODED_MCT VARCHAR(64) PRIMARY KEY,
  MCT_BSE_AR  VARCHAR(255),
  MCT_NM      VARCHAR(255),
  MCT_BRD_NUM VARCHAR(64),
  MCT_SIGUNGU_NM VARCHAR(100),
  HPSN_MCT_ZCD_NM VARCHAR(100),
  HPSN_MCT_BZN_CD_NM VARCHAR(100),
  ARE_D DATE NULL,
  MCT_ME_D DATE NULL
);

DROP TABLE IF EXISTS stg_perf;
CREATE TABLE stg_perf (
  ENCODED_MCT VARCHAR(64),
  TA_YM CHAR(6),
  MCT_OPE_MS_CN VARCHAR(50),
  RC_M1_SAA DECIMAL(12,2) NULL,
  RC_M1_TO_UE_CT DECIMAL(12,2) NULL,
  RC_M1_UE_CUS_CN VARCHAR(50),
  RC_M1_AV_NP_AT DECIMAL(12,2) NULL,
  APV_CE_RAT VARCHAR(50),
  DLV_SAA_RAT DECIMAL(6,4) NULL,
  M1_SME_RY_SAA_RAT DECIMAL(6,4) NULL,
  M1_SME_RY_CNT_RAT DECIMAL(6,4) NULL,
  M12_SME_RY_SAA_PCE_RT DECIMAL(6,4) NULL,
  M12_SME_BZN_SAA_PCE_RT DECIMAL(6,4) NULL,
  M12_SME_RY_ME_MCT_RAT DECIMAL(6,4) NULL,
  M12_SME_BZN_ME_MCT_RAT DECIMAL(6,4) NULL,
  PRIMARY KEY (ENCODED_MCT, TA_YM)
);

DROP TABLE IF EXISTS stg_demo;
CREATE TABLE stg_demo (
  ENCODED_MCT VARCHAR(64),
  TA_YM CHAR(6),
  M12_MAL_1020_RAT DECIMAL(6,2) NULL,
  M12_MAL_30_RAT DECIMAL(6,2) NULL,
  M12_MAL_40_RAT DECIMAL(6,2) NULL,
  M12_MAL_50_RAT DECIMAL(6,2) NULL,
  M12_MAL_60_RAT DECIMAL(6,2) NULL,
  M12_FME_1020_RAT DECIMAL(6,2) NULL,
  M12_FME_30_RAT DECIMAL(6,2) NULL,
  M12_FME_40_RAT DECIMAL(6,2) NULL,
  M12_FME_50_RAT DECIMAL(6,2) NULL,
  M12_FME_60_RAT DECIMAL(6,2) NULL,
  MCT_UE_CLN_REU_RAT DECIMAL(6,2) NULL,
  MCT_UE_CLN_NEW_RAT DECIMAL(6,2) NULL,
  RC_M1_SHC_RSD_UE_CLN_RAT DECIMAL(6,2) NULL,
  RC_M1_SHC_WP_UE_CLN_RAT DECIMAL(6,2) NULL,
  RC_M1_SHC_FLP_UE_CLN_RAT DECIMAL(6,2) NULL,
  PRIMARY KEY (ENCODED_MCT, TA_YM)
);

-- DIM / FX
DROP TABLE IF EXISTS dim_store;
CREATE TABLE dim_store (
  mct_id VARCHAR(64) PRIMARY KEY,
  name VARCHAR(255),
  sector VARCHAR(100),
  biz_zone VARCHAR(100),
  sigungu VARCHAR(100),
  addr VARCHAR(255),
  road_name VARCHAR(120),
  brand_code VARCHAR(64),
  open_dt DATE NULL,
  close_dt DATE NULL
);

DROP TABLE IF EXISTS fx_age_mix;
CREATE TABLE fx_age_mix (
  mct_id VARCHAR(64),
  ym CHAR(6),
  age_10s DECIMAL(6,4),
  age_30s DECIMAL(6,4),
  age_40s DECIMAL(6,4),
  age_50s DECIMAL(6,4),
  age_60p DECIMAL(6,4),
  PRIMARY KEY (mct_id, ym)
);

DROP TABLE IF EXISTS fx_audience_type;
CREATE TABLE fx_audience_type (
  mct_id VARCHAR(64),
  ym CHAR(6),
  NEW DECIMAL(6,4),
  REU DECIMAL(6,4),
  RES DECIMAL(6,4),
  WORK DECIMAL(6,4),
  FLOW DECIMAL(6,4),
  PRIMARY KEY (mct_id, ym)
);

DROP TABLE IF EXISTS fx_perf_flags;
CREATE TABLE fx_perf_flags (
  mct_id VARCHAR(64),
  ym CHAR(6),
  cancel_high TINYINT(1),
  delivery_high TINYINT(1),
  PRIMARY KEY (mct_id, ym)
);

DROP TABLE IF EXISTS fx_region_age_avg;
CREATE TABLE fx_region_age_avg (
  sector VARCHAR(100),
  biz_zone VARCHAR(100),
  ym CHAR(6),
  age_10s DECIMAL(6,4),
  age_30s DECIMAL(6,4),
  age_40s DECIMAL(6,4),
  age_50s DECIMAL(6,4),
  age_60p DECIMAL(6,4),
  PRIMARY KEY (sector, biz_zone, ym)
);

-- Affinity
DROP TABLE IF EXISTS dim_age_affinity_media;
CREATE TABLE dim_age_affinity_media (
  version VARCHAR(32),
  age_group VARCHAR(10),
  channel VARCHAR(50),
  affinity DECIMAL(6,4),
  PRIMARY KEY (version, age_group, channel)
);

DROP TABLE IF EXISTS dim_age_affinity_keyword;
CREATE TABLE dim_age_affinity_keyword (
  version VARCHAR(32),
  age_group VARCHAR(10),
  keyword VARCHAR(50),
  affinity DECIMAL(6,4),
  PRIMARY KEY (version, age_group, keyword)
);

-- 결과
DROP TABLE IF EXISTS mat_age_channel_score;
CREATE TABLE mat_age_channel_score (
  mct_id VARCHAR(64),
  ym CHAR(6),
  version VARCHAR(32),
  channel VARCHAR(50),
  score DECIMAL(8,4),
  PRIMARY KEY (mct_id, ym, version, channel)
);

DROP TABLE IF EXISTS mat_age_keyword_score;
CREATE TABLE mat_age_keyword_score (
  mct_id VARCHAR(64),
  ym CHAR(6),
  version VARCHAR(32),
  keyword VARCHAR(50),
  score DECIMAL(8,4),
  PRIMARY KEY (mct_id, ym, version, keyword)
);

DROP TABLE IF EXISTS mat_prompt_bundle;
CREATE TABLE mat_prompt_bundle (
  mct_id   VARCHAR(64),
  ym       CHAR(6),
  version  VARCHAR(32),
  bundle   JSON,
  PRIMARY KEY (mct_id, ym, version)
);
"""

def run():
    exec_sql_script(DDL)
    print("테이블 생성")

if __name__ == "__main__":
    run()
