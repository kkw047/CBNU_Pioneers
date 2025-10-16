import sys, pathlib
import pandas as pd
import requests, time
import streamlit as st

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parent[1]
TASKS_DIR = HERE.parent

for p in (str(ROOT), str(TASKS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dbconnect import fetch_sql, exec_sql, exec_sql_script
except ModuleNotFoundError:
    from tasks.dbconnect import fetch_sql, exec_sql, exec_sql_script

KAKAO_API_KEY = st.secrets.get("KAKAO_API_KEY", "")

def geocoding(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    headers = {"Authorization" : f"KakaoAK {KAKAO_API_KEY}"}
    params = {"query" : address}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()

        if result["documents"]:
            location = result["documents"][0]

            return location['x'], location['y']
        else:
            return None, None
    else:
        return None, None

INSERT_BASIC_QUERY =r"""
INSERT INTO mat_basic_crisis (mct_id, addr, brand_code, close_dt) 
SELECT mct_id, addr, brand_code, close_dt 
FROM (
    SELECT first.ENCODED_MCT AS mct_id, first.MCT_BSE_AR AS addr, 
        first.HPSN_MCT_ZCD_NM AS brand_code, first.MCT_ME_D AS close_dt
    FROM stg_store first 
) AS sub
ON DUPLICATE KEY UPDATE
    addr = sub.addr,
    brand_code = sub.brand_code,
    close_dt = sub.close_dt;
"""

SPATIAL_INDEX_QUERY = r"""
SET SQL_SAFE_UPDATES = 0;
DELETE FROM mat_basic_crisis WHERE location IS NULL;
ALTER TABLE mat_basic_crisis MODIFY COLUMN location POINT NOT NULL SRID 4326;
CREATE SPATIAL INDEX sp_idx_location ON mat_basic_crisis(location);
CREATE INDEX idx_brand_code ON mat_basic_crisis(brand_code);
SET SQL_SAFE_UPDATES = 1;
"""

INSERT_SQL = r"""
INSERT INTO mat_prompt_bundle_crisis(
    mct_id, addr, resident_user, floating_user
)
SELECT mct_id, addr, ROUND(AVG(resident_user), 5) AS avg_resident_user, ROUND(AVG(floating_user), 5) AS avg_floating_user
FROM (
    SELECT third.ENCODED_MCT AS mct_id, first.addr AS addr,
        COALESCE(NULLIF(LEAST(GREATEST(third.RC_M1_SHC_RSD_UE_CLN_RAT, 0), 100), -999999.9) / 100.0, 0) AS resident_user,
        COALESCE(NULLIF(LEAST(GREATEST(third.RC_M1_SHC_FLP_UE_CLN_RAT, 0), 100), -999999.9) / 100.0, 0) AS floating_user
    FROM stg_demo third
    JOIN mat_basic_crisis first ON third.ENCODED_MCT = first.mct_id
) AS monthly_normalized_data GROUP BY mct_id, addr
ON DUPLICATE KEY UPDATE
    addr = VALUES(addr),
    resident_user = VALUES(resident_user),
    floating_user = VALUES(floating_user);
"""

INSERT_LOCATION_QUERY = r"""
UPDATE mat_prompt_bundle_crisis AS prompt JOIN mat_basic_crisis AS basic ON prompt.mct_id = basic.mct_id
SET prompt.location = basic.location;
ALTER TABLE mat_prompt_bundle_crisis MODIFY COLUMN location POINT NOT NULL SRID 4326;
CREATE SPATIAL INDEX sp_idx_location_prompt ON mat_prompt_bundle_crisis(location);
"""

# UPDATE_CLOSE_RATE_QUERY = r"""
# SET SQL_SAFE_UPDATES = 0;
# UPDATE mat_prompt_bundle_crisis AS prompt JOIN mat_basic_crisis AS b ON prompt.mct_id = b.mct_id
# LEFT JOIN (
#     SELECT brand_code, COUNT(CASE WHEN close_dt IS NOT NULL THEN 1 END) / COUNT(*) AS rate
#     FROM mat_basic_crisis GROUP BY brand_code
# ) AS rates_B ON b.brand_code = rates_B.brand_code
# LEFT JOIN (
#     SELECT b1.mct_id,
#         SUM(CASE WHEN b1.brand_code = b2.brand_code AND b2.close_dt IS NOT NULL THEN 1 ELSE 0 END) /
#         NULLIF(SUM(CASE WHEN b1.brand_code = b2.brand_code THEN 1 ELSE 0 END), 0) AS rates_A,
#         SUM(CASE WHEN b2.close_dt IS NOT NULL THEN 1 ELSE 0 END) /
#         NULLIF(COUNT(b2.mct_id), 0) AS rates_C
#     FROM mat_basic_crisis AS b1 JOIN mat_basic_crisis AS b2
#     ON MBRWithin(POINT(ST_Y(b2.location), ST_X(b2.location)), ST_Buffer(POINT(ST_Y(b1.location), ST_X(b1.location)), 0.0045))
#     AND ST_Distance_Sphere(POINT(ST_Y(b1.location), ST_X(b1.location)), POINT(ST_Y(b2.location), ST_X(b2.location))) <= 500
#     GROUP BY b1.mct_id
# ) AS rates_AC ON prompt.mct_id = rates_AC.mct_id
#
# SET prompt.close_rate_A = COALESCE(rates_AC.rates_A, 0),
#     prompt.close_rate_B = COALESCE(rates_B.rate, 0),
#     prompt.close_rate_C = COALESCE(rates_AC.rate_C, 0);
# SET SQL_SAFE_UPDATES = 1;
# """

def run():
    exec_sql(INSERT_BASIC_QUERY)

    select_query = r"""SELECT mct_id, addr, FROM mat_basic_crisis"""
    need_geocoding = fetch_sql(select_query)

    for row in need_geocoding:
        mct_id, addr = row

        try:
            longitude, latitude = geocoding(addr)
            if longitude and latitude:
                update_query = r"""UPDATE mat_basic_crisis SET location = ST_SRID(POINT(:longitude, :latitude), 4326) WHERE mct_id = :mct_id"""
                exec_sql(update_query, {"longitude": longitude, "latitude": latitude, "mct_id" : mct_id})
            time.sleep(0.1)
        except Exception as e:
            print(f"An error occurred while processing {addr} : {e}")
    exec_sql_script(SPATIAL_INDEX_QUERY)

    exec_sql(INSERT_SQL)
    exec_sql(INSERT_LOCATION_QUERY)

if __name__ == '__main__':
    run()