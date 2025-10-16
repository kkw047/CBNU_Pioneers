import sys, pathlib
from tqdm import tqdm
from sqlalchemy import text

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

# 데이터 받아오기 및 위치정보에 공간인덱스 설정
A = [], B = [], C = []

READ_QUERY = r"""SELECT mct_id FROM mat_basic_crisis"""
mct_id_list = fetch_sql(READ_QUERY)

for mct_id in tqdm(mct_id_list, total=len(mct_id_list)):
    READ_QUERY_A = r"""SELECT b.close_dt FROM mat_basic_crisis AS a JOIN mat_basic_crisis AS b
    ON a.mct_id = :mct_id AND a.mct_id != b.mct_id AND a.brand_code = b.brand_code
    AND MBRWithin(POINT(ST_Y(b.location), ST_X(b.location)), ST_Buffer(POINT(ST_Y(a.location), ST_X(a.location)), 0.009))
    WHERE ST_Distance_Sphere(POINT(ST_Y(a.location), ST_X(a.location)), POINT(ST_Y(b.location), ST_X(b.location))) <= 1000;"""

    mct_id_list_A = fetch_sql(READ_QUERY_A, {'mct_id': mct_id[0]})
    total_A = len(mct_id_list_A)
    closed_A = sum(1 for row in mct_id_list_A if row.close_dt is not None)
    rate_A = closed_A / total_A if total_A > 0 else 0
    A.append(rate_A)

    READ_QUERY_B = r"""SELECT b.close_dt FROM mat_basic_crisis AS a JOIN mat_basic_crisis AS b
    ON a.brand_code = b.brand_code WHERE a.mct_id = :mct_id AND a.mct_id != b.mct_id;"""

    mct_id_list_B = fetch_sql(READ_QUERY_B, {'mct_id': mct_id[0]})
    total_B = len(mct_id_list_B)
    closed_B = sum(1 for row in mct_id_list_B if row.close_dt is not None)
    rate_B = closed_B / total_B if total_B > 0 else 0
    B.append(rate_B)

    READ_QUERY_C = r"""SELECT b.close_dt FROM mat_basic_crisis AS a JOIN mat_basic_crisis AS b
    ON a.mct_id = :mct_id AND a.mct_id != b.mct_id
    AND MBRWithin(POINT(ST_Y(b.location), ST_X(b.location)), ST_Buffer(POINT(ST_Y(a.location), ST_X(a.location)), 0.009))
    WHERE ST_Distance_Sphere(POINT(ST_Y(a.location), ST_X(a.location)), POINT(ST_Y(b.location), ST_X(b.location))) <= 1000;"""

    mct_id_list_C = fetch_sql(READ_QUERY_C, {'mct_id': mct_id[0]})
    total_C = len(mct_id_list_C)
    closed_C = sum(1 for row in mct_id_list_C if row.close_dt is not None)
    rate_C = closed_C / total_C if total_C > 0 else 0
    C.append(rate_C)

for mct_id, rate_A, rate_B, rate_C in tqdm(zip(mct_id_list, A, B, C), total=len(mct_id_list)):
    UPDATE_QUERY = r"""UPDATE mat_prompt_bundle_crisis SET close_rate_A = :rate_A, close_rate_B = :rate_B, close_rate_C = :rate_C
    WHERE mct_id = :mct_id;"""
    exec_sql(UPDATE_QUERY, {"rate_A" : rate_A, "rate_B" : rate_B, "rate_C" : rate_C, "mct_id" : mct_id})
