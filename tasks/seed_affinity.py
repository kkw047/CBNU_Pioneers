# tasks/seed_affinity.py
import re
import math
from pathlib import Path

import pandas as pd
from sqlalchemy import text

# ✅ 같은 폴더의 dbconnect.py 사용
from .dbconnect import get_engine

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VERSION = "v1_2025Q3"


def map_age(raw: str) -> str | None:
    """
    엑셀의 다양한 연령 표현을 10s/30s/40s/50s/60p 로 정규화
    - 10대/20대/teens/1020 등은 10s로 묶음
    - 30/40/50/60+는 각 세그로 매핑
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()

    if any(k in s for k in [
        "10대", "10 대", "10s", "teens", "youth",
        "20대", "20 대", "20s", "1020", "10-20", "10~20",
        "10대이하", "20대이하"
    ]):
        return "10s"

    m = re.search(r"(10|20|30|40|50|60)", s)
    if m:
        n = int(m.group(0))
        if n in (10, 20):
            return "10s"
        if n == 30:
            return "30s"
        if n == 40:
            return "40s"
        if n == 50:
            return "50s"
        if n >= 60:
            return "60p"

    if "30s" in s or "30 대" in s:
        return "30s"
    if "40s" in s or "40 대" in s:
        return "40s"
    if "50s" in s or "50 대" in s:
        return "50s"
    if "60s" in s or "60 대" in s or "이상" in s or "고령" in s:
        return "60p"

    return None


def as_ratio(x) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        s = str(x).strip().replace(",", "")
        if s in ["", "None", "none", "nan", "NaN", "-999999.9", "--", "-", "na", "NA"]:
            return None
        if s.endswith("%"):
            s = s[:-1]
        val = float(s)
        if val > 1.0:       # 35 → 0.35
            val = val / 100.0
        if not (0 <= val <= 1.0):
            return None
        return round(val, 4)
    except Exception:
        return None


def _read_excel_any_header(path: Path) -> pd.DataFrame:
    last_err = None
    for header_row in (0, 1, 2):
        try:
            df = pd.read_excel(path, header=header_row)
            if isinstance(df, pd.DataFrame) and len(df.columns) > 0:
                return df
        except Exception as e:
            last_err = e
    raise last_err if last_err else ValueError(f"엑셀을 읽지 못했습니다: {path}")


def seed_media_affinity(cx, version: str = VERSION):
    df = _read_excel_any_header(DATA_DIR / "age_media_contact.xlsx")

    # --- 케이스 B: 연령 컬럼이 있는 일반 테이블이면 기존 방식 사용 ---
    age_cols = [c for c in df.columns if map_age(str(c))]
    if age_cols:
        # 채널/매체 컬럼 추정
        chan_col = None
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in ["채널", "매체", "channel", "media", "플랫폼", "platform", "service", "서비스"]):
                chan_col = c
                break
        if chan_col is None:
            chan_col = df.columns[0]

        rows = []
        for _, r in df.iterrows():
            channel = str(r.get(chan_col, "")).strip()
            if not channel or channel.lower() in ["nan", "none"]:
                continue
            channel = channel.lower().replace(" ", "_")
            for c in age_cols:
                ag = map_age(str(c))
                v = as_ratio(r[c])
                if ag and (v is not None):
                    rows.append({"age_group": ag, "channel": channel, "affinity": v})

        for row in rows:
            cx.execute(
                text("REPLACE INTO dim_age_affinity_media(version,age_group,channel,affinity) VALUES(:v,:a,:c,:s)"),
                {"v": version, "a": row["age_group"], "c": row["channel"], "s": row["affinity"]},
            )
        print(f"• media affinity upsert (table-by-age): {len(rows)} rows")
        return


    min_age_col = next((c for c in df.columns if any(k in str(c) for k in ["최소나이", "최소 나이", "min", "최소", "min_age"])), None)
    max_age_col = next((c for c in df.columns if any(k in str(c) for k in ["최대나이", "최대 나이", "max", "최대", "max_age"])), None)
    if min_age_col is None or max_age_col is None:
        print(">> DEBUG media columns:", list(map(str, df.columns)))
        raise ValueError("최소나이/최대나이 컬럼을 찾을 수 없습니다. (media wide-format)")

    media_cols = [c for c in df.columns if "매체" in str(c)]
    rate_cols  = [c for c in df.columns if ("접촉률" in str(c)) or ("주간" in str(c) and "시간" not in str(c))]

    pair_count = min(len(media_cols), len(rate_cols))
    media_cols = media_cols[:pair_count]
    rate_cols  = rate_cols[:pair_count]

    def age_to_bucket(min_age, max_age) -> str | None:
        try:
            a = float(min_age); b = float(max_age)
            mid = (a + b) / 2.0
        except Exception:
            return None
        if mid < 30:  return "10s"
        if mid < 40:  return "30s"
        if mid < 50:  return "40s"
        if mid < 60:  return "50s"
        return "60p"

    rows = []
    for _, r in df.iterrows():
        ag = age_to_bucket(r[min_age_col], r[max_age_col])
        if not ag:
            continue
        for mcol, rcol in zip(media_cols, rate_cols):
            ch = str(r.get(mcol, "")).strip()
            if not ch or ch.lower() in ["nan", "none"]:
                continue
            ch = ch.lower().replace(" ", "_")
            v = as_ratio(r.get(rcol))
            if v is None:
                continue
            rows.append({"age_group": ag, "channel": ch, "affinity": v})

    if rows:
        tmp = pd.DataFrame(rows)
        agg = tmp.groupby(["age_group", "channel"], as_index=False)["affinity"].mean()
        # DB upsert
        for _, row in agg.iterrows():
            cx.execute(
                text("REPLACE INTO dim_age_affinity_media(version,age_group,channel,affinity) VALUES(:v,:a,:c,:s)"),
                {"v": version, "a": row["age_group"], "c": row["channel"], "s": float(row["affinity"])},
            )
        print(f"• media affinity upsert (min/max age wide): {len(agg)} rows")
    else:
        print("• media wide-format에서 유효 데이터가 없습니다.")


def seed_keyword_affinity(cx, version: str = VERSION):
    df = _read_excel_any_header(DATA_DIR / "age_interest.xlsx")

    age_cols_direct = [c for c in df.columns if map_age(str(c))]
    if age_cols_direct:
        kw_col = None
        for c in df.columns:
            cl = str(c).lower()
            if any(k in cl for k in ["키워드", "keyword", "관심", "theme", "주제"]):
                kw_col = c
                break
        if kw_col is None:
            kw_col = df.columns[0]

        rows = []
        for _, r in df.iterrows():
            kw = str(r.get(kw_col, "")).strip()
            if not kw or kw.lower() in ["nan", "none"]:
                continue
            keyword = kw.replace(" ", "")
            for c in age_cols_direct:
                ag = map_age(str(c))
                v = as_ratio(r[c])
                if ag and (v is not None):
                    rows.append({"age_group": ag, "keyword": keyword, "affinity": v})

        for row in rows:
            cx.execute(
                text("REPLACE INTO dim_age_affinity_keyword(version,age_group,keyword,affinity) "
                     "VALUES(:v,:a,:k,:s)"),
                {"v": version, "a": row["age_group"], "k": row["keyword"], "s": row["affinity"]}
            )
        print(f"• keyword affinity upsert (table-by-age): {len(rows)} rows")
        return

    min_age_col = next((c for c in df.columns
                        if any(k in str(c) for k in ["최소나이", "최소 나이", "min", "최소", "min_age"])), None)
    max_age_col = next((c for c in df.columns
                        if any(k in str(c) for k in ["최대나이", "최대 나이", "max", "최대", "max_age"])), None)
    if min_age_col is None or max_age_col is None:
        print(">> DEBUG interest columns:", list(map(str, df.columns)))
        raise ValueError("연령대 컬럼을 찾을 수 없습니다. (interest wide-format, min/max 없음)")

    ratio_cols = [c for c in df.columns if str(c).endswith("비율")]
    pairs = []
    for rc in ratio_cols:
        base = str(rc).removesuffix("비율").rstrip("_")
        # base와 rc 모두 실제 컬럼으로 존재해야 함
        base_col = next((c for c in df.columns if str(c) == base), None)
        if base_col is not None:
            pairs.append((base_col, rc))

    direct_numeric_cols = [c for c in df.columns
                           if any(s in str(c) for s in ["_전체", "_개인"])
                           and c not in [rc for _, rc in pairs]]
    for c in direct_numeric_cols:
        pairs.append((c, c))

    if not pairs:
        print(">> DEBUG interest columns:", list(map(str, df.columns)))
        raise ValueError("연령대 컬럼을 찾을 수 없습니다. (interest wide-format, 페어 없음)")

    def age_to_bucket(min_age, max_age) -> str | None:
        try:
            a = float(min_age); b = float(max_age)
            mid = (a + b) / 2.0
        except Exception:
            return None
        if mid < 30:  return "10s"
        if mid < 40:  return "30s"
        if mid < 50:  return "40s"
        if mid < 60:  return "50s"
        return "60p"

    rows = []
    for _, r in df.iterrows():
        ag = age_to_bucket(r[min_age_col], r[max_age_col])
        if not ag:
            continue
        for base_col, ratio_col in pairs:
            if base_col == ratio_col:

                keyword = str(base_col).strip()
                v = as_ratio(r.get(ratio_col))
            else:

                kw_text = str(r.get(base_col, "")).strip()
                if not kw_text or kw_text.lower() in ["nan", "none"]:
                    continue
                keyword = kw_text.replace(" ", "")
                v = as_ratio(r.get(ratio_col))
            if v is None:
                continue
            rows.append({"age_group": ag, "keyword": keyword, "affinity": v})

    if rows:
        tmp = pd.DataFrame(rows)
        agg = tmp.groupby(["age_group", "keyword"], as_index=False)["affinity"].mean()
        for _, row in agg.iterrows():
            cx.execute(
                text("REPLACE INTO dim_age_affinity_keyword(version,age_group,keyword,affinity) "
                     "VALUES(:v,:a,:k,:s)"),
                {"v": version, "a": row["age_group"], "k": row["keyword"], "s": float(row["affinity"])}
            )
        print(f"• keyword affinity upsert (min/max age wide): {len(agg)} rows")
    else:
        print("• interest wide-format에서 유효 데이터가 없습니다.")



def seed_video_behavior_as_media(cx, version: str = VERSION):
    path = DATA_DIR / "online_video_behavior.xlsx"
    try:
        df = _read_excel_any_header(path)
    except FileNotFoundError:
        print("• online_video_behavior.xlsx 없음 → 스킵")
        return

    plat_col = None
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["플랫폼", "platform", "채널", "service", "서비스"]):
            plat_col = c
            break
    if plat_col is None:
        plat_col = df.columns[0]

    age_cols = [c for c in df.columns if map_age(str(c))]
    if not age_cols and len(df) > 0 and any(map_age(str(x)) for x in df.iloc[:, 0].astype(str)):
        df = df.set_index(plat_col).T.reset_index()
        plat_col = df.columns[0]
        age_cols = [c for c in df.columns if map_age(str(c))]
    if not age_cols:
        print("• 동영상 파일에서 연령대 컬럼을 못 찾음 → 스킵")
        return

    rows = []
    for _, r in df.iterrows():
        channel = str(r.get(plat_col, "")).strip()
        if not channel or channel.lower() in ["nan", "none"]:
            continue
        channel = channel.lower().replace(" ", "_")
        for c in age_cols:
            ag = map_age(str(c))
            v = as_ratio(r[c])
            if ag and (v is not None):
                rows.append({"age_group": ag, "channel": channel, "affinity": v})

    for row in rows:
        cx.execute(
            text(
                "REPLACE INTO dim_age_affinity_media(version,age_group,channel,affinity) "
                "VALUES(:v,:a,:c,:s)"
            ),
            {"v": version, "a": row["age_group"], "c": row["channel"], "s": row["affinity"]},
        )
    print(f"• video->media affinity upsert: {len(rows)} rows")


def run(version: str = VERSION):
    eng = get_engine()
    with eng.begin() as cx:
        seed_media_affinity(cx, version)
        seed_keyword_affinity(cx, version)
        seed_video_behavior_as_media(cx, version)



if __name__ == "__main__":
    run()
