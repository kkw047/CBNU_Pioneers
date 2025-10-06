from __future__ import annotations
import sys, pathlib, json, argparse, shutil, subprocess, datetime, re
import pandas as pd
from sqlalchemy import text

# ---------- import plumbing ----------
HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
TASKS = HERE.parent
for p in (str(ROOT), str(TASKS)):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from dbconnect import get_engine, exec_sql
except ModuleNotFoundError:
    from tasks.dbconnect import get_engine, exec_sql

# ---------- 설정 ----------
AFFINITY_VERSION = "v1_2025Q3"
TOP_MEDIA_DEFAULT = 5
TOP_KW_DEFAULT = 10

# (신 스키마) 프롬프트 저장 테이블
DDL_PROMPT_BUNDLE = r"""
CREATE TABLE IF NOT EXISTS mat_prompt_bundle (
  mct_id   VARCHAR(64),
  ym       CHAR(6),
  version  VARCHAR(32),
  bundle   JSON,
  PRIMARY KEY (mct_id, ym, version)
);
"""

# 최신 k (있으면 사용)
SQL_LATEST_K = "SELECT k FROM mat_store_cluster ORDER BY created_at DESC LIMIT 1"

# 폴백용 프로파일/규칙/추천
SQL_PROFILE = r"""
SELECT cluster_id, k, feature, mean, std, n
FROM mat_cluster_profile
WHERE k=:k
"""
SQL_BEST_RULE = r"""
SELECT rule_id, cluster_id, rule_text, rule_precision, support, depth
FROM mat_tree_rules
WHERE k=:k AND cluster_id=:cid
ORDER BY rule_precision DESC, support DESC, depth ASC, rule_id ASC
LIMIT 1
"""
SQL_MEDIA = r"""
SELECT channel, affinity
FROM dim_age_affinity_media
WHERE version=:v AND age_group=:ag
ORDER BY affinity DESC
LIMIT :topn
"""
SQL_KW = r"""
SELECT keyword, affinity
FROM dim_age_affinity_keyword
WHERE version=:v AND age_group=:ag
ORDER BY affinity DESC
LIMIT :topn
"""

# per-store / group 조회
SQL_STORE_ONE = r"""
SELECT f.mct_id, f.ym, f.sector, f.biz_zone,
       f.age_10s,f.age_30s,f.age_40s,f.age_50s,f.age_60p,
       f.aud_new,f.aud_reu,f.aud_res,f.aud_work,f.aud_flow,
       f.cancel_high,f.delivery_high,
       c.cluster_id, c.k
FROM vw_store_features f
JOIN mat_store_cluster c ON c.mct_id=f.mct_id
WHERE f.mct_id=:m
ORDER BY c.created_at DESC
LIMIT 1
"""
SQL_STORE_GROUP_MEAN = r"""
SELECT 
  AVG(f.age_10s)       AS age_10s,
  AVG(f.age_30s)       AS age_30s,
  AVG(f.age_40s)       AS age_40s,
  AVG(f.age_50s)       AS age_50s,
  AVG(f.age_60p)       AS age_60p,
  AVG(f.aud_new)       AS aud_new,
  AVG(f.aud_reu)       AS aud_reu,
  AVG(f.aud_res)       AS aud_res,
  AVG(f.aud_work)      AS aud_work,
  AVG(f.aud_flow)      AS aud_flow,
  AVG(f.cancel_high)   AS cancel_high,
  AVG(f.delivery_high) AS delivery_high,
  (SELECT k FROM mat_store_cluster ORDER BY created_at DESC LIMIT 1) AS k
FROM vw_store_features f
JOIN stg_store s ON s.ENCODED_MCT = f.mct_id
WHERE (:area   IS NULL OR f.biz_zone LIKE CONCAT('%',:area,'%') OR s.MCT_SIGUNGU_NM LIKE CONCAT('%',:area,'%'))
  AND (:sector IS NULL OR f.sector   LIKE CONCAT('%',:sector,'%'))
"""
SQL_STORE_GROUP_REP_CLUSTER = r"""
SELECT c.cluster_id, COUNT(*) AS cnt
FROM mat_store_cluster c
JOIN vw_store_features f ON f.mct_id=c.mct_id
JOIN stg_store s ON s.ENCODED_MCT = f.mct_id
WHERE (:area   IS NULL OR f.biz_zone LIKE CONCAT('%',:area,'%') OR s.MCT_SIGUNGU_NM LIKE CONCAT('%',:area,'%'))
  AND (:sector IS NULL OR f.sector   LIKE CONCAT('%',:sector,'%'))
GROUP BY c.cluster_id
ORDER BY cnt DESC
LIMIT 1
"""
SQL_VOCAB = r"""
SELECT DISTINCT HPSN_MCT_ZCD_NM AS sector, NULL AS area
FROM stg_store WHERE HPSN_MCT_ZCD_NM IS NOT NULL
UNION ALL
SELECT DISTINCT NULL AS sector, MCT_SIGUNGU_NM AS area
FROM stg_store WHERE MCT_SIGUNGU_NM IS NOT NULL
UNION ALL
SELECT DISTINCT NULL AS sector, HPSN_MCT_BZN_CD_NM AS area
FROM stg_store WHERE HPSN_MCT_BZN_CD_NM IS NOT NULL
"""

# ---------- 유틸 ----------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w가-힣\s\*]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def mask_shop_like_token(s: str) -> str|None:
    stars = [t for t in s.split() if "*" in t]
    return (" ".join(stars)) if stars else None

def load_vocab(eng):
    df = pd.read_sql(SQL_VOCAB, eng)
    sectors = set([x for x in df["sector"].dropna().unique()])
    areas   = set([x for x in df["area"].dropna().unique()])
    sectors_sorted = sorted(sectors, key=len, reverse=True)
    areas_sorted   = sorted(areas, key=len, reverse=True)
    return sectors, areas, sectors_sorted, areas_sorted

def longest_match(text: str, candidates_sorted: list[str]) -> str|None:
    for cand in candidates_sorted:
        if cand and cand in text:
            return cand
    return None

def dominant_age_bucket(stats: dict[str, float]) -> str:
    keys = ["10s","30s","40s","50s","60p"]
    vals = [stats.get(f"age_{k}", 0.0) for k in keys]
    idx = max(range(len(vals)), key=lambda i: vals[i])
    return keys[idx]

# ---------- 프롬프트 빌더 ----------
class PromptBuilder:
    def build_title(self, cid: int, dom_age: str, ctx_title: str|None=None) -> str:
        base = f"[Cluster {cid}] {dom_age} 중심 고객 타겟 브리프"
        return (f"{ctx_title} | {base}" if ctx_title else base)[:255]

    def build_user_prompt(
        self, k: int, cid: int, dom_age: str, prof: dict[str, float],
        rule_text: str | None, media_list, kw_list,
        user_input: str = "", context_lines: list[str] | None = None,
    ) -> str:
        age_line = f"- 연령 믹스: 10s={prof.get('age_10s',0):.2f}, 30s={prof.get('age_30s',0):.2f}, 40s={prof.get('age_40s',0):.2f}, 50s={prof.get('age_50s',0):.2f}, 60p={prof.get('age_60p',0):.2f}"
        aud_line = f"- 오디언스: NEW={prof.get('aud_new',0):.2f}, REU={prof.get('aud_reu',0):.2f}, RES={prof.get('aud_res',0):.2f}, WORK={prof.get('aud_work',0):.2f}, FLOW={prof.get('aud_flow',0):.2f}"
        flag_line = f"- 플래그: cancel_high={prof.get('cancel_high',0):.2f}, delivery_high={prof.get('delivery_high',0):.2f}"
        media_line = ", ".join([f"{c}({a:.2f})" for c,a in media_list]) if media_list else "없음"
        kw_line    = ", ".join([f"{w}({a:.2f})" for w,a in kw_list]) if kw_list else "없음"

        ctx_block = ""
        if context_lines:
            ctx_block = "\n[지역/업종/점포 컨텍스트]\n- " + "\n- ".join(context_lines)

        user_block = ""
        if user_input:
            user_block = f'\n[사용자 입력(상황/위치)]\n- "{user_input}"'

        return f"""당신은 마케팅 카피라이터이자 미디어 플래너입니다.
아래 타겟 브리프에 맞춰 ‘한글’로 톤 앤 매너와 메시지 키를 제안하고, 채널/키워드 추천을 함께 제시하세요.

[클러스터]
- k: {k}
- Cluster ID: {cid}
- 지배 연령대: {dom_age}
- 연령 믹스 / 오디언스 / 플래그
{age_line}
{aud_line}
{flag_line}

[대표 규칙(의사결정트리 경로)]
- {rule_text or '규칙 미확인'}

[추천 채널 Top {len(media_list)}]
- {media_line}

[추천 키워드 Top {len(kw_list)}]
- {kw_line}{user_block}{ctx_block}

[출력 형식]
1) 한 줄 콘셉트
2) 핵심 메시지 (3개, 각 20자 내외)
3) R 그래프 요약 해석 (연령 우세/열세, 오디언스 강·약, cancel/delivery 시사점)
4) 채널 운영 가이드 (Top N 채널별 KPI/포인트 1~2줄)
5) 검색/콘텐츠 키워드 (10개, 쉼표구분)
6) KPI 가설 & 실험 아이디어 (1차 KPI + A/B 2안)
7) 대안 타깃(2개)와 각색 포인트
8) 리스크 & 대응
9) 실행 체크리스트
"""

# ✅ 레지스트리 추가 (여기가 빠져 있어서 NameError 났음)
PROMPT_BUILDERS = {"propensity": PromptBuilder()}

# ---------- R 자산 ----------
def ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)

def write_cluster_assets(out_dir: pathlib.Path, k: int, cid: int, prof: dict[str, float]):
    ensure_dir(out_dir)
    age_csv = out_dir / f"cluster_{k}_{cid}_age_mix.csv"
    aud_csv = out_dir / f"cluster_{k}_{cid}_audience.csv"
    pd.DataFrame([
        {"bucket":"10s", "value": float(prof.get("age_10s",0))},
        {"bucket":"30s", "value": float(prof.get("age_30s",0))},
        {"bucket":"40s", "value": float(prof.get("age_40s",0))},
        {"bucket":"50s", "value": float(prof.get("age_50s",0))},
        {"bucket":"60p", "value": float(prof.get("age_60p",0))}
    ]).to_csv(age_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"metric":"NEW",  "value": float(prof.get("aud_new",0))},
        {"metric":"REU",  "value": float(prof.get("aud_reu",0))},
        {"metric":"RES",  "value": float(prof.get("aud_res",0))},
        {"metric":"WORK", "value": float(prof.get("aud_work",0))},
        {"metric":"FLOW", "value": float(prof.get("aud_flow",0))}
    ]).to_csv(aud_csv, index=False, encoding="utf-8-sig")

    r_script = out_dir / f"cluster_{k}_{cid}_plots.R"
    png1 = out_dir / f"cluster_{k}_{cid}_age_mix.png"
    png2 = out_dir / f"cluster_{k}_{cid}_audience.png"
    r_code = f"""
args <- commandArgs(trailingOnly=TRUE)
age_path <- "{age_csv.as_posix()}"
aud_path <- "{aud_csv.as_posix()}"
png_age <- "{png1.as_posix()}"
png_aud <- "{png2.as_posix()}"
suppressWarnings(suppressMessages({{
  if (!require(ggplot2)) install.packages("ggplot2", repos="https://cloud.r-project.org")
  library(ggplot2)
}}))
age_df <- read.csv(age_path, encoding="UTF-8")
p1 <- ggplot(age_df, aes(x=bucket, y=value)) + geom_bar(stat="identity") +
  labs(title=paste("Cluster {cid} @ k={k} - Age Mix"), x="Age Bucket", y="Share") + theme_minimal()
ggsave(filename=png_age, plot=p1, width=6, height=4, dpi=150)
aud_df <- read.csv(aud_path, encoding="UTF-8")
p2 <- ggplot(aud_df, aes(x=metric, y=value)) + geom_bar(stat="identity") +
  labs(title=paste("Cluster {cid} @ k={k} - Audience"), x="Audience", y="Score") + theme_minimal()
ggsave(filename=png_aud, plot=p2, width=6, height=4, dpi=150)
"""
    r_script.write_text(r_code, encoding="utf-8")
    rscript = shutil.which("Rscript")
    if rscript:
        try:
            subprocess.run([rscript, r_script.as_posix()], check=True)
        except subprocess.CalledProcessError:
            pass

# ---------- 저장 ----------
def save_bundle(cx, mct_id: str, ym: str, version: str, payload: dict):
    cx.execute(
        text("""REPLACE INTO mat_prompt_bundle (mct_id, ym, version, bundle)
                VALUES (:m, :ym, :v, :b)"""),
        {"m": mct_id, "ym": ym, "v": version, "b": json.dumps(payload, ensure_ascii=False)},
    )

# ---------- 메인 ----------
def generate_propensity_prompts(k: int | None, top_media: int, top_kw: int, make_plots: bool, assets_dir: str,
                                user_input: str, mct_id: str|None, area: str|None, sector: str|None,
                                per_store: bool):
    eng = get_engine()
    exec_sql(DDL_PROMPT_BUNDLE)

    if k is None:
        with eng.begin() as cx:
            k = cx.execute(text(SQL_LATEST_K)).scalar()
        if k is None:
            print("mat_store_cluster 에 k가 없습니다. cluster_segments 먼저 실행하세요.")
            return
    k = int(k)

    builder = PROMPT_BUILDERS["propensity"]

    # 0) 사용자 입력 해석
    user_input = user_input or ""
    text_norm = normalize_text(user_input)
    masked_name = mask_shop_like_token(text_norm)

    if per_store and not mct_id:
        sectors, areas, sectors_sorted, areas_sorted = load_vocab(eng)
        if not area:
            area = longest_match(text_norm, areas_sorted)
        if not sector:
            sector = longest_match(text_norm, sectors_sorted)

    created = 0

    with eng.begin() as cx:
        # (1) 정확 mct_id
        if per_store and mct_id:
            row = cx.execute(text(SQL_STORE_ONE), {"m": mct_id}).mappings().first()
            if row:
                prof = {k2: float(row[k2]) for k2 in [
                    "age_10s","age_30s","age_40s","age_50s","age_60p",
                    "aud_new","aud_reu","aud_res","aud_work","aud_flow",
                    "cancel_high","delivery_high"
                ]}
                cid = int(row["cluster_id"]); k = int(row["k"])
                dom_age = dominant_age_bucket(prof)
                rule_row = cx.execute(text(SQL_BEST_RULE), {"k": k, "cid": cid}).mappings().first()
                rule_text = rule_row["rule_text"] if rule_row else None
                media_rows = cx.execute(text(SQL_MEDIA), {"v": AFFINITY_VERSION, "ag": dom_age, "topn": int(top_media)}).fetchall()
                kw_rows    = cx.execute(text(SQL_KW),    {"v": AFFINITY_VERSION, "ag": dom_age, "topn": int(top_kw)}).fetchall()
                media_list = [(r[0], float(r[1])) for r in media_rows]
                kw_list    = [(r[0], float(r[1])) for r in kw_rows]
                ctx_lines = [f"biz_zone={row['biz_zone']}", f"sector={row['sector']}", f"mct_id={row['mct_id']}"]
                if masked_name: ctx_lines.append(f"masked_store_name={masked_name}")
                title = builder.build_title(cid, dom_age, ctx_title=f"{row['biz_zone']}/{row['sector']}")
                user_prompt = builder.build_user_prompt(k, cid, dom_age, prof, rule_text, media_list, kw_list, user_input=user_input, context_lines=ctx_lines)
                bundle = {
                    "k": k, "cluster_id": cid, "title": title, "system_prompt": None, "user_prompt": user_prompt,
                    "rule_text": rule_text, "dom_age": dom_age, "profile": prof,
                    "rec_channels": [{"channel": c, "affinity": a} for c,a in media_list],
                    "rec_keywords": [{"keyword": w, "affinity": a} for w,a in kw_list],
                    "context_lines": ctx_lines, "user_input": user_input
                }
                save_bundle(cx, mct_id=row["mct_id"], ym=row["ym"], version=AFFINITY_VERSION, payload=bundle)
                created += 1
                if make_plots:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_dir = pathlib.Path(assets_dir or "assets") / "propensity" / ts
                    write_cluster_assets(out_dir, k, cid, prof)
                print(f"✅ per-store prompt saved (mct_id: {mct_id})")
                print(f"• target k = {k}")
                return

        # (2) 지역×업종 평균
        if per_store and (area or sector):
            g = cx.execute(text(SQL_STORE_GROUP_MEAN), {"area": area, "sector": sector}).mappings().first()
            if g and g["k"] is not None:
                prof = {
                    "age_10s": float(g["age_10s"] or 0), "age_30s": float(g["age_30s"] or 0),
                    "age_40s": float(g["age_40s"] or 0), "age_50s": float(g["age_50s"] or 0),
                    "age_60p": float(g["age_60p"] or 0),
                    "aud_new": float(g["aud_new"] or 0), "aud_reu": float(g["aud_reu"] or 0),
                    "aud_res": float(g["aud_res"] or 0), "aud_work": float(g["aud_work"] or 0),
                    "aud_flow": float(g["aud_flow"] or 0),
                    "cancel_high": float(g["cancel_high"] or 0),
                    "delivery_high": float(g["delivery_high"] or 0),
                }
                k = int(g["k"])
                cr = cx.execute(text(SQL_STORE_GROUP_REP_CLUSTER), {"area": area, "sector": sector}).mappings().first()
                cid = int(cr["cluster_id"]) if cr else 0
                dom_age = dominant_age_bucket(prof)
                rule_row = cx.execute(text(SQL_BEST_RULE), {"k": k, "cid": cid}).mappings().first()
                rule_text = rule_row["rule_text"] if rule_row else None
                media_rows = cx.execute(text(SQL_MEDIA), {"v": AFFINITY_VERSION, "ag": dom_age, "topn": int(top_media)}).fetchall()
                kw_rows    = cx.execute(text(SQL_KW),    {"v": AFFINITY_VERSION, "ag": dom_age, "topn": int(top_kw)}).fetchall()
                media_list = [(r[0], float(r[1])) for r in media_rows]
                kw_list    = [(r[0], float(r[1])) for r in kw_rows]
                ctx_title = f"{(area or '').strip()}/{(sector or '').strip()}".strip("/")
                ctx_lines = []
                if area:   ctx_lines.append(f"area={area}")
                if sector: ctx_lines.append(f"sector={sector}")
                if masked_name: ctx_lines.append(f"masked_store_name={masked_name}")
                title = builder.build_title(cid, dom_age, ctx_title=ctx_title)
                user_prompt = builder.build_user_prompt(k, cid, dom_age, prof, rule_text, media_list, kw_list, user_input=user_input, context_lines=ctx_lines)
                bundle = {
                    "k": k, "cluster_id": cid, "title": title, "system_prompt": None, "user_prompt": user_prompt,
                    "rule_text": rule_text, "dom_age": dom_age, "profile": prof,
                    "rec_channels": [{"channel": c, "affinity": a} for c,a in media_list],
                    "rec_keywords": [{"keyword": w, "affinity": a} for w,a in kw_list],
                    "context_lines": ctx_lines, "user_input": user_input
                }
                synthetic_id = f"GROUP::{(area or 'ALL')}/{(sector or 'ALL')}"
                ym_now = datetime.datetime.now().strftime("%Y%m")
                save_bundle(cx, mct_id=synthetic_id, ym=ym_now, version=AFFINITY_VERSION, payload=bundle)
                created += 1
                if make_plots:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_dir = pathlib.Path(assets_dir or "assets") / "propensity" / ts
                    write_cluster_assets(out_dir, k, cid, prof)
                print(f"✅ per-store prompt saved (group: {ctx_title})")
                print(f"• target k = {k}")
                return

    # (3) 폴백: 클러스터 평균
    prof_df = pd.read_sql(SQL_PROFILE, eng, params={"k": k})
    if prof_df.empty:
        print("mat_cluster_profile 에 데이터가 없습니다. cluster_segments 먼저 실행하세요.")
        return
    prof_pivot = prof_df.pivot_table(index=["cluster_id"], columns="feature", values="mean", aggfunc="first").reset_index()
    with eng.begin() as cx:
        ym_now = datetime.datetime.now().strftime("%Y%m")
        for _, row in prof_pivot.iterrows():
            cid = int(row["cluster_id"])
            prof = {col: float(row[col]) for col in prof_pivot.columns if col != "cluster_id"}
            dom_age = dominant_age_bucket(prof)
            rule_row = cx.execute(text(SQL_BEST_RULE), {"k": k, "cid": cid}).mappings().first()
            rule_text = rule_row["rule_text"] if rule_row else None
            media_rows = cx.execute(text(SQL_MEDIA), {"v": AFFINITY_VERSION, "ag": dom_age, "topn": int(top_media)}).fetchall()
            kw_rows    = cx.execute(text(SQL_KW),    {"v": AFFINITY_VERSION, "ag": dom_age, "topn": int(top_kw)}).fetchall()
            media_list = [(r[0], float(r[1])) for r in media_rows]
            kw_list    = [(r[0], float(r[1])) for r in kw_rows]
            ctx_lines = []
            if area:   ctx_lines.append(f"area={area}")
            if sector: ctx_lines.append(f"sector={sector}")
            if masked_name: ctx_lines.append(f"masked_store_name={masked_name}")
            title = builder.build_title(cid, dom_age)
            user_prompt = builder.build_user_prompt(k, cid, dom_age, prof, rule_text, media_list, kw_list, user_input=user_input, context_lines=ctx_lines)
            bundle = {
                "k": k, "cluster_id": cid, "title": title, "system_prompt": None, "user_prompt": user_prompt,
                "rule_text": rule_text, "dom_age": dom_age, "profile": prof,
                "rec_channels": [{"channel": c, "affinity": a} for c,a in media_list],
                "rec_keywords": [{"keyword": w, "affinity": a} for w,a in kw_list],
                "context_lines": ctx_lines, "user_input": user_input
            }
            synthetic_id = f"CLUSTER::{cid}"
            save_bundle(cx, mct_id=synthetic_id, ym=ym_now, version=AFFINITY_VERSION, payload=bundle)
    print(f"✅ prompts saved into mat_prompt_bundle (k={k})")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Generate propensity-based prompts (store/group/cluster) + optional R plots.")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--top-media", type=int, default=TOP_MEDIA_DEFAULT)
    ap.add_argument("--top-kw", type=int, default=TOP_KW_DEFAULT)
    ap.add_argument("--make-plots", action="store_true")
    ap.add_argument("--assets-dir", type=str, default="assets")
    ap.add_argument("--user-input", type=str, default="")
    ap.add_argument("--mct-id", type=str, default=None)
    ap.add_argument("--area", type=str, default=None)
    ap.add_argument("--sector", type=str, default=None)
    ap.add_argument("--per-store", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    generate_propensity_prompts(
        k=args.k, top_media=args.top_media, top_kw=args.top_kw,
        make_plots=args.make_plots, assets_dir=args.assets_dir,
        user_input=args.user_input, mct_id=args.mct_id,
        area=args.area, sector=args.sector, per_store=args.per_store,
    )

if __name__ == "__main__":
    main()
