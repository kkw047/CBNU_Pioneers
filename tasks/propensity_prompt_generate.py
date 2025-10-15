
# tasks/propensity_prompt_generate.py
from __future__ import annotations
import sys, pathlib, json, argparse, shutil, subprocess, datetime, re
import pandas as pd
from sqlalchemy import text

# ---- import plumbing
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

# ---- 설정
AFFINITY_VERSION = "v1_2025Q3"
TOP_MEDIA_DEFAULT = 5
TOP_KW_DEFAULT = 10

DDL_PROMPT_BUNDLE = r"""
CREATE TABLE IF NOT EXISTS mat_prompt_bundle (
  mct_id   VARCHAR(64),
  ym       CHAR(6),
  version  VARCHAR(32),
  bundle   JSON,
  PRIMARY KEY (mct_id, ym, version)
);
"""

SQL_LATEST_K = "SELECT k FROM mat_store_cluster ORDER BY created_at DESC LIMIT 1"
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
  AVG(f.age_10s) AS age_10s, AVG(f.age_30s) AS age_30s, AVG(f.age_40s) AS age_40s,
  AVG(f.age_50s) AS age_50s, AVG(f.age_60p) AS age_60p,
  AVG(f.aud_new) AS aud_new, AVG(f.aud_reu) AS aud_reu, AVG(f.aud_res) AS aud_res,
  AVG(f.aud_work) AS aud_work, AVG(f.aud_flow) AS aud_flow,
  AVG(f.cancel_high) AS cancel_high, AVG(f.delivery_high) AS delivery_high,
  (SELECT k FROM mat_store_cluster ORDER BY created_at DESC LIMIT 1) AS k
FROM vw_store_features f
JOIN stg_store s ON s.ENCODED_MCT = f.mct_id
WHERE (:area IS NULL OR f.biz_zone LIKE CONCAT('%',:area,'%') OR s.MCT_SIGUNGU_NM LIKE CONCAT('%',:area,'%'))
  AND (:sector IS NULL OR f.sector LIKE CONCAT('%',:sector,'%'))
"""
SQL_STORE_GROUP_REP_CLUSTER = r"""
SELECT c.cluster_id, COUNT(*) AS cnt
FROM mat_store_cluster c
JOIN vw_store_features f ON f.mct_id=c.mct_id
JOIN stg_store s ON s.ENCODED_MCT = f.mct_id
WHERE (:area IS NULL OR f.biz_zone LIKE CONCAT('%',:area,'%') OR s.MCT_SIGUNGU_NM LIKE CONCAT('%',:area,'%'))
  AND (:sector IS NULL OR f.sector LIKE CONCAT('%',:sector,'%'))
GROUP BY c.cluster_id
ORDER BY cnt DESC
LIMIT 1
"""
SQL_VOCAB = r"""
SELECT DISTINCT HPSN_MCT_ZCD_NM AS sector, NULL AS area FROM stg_store WHERE HPSN_MCT_ZCD_NM IS NOT NULL
UNION ALL SELECT DISTINCT NULL AS sector, MCT_SIGUNGU_NM AS area FROM stg_store WHERE MCT_SIGUNGU_NM IS NOT NULL
UNION ALL SELECT DISTINCT NULL AS sector, HPSN_MCT_BZN_CD_NM AS area FROM stg_store WHERE HPSN_MCT_BZN_CD_NM IS NOT NULL
"""

# ---- 유틸
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
    sectors = set(df["sector"].dropna().unique())
    areas   = set(df["area"].dropna().unique())
    return sectors, areas, sorted(sectors, key=len, reverse=True), sorted(areas, key=len, reverse=True)

def longest_match(text: str, cands: list[str]) -> str|None:
    for x in cands:
        if x and x in text:
            return x
    return None

def dominant_age_bucket(stats: dict[str, float]) -> str:
    keys = ["10s","30s","40s","50s","60p"]
    vals = [stats.get(f"age_{k}", 0.0) for k in keys]
    return keys[max(range(len(vals)), key=lambda i: vals[i])]

# ---- 프롬프트 빌더(요약판)
class PromptBuilder:
    def build_title(self, cid: int, dom_age: str, ctx_title: str|None=None) -> str:
        base = f"[Cluster {cid}] {dom_age} 중심 고객 타겟 브리프"
        return (f"{ctx_title} | {base}" if ctx_title else base)[:255]

    def build_user_prompt(
        self, k: int, cid: int, dom_age: str, prof: dict[str, float],
        rule_text: str|None, media_list, kw_list,
        user_input: str = "", context_lines: list[str]|None=None,
    ) -> str:
        age_line = f"- 연령 믹스: 10s={prof.get('age_10s',0):.2f}, 30s={prof.get('age_30s',0):.2f}, 40s={prof.get('age_40s',0):.2f}, 50s={prof.get('age_50s',0):.2f}, 60p={prof.get('age_60p',0):.2f}"
        aud_line = f"- 오디언스: NEW={prof.get('aud_new',0):.2f}, REU={prof.get('aud_reu',0):.2f}, RES={prof.get('aud_res',0):.2f}, WORK={prof.get('aud_work',0):.2f}, FLOW={prof.get('aud_flow',0):.2f}"
        flag_line= f"- 플래그: cancel_high={prof.get('cancel_high',0):.2f}, delivery_high={prof.get('delivery_high',0):.2f}"
        media_line = ", ".join([f"{c}({a:.2f})" for c,a in media_list]) if media_list else "없음"
        kw_line    = ", ".join([f"{w}({a:.2f})" for w,a in kw_list]) if kw_list else "없음"

        ctx_block = "\n[지역/업종/점포 컨텍스트]\n- " + "\n- ".join(context_lines) if context_lines else ""
        user_block = f'\n[사용자 입력]\n- "{user_input}"' if user_input else ""

        return f"""당신은 대한민국 소상공인을 위한 최고의 마케팅 컨설턴트입니다.
        제공된 데이터 브리프를 바탕으로, Markdown을 활용하여 구조화된 최종 보고서를 작성합니다.

        [데이터 브리프]
        - 클러스터 ID: {cid} (k={k})
        - 지배 연령대: {dom_age}
        {age_line}
        {aud_line}
        {flag_line}
        - 대표 규칙: {rule_text or '규칙 미확인'}
        - 추천 채널 Raw Data (Top {len(media_list)}): {media_line}
        - 추천 키워드 Raw Data (Top {len(kw_list)}): {kw_line}{user_block}{ctx_block}

        [작성 원칙]
        - 보고서 시작 부분에, 사용자 입력을 바탕으로 어떤 가게에 대한 분석인지 언급하며 자연스러운 인사말을 한두 문장 작성합니다.
        - Markdown을 사용하여 보고서의 구조와 가독성을 높입니다.
        - 최상위 섹션 제목은 '###' (H3 헤더)를 사용합니다.
        - 중간 항목은 '*' (별표) 불릿 포인트를 사용하고, 제목 부분은 굵게(**) 표시합니다.
        - 항목의 제목 줄 바로 다음 줄부터 내용을 시작하며, 중간에 불필요한 공백 줄을 삽입하지 않습니다.
        - 세부 설명이나 하위 목록은 중간 항목 아래에 두 칸 들여쓰고 '-' (대시)를 사용합니다.
        - 사장님의 눈높이에 맞춰 쉽고 명확하게 설명하며, 과장되거나 단정적인 표현은 피합니다.

        [출력 형식]
        사장님, 요청하신 '{user_input or "가게"}'에 대한 데이터 기반 마케팅 전략 분석을 완료했습니다.  
        아래 보고서를 통해 새로운 기회를 발견해 보세요!
        
        ---

        ### 우리 가게 현황 요약
        * **주요 고객 분석:** (그래프와 연령 믹스 데이터를 해석하여 어떤 연령대의 고객이 왜 많이 오는지 설명하고, 가장 비중이 높은 고객층과 낮은 고객층을 비교하여 특징을 설명)
        * **고객 행동 패턴:** (오디언스 데이터와 대표 규칙을 바탕으로 신규/재방문/직장인 등 고객들의 특징적인 행동을 요약)
        * **강점 및 기회:**
          - **강점:** (데이터를 종합하여 현재 가게의 강점은 무엇인지 설명)
          - **기회:** (데이터를 바탕으로 어떤 새로운 기회를 포착할 수 있을지 제안)
        
        ### 핵심 마케팅 전략 제안
        * **한 줄 컨셉:** (가게의 강점과 타겟 고객을 아우르는 기억하기 쉬운 한 줄 컨셉)
        * **핵심 메시지:** (고객의 마음을 사로잡을 핵심 메시지 3개를 `-` 불릿을 사용하여 목록으로 작성)
        * **기대 효과 및 목표 KPI:**
          - (이 전략을 실행했을 때 예상되는 긍정적인 변화를 설명)
          - **목표 KPI:** (예: 예약 취소율 30% 감소, 재방문 고객 15% 증가 등 구체적인 수치 제시)
        * **주요 리스크 및 대응 방안:**
          - **리스크:** (데이터에서 발견되는 잠재적 위험 요소를 지적하고, 이것이 의미할 수 있는 운영상의 문제를 설명)
          - **대응 방안:** (리스크를 관리하고 고객 경험을 개선하기 위한 구체적인 액션 아이템 제안)
        
        ### 추천 마케팅 채널 및 키워드
        * **추천 채널:** (가장 적합한 채널 3개를 선정하고, 각 채널명과 추천 이유를 `-` 불릿을 사용하여 번호 없는 목록으로 작성)
        * **추천 키워드:** (가장 효과적인 키워드 10개를 쉼표로 구분하여 한 줄로 제시)
        * **확장을 위한 대안 타겟:**
          - **대안 타겟 1 (예: 10~30대 젊은 층):**
            - **각색 포인트:** (해당 타겟을 공략하기 위해 기존 전략을 어떻게 변형해야 할지 설명. 예: SNS 이벤트, 비주얼 강조 메뉴 등)
          - **대안 타겟 2 (예: 직장인):**
            - **각색 포인트:** (설명. 예: 점심 특선, 빠른 회전율을 위한 시스템 개선 등)
        
        ---
        더 구체적인 실행 계획이 필요하신가요? 원하신다면 **'마케팅 실행 체크리스트'**를 생성해 드릴 수 있습니다.
        """

PROMPT_BUILDERS = {"propensity": PromptBuilder()}

# ---- R 자산(가독성 향상)
def ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)

def write_cluster_assets(out_dir: pathlib.Path, k: int, cid: int, prof: dict[str, float]):
    ensure_dir(out_dir)
    age_csv = out_dir / f"cluster_{k}_{cid}_age_mix.csv"
    aud_csv = out_dir / f"cluster_{k}_{cid}_audience.csv"
    pd.DataFrame([
        {"bucket":"10s","value":float(prof.get("age_10s",0))},
        {"bucket":"30s","value":float(prof.get("age_30s",0))},
        {"bucket":"40s","value":float(prof.get("age_40s",0))},
        {"bucket":"50s","value":float(prof.get("age_50s",0))},
        {"bucket":"60p","value":float(prof.get("age_60p",0))}
    ]).to_csv(age_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"metric":"NEW","value":float(prof.get("aud_new",0))},
        {"metric":"REU","value":float(prof.get("aud_reu",0))},
        {"metric":"RES","value":float(prof.get("aud_res",0))},
        {"metric":"WORK","value":float(prof.get("aud_work",0))},
        {"metric":"FLOW","value":float(prof.get("aud_flow",0))}
    ]).to_csv(aud_csv, index=False, encoding="utf-8-sig")

    r_script = out_dir / f"cluster_{k}_{cid}_plots.R"
    png1 = out_dir / f"cluster_{k}_{cid}_age_mix.png"
    png2 = out_dir / f"cluster_{k}_{cid}_audience.png"

    r_code = f"""
suppressWarnings(suppressMessages({{
  library(ggplot2); library(scales); library(dplyr)
}}))
age_path <- "{age_csv.name}"; aud_path <- "{aud_csv.name}"
png_age <- "{png1.name}";    png_aud <- "{png2.name}"

# Age Mix
age_df <- read.csv(age_path, encoding="UTF-8")
age_df$bucket <- factor(age_df$bucket, levels=age_df$bucket[order(age_df$value,decreasing=TRUE)])
p1 <- ggplot(age_df, aes(x=bucket, y=value)) +
  geom_col(width=.6) +
  geom_text(aes(label=percent(value, accuracy=1)), vjust=-.3, size=5) +
  scale_y_continuous(labels=percent, limits=c(0,1)) +
  labs(title="연령 믹스(점유율, %) — Cluster {cid} / k={k}",
       subtitle="큰 막대일수록 손님 비중이 큼", x="연령대", y="비중(%)",
       caption="비율(0~1)을 %로 변환하여 표기") +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold", size=18))
ggsave(png_age, p1, width=8, height=5, dpi=150)

# Audience (한국어 라벨 + 강/보통/약)
aud_df <- read.csv(aud_path, encoding="UTF-8") |>
  mutate(metric=recode(metric,"NEW"="신규","REU"="단골","RES"="예약","WORK"="직장인","FLOW"="유동"),
         band=cut(value, breaks=c(-Inf,.50,.75,Inf), labels=c("약함","보통","강함"))) |>
  filter(!is.na(value), value>=0.60) |>
  arrange(desc(value))
aud_df$metric <- factor(aud_df$metric, levels=aud_df$metric)
p2 <- ggplot(aud_df, aes(x=metric, y=value, fill=band)) +
  geom_col(width=.6) +
  geom_text(aes(label=percent(value, accuracy=1)), vjust=-.3, size=5) +
  scale_y_continuous(labels=percent, limits=c(0,1)) +
  scale_fill_manual(values=c("강함"="#666666","보통"="#999999","약함"="#CCCCCC")) +
  labs(title="오디언스 신호(%) — Cluster {cid} / k={k}",
       subtitle="단골/유동/신규/예약/직장인 (0~1 스코어 → %)", x="오디언스", y="수준(%)", fill="강도",
       caption="임계값 미만(예: 60%)은 제외") +
  theme_minimal(base_size=12) +
  theme(plot.title=element_text(face="bold", size=18), legend.position="top")
ggsave(png_aud, p2, width=8, height=5, dpi=150)
"""
    r_script.write_text(r_code, encoding="utf-8")

    rscript = shutil.which("Rscript")
    if rscript:
        try:
            subprocess.run(
                [rscript, r_script.name],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
                cwd=out_dir
            )
        except subprocess.TimeoutExpired:
            print(f"!!! R SCRIPT TIMEOUT EXPIRED (60s) in {out_dir} !!!")
        except subprocess.CalledProcessError:
            print(f"!!! R SCRIPT FAILED WITH AN ERROR in {out_dir} !!!")

    out = {"age_csv": age_csv, "aud_csv": aud_csv, "r_script": r_script}
    if png1.exists(): out["age_png"] = str(png1.resolve())
    if png2.exists(): out["aud_png"] = str(png2.resolve())
    return out

# ---- 저장
def save_bundle(cx, mct_id: str, ym: str, version: str, payload: dict):
    cx.execute(text("""REPLACE INTO mat_prompt_bundle (mct_id, ym, version, bundle)
                       VALUES (:m, :ym, :v, :b)"""),
               {"m": mct_id, "ym": ym, "v": version, "b": json.dumps(payload, ensure_ascii=False)})


# ---- 메인 로직
def generate_propensity_prompts(k: int|None, top_media: int, top_kw: int, make_plots: bool, assets_dir: str,
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

    # 사용자 입력 파싱
    user_input = user_input or ""
    text_norm = normalize_text(user_input)
    masked_name = mask_shop_like_token(text_norm)

    if per_store and not mct_id:
        sectors, areas, sectors_sorted, areas_sorted = load_vocab(eng)
        if not area:   area = longest_match(text_norm, areas_sorted)
        if not sector: sector = longest_match(text_norm, sectors_sorted)

    with eng.begin() as cx:
        # 1) 정확 mct_id
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
                media = [(r[0], float(r[1])) for r in cx.execute(text(SQL_MEDIA), {"v":AFFINITY_VERSION,"ag":dom_age,"topn":int(top_media)}).fetchall()]
                kws   = [(r[0], float(r[1])) for r in cx.execute(text(SQL_KW),    {"v":AFFINITY_VERSION,"ag":dom_age,"topn":int(top_kw)}).fetchall()]
                ctx_lines = [f"biz_zone={row['biz_zone']}", f"sector={row['sector']}", f"mct_id={row['mct_id']}"]
                if masked_name: ctx_lines.append(f"masked_store_name={masked_name}")
                title = builder.build_title(cid, dom_age, f"{row['biz_zone']}/{row['sector']}")
                user_prompt = builder.build_user_prompt(k, cid, dom_age, prof, rule_text, media, kws, user_input, ctx_lines)

                bundle = {"k":k, "cluster_id":cid, "title":title, "system_prompt":None,
                          "user_prompt":user_prompt, "rule_text":rule_text, "dom_age":dom_age,
                          "profile":prof, "rec_channels":[{"channel":c,"affinity":a} for c,a in media],
                          "rec_keywords":[{"keyword":w,"affinity":a} for w,a in kws],
                          "context_lines":ctx_lines, "user_input":user_input}

                if make_plots:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_dir = pathlib.Path(assets_dir or "assets")/"propensity"/ts
                    assets = write_cluster_assets(out_dir, k, cid, prof)
                    bundle["assets"] = {
                        "age_png": str(assets["age_png"]) if "age_png" in assets else None,
                        "aud_png": str(assets["aud_png"]) if "aud_png" in assets else None,
                    }
                save_bundle(cx, mct_id=row["mct_id"], ym=row["ym"], version=AFFINITY_VERSION, payload=bundle)
                print(f"✅ per-store prompt saved (mct_id: {mct_id})\n• target k = {k}")
                return

        # 2) 지역×업종 평균
        if per_store and (area or sector):
            g = cx.execute(text(SQL_STORE_GROUP_MEAN), {"area":area, "sector":sector}).mappings().first()
            if g and g["k"] is not None:
                prof = {
                    "age_10s":float(g["age_10s"] or 0), "age_30s":float(g["age_30s"] or 0),
                    "age_40s":float(g["age_40s"] or 0), "age_50s":float(g["age_50s"] or 0),
                    "age_60p":float(g["age_60p"] or 0),
                    "aud_new":float(g["aud_new"] or 0), "aud_reu":float(g["aud_reu"] or 0),
                    "aud_res":float(g["aud_res"] or 0), "aud_work":float(g["aud_work"] or 0),
                    "aud_flow":float(g["aud_flow"] or 0),
                    "cancel_high":float(g["cancel_high"] or 0),
                    "delivery_high":float(g["delivery_high"] or 0),
                }
                k=int(g["k"])
                cr = cx.execute(text(SQL_STORE_GROUP_REP_CLUSTER), {"area":area, "sector":sector}).mappings().first()
                cid = int(cr["cluster_id"]) if cr else 0
                dom_age = dominant_age_bucket(prof)
                rule_row = cx.execute(text(SQL_BEST_RULE), {"k":k, "cid":cid}).mappings().first()
                rule_text = rule_row["rule_text"] if rule_row else None
                media = [(r[0], float(r[1])) for r in cx.execute(text(SQL_MEDIA), {"v":AFFINITY_VERSION,"ag":dom_age,"topn":int(top_media)}).fetchall()]
                kws   = [(r[0], float(r[1])) for r in cx.execute(text(SQL_KW),    {"v":AFFINITY_VERSION,"ag":dom_age,"topn":int(top_kw)}).fetchall()]
                ctx_title = f"{(area or '').strip()}/{(sector or '').strip()}".strip("/")
                ctx_lines = ([f"area={area}"] if area else []) + ([f"sector={sector}"] if sector else [])
                if masked_name: ctx_lines.append(f"masked_store_name={masked_name}")
                title = builder.build_title(cid, dom_age, ctx_title)
                user_prompt = builder.build_user_prompt(k, cid, dom_age, prof, rule_text, media, kws, user_input, ctx_lines)
                bundle = {"k":k,"cluster_id":cid,"title":title,"system_prompt":None,"user_prompt":user_prompt,
                          "rule_text":rule_text,"dom_age":dom_age,"profile":prof,
                          "rec_channels":[{"channel":c,"affinity":a} for c,a in media],
                          "rec_keywords":[{"keyword":w,"affinity":a} for w,a in kws],
                          "context_lines":ctx_lines,"user_input":user_input}
                if make_plots:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_dir = pathlib.Path(assets_dir or "assets")/"propensity"/ts
                    assets = write_cluster_assets(out_dir, k, cid, prof)
                    bundle["assets"] = {"age_png":str(assets.get("age_png")) if assets.get("age_png") else None,
                                        "aud_png":str(assets.get("aud_png")) if assets.get("aud_png") else None}
                synthetic_id = f"GROUP::{(area or 'ALL')}/{(sector or 'ALL')}"
                ym_now = datetime.datetime.now().strftime("%Y%m")
                save_bundle(cx, mct_id=synthetic_id, ym=ym_now, version=AFFINITY_VERSION, payload=bundle)
                print(f"✅ per-store prompt saved (group: {ctx_title})\n• target k = {k}")
                return bundle

    # 3) 폴백: 클러스터 평균
    prof_df = pd.read_sql(SQL_PROFILE, get_engine(), params={"k": k})
    if prof_df.empty:
        print("mat_cluster_profile 에 데이터가 없습니다. cluster_segments 먼저 실행하세요.")
        return
    prof_pivot = prof_df.pivot_table(index=["cluster_id"], columns="feature", values="mean", aggfunc="first").reset_index()
    with get_engine().begin() as cx:
        ym_now = datetime.datetime.now().strftime("%Y%m")
        for _, r in prof_pivot.iterrows():
            cid = int(r["cluster_id"])
            prof = {c: float(r[c]) for c in prof_pivot.columns if c != "cluster_id"}
            dom_age = dominant_age_bucket(prof)
            rule_row = cx.execute(text(SQL_BEST_RULE), {"k": k, "cid": cid}).mappings().first()
            rule_text = rule_row["rule_text"] if rule_row else None
            media = [(x[0], float(x[1])) for x in cx.execute(text(SQL_MEDIA), {"v":AFFINITY_VERSION,"ag":dom_age,"topn":int(TOP_MEDIA_DEFAULT)}).fetchall()]
            kws   = [(x[0], float(x[1])) for x in cx.execute(text(SQL_KW),    {"v":AFFINITY_VERSION,"ag":dom_age,"topn":int(TOP_KW_DEFAULT)}).fetchall()]
            title = PromptBuilder().build_title(cid, dom_age)
            user_prompt = PromptBuilder().build_user_prompt(k, cid, dom_age, prof, rule_text, media, kws)
            bundle = {"k":k,"cluster_id":cid,"title":title,"system_prompt":None,"user_prompt":user_prompt,
                      "rule_text":rule_text,"dom_age":dom_age,"profile":prof,
                      "rec_channels":[{"channel":c,"affinity":a} for c,a in media],
                      "rec_keywords":[{"keyword":w,"affinity":a} for w,a in kws],
                      "context_lines":[],"user_input":""}
            save_bundle(cx, mct_id=f"CLUSTER::{cid}", ym=ym_now, version=AFFINITY_VERSION, payload=bundle)
    print(f"✅ prompts saved into mat_prompt_bundle (k={k})")

# ---- CLI
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
    k=args.k, top_media=args.top_media, top_kw=args.top_kw, make_plots=args.make_plots, assets_dir=args.assets_dir,
    user_input=args.user_input, mct_id=args.mct_id, area=args.area, sector=args.sector, per_store=args.per_store)

if __name__ == "__main__":
    main()
