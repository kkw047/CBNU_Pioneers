from __future__ import annotations
import sys, pathlib
import pandas as pd
from sqlalchemy import text

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dbconnect import get_engine
except ModuleNotFoundError:
    from tasks.dbconnect import get_engine

SQL_FIND_STORE_BY_PREFIX = r"""SELECT mct_id, name, sector, addr FROM dim_store WHERE name LIKE :search;"""

SQL_CRISIS_DATA = r"""SELECT close_rate_A, close_rate_B, close_rate_C, resident_user, floating_user
FROM mat_prompt_bundle_crisis WHERE mct_id = :mct_id;"""

SQL_FIND_STORE_BY_ID = r"""SELECT mct_id, name, sector, addr FROM dim_store WHERE mct_id = :mct_id;"""

def is_approx_equal(a, b):
    a, b = a * 100, b * 100
    tolerance = max(5, min(a, b) * 0.25)
    return abs(a - b) <= tolerance

def analyze_crisis_situation(rates: dict) -> tuple:
    a = rates.get('close_rate_A', 0)
    b = rates.get('close_rate_B', 0)
    c = rates.get('close_rate_C', 0)
    situations = {
        "SINKING_MARKET": ("상권 붕괴형 위기", "업종 자체는 유망하지만, 가게가 위치한 상권 전체가 무너지고 있어 위험한 상황입니다.", "상권의 유동인구를 늘리기 위한 공동 프로모션, 배달/온라인 판매 강화를 통한 상권 의존도 탈피 전략이 필요합니다."),
        "DECLINING_INDUSTRY": ("업종 쇠퇴형 위기", "상권은 활발하지만, 안타깝게도 현재 업종 자체가 사양길에 접어들어 전망이 어둡습니다.", "신규 메뉴 개발, 업종 전환(리브랜딩), 기존 고객 대상 멤버십 강화를 통한 이탈 방지가 시급합니다."),
        "HYPER_COMPETITION": ("과당 경쟁형 위기", "상권과 업종 모두 괜찮지만, 유독 이 상권 내에서 동일 업종 경쟁이 너무 치열하거나, 우리 가게가 경쟁력을 잃고 있습니다.", "경쟁 업체와 차별화되는 우리 가게만의 강점(가격, 품질, 서비스)을 부각하고, 타겟 고객을 더 세분화하는 전략이 필요합니다."),
        "STRONG_MARKET_SURVIVOR": ("상권 특수형 성장", "전국적으로 업종이 불황임에도, 현재 상권의 특수성 덕분에 훌륭하게 방어하며 성장하고 있습니다.", "현재의 강점을 유지하며, 상권 내 독점적 지위를 강화하기 위한 VVIP 마케팅, 신메뉴 출시를 통한 고객 만족도 극대화가 좋습니다."),
        "STRONG_INDUSTRY_SURVIVOR": ("업종 특수형 선방", "상권 전체가 심각한 불황이지만, 업종 자체의 힘 덕분에 주변 가게들보다 월등히 잘 버티고 있습니다.", "업종의 강점을 더욱 부각시키는 전문성 강조 마케팅과, 충성 고객 대상의 커뮤니티 활성화 전략을 추천합니다."),
        "MARKET_DOMINATOR": ("초격차 시장 지배자", "업종과 상권 모두가 어려운 최악의 상황 속에서, 압도적인 경쟁력으로 위기를 기회로 만든 상태입니다.", "현재의 성공 방정식을 유지하며, 브랜드 가치를 높이는 스토리텔링 마케팅과, 2호점 등 사업 확장을 고려해볼 시점입니다."),
        "CRITICAL_SITUATION": ("총체적 난국", "업종 전망도 좋지 않고, 상권마저 침체된데다, 우리 가게의 경쟁력마저 약해진 가장 위험한 상태입니다.", "단기적으로는 파격적인 할인 프로모션으로 현금 흐름을 확보하고, 장기적으로는 업종 전환이나 가게 이전을 심각하게 고려해야 합니다."),
        "STABLE_GROWTH": ("안정 성장기", "가게, 업종, 상권이 모두 같은 흐름을 타며 안정적으로 성장하고 있습니다.", "현재의 안정세를 유지하며, 신규 고객 유치를 위한 온라인 광고와 재방문 고객을 위한 쿠폰/포인트 제도를 병행하는 것을 추천합니다."),
        "TOTAL_RECESSION": ("총체적 불황", "가게, 업종, 상권이 모두 같은 흐름으로 침체기에 빠져있습니다.", "비용 절감을 최우선으로 하되, '가성비'를 강조하는 메뉴나 서비스를 통해 불황 속 실속을 챙기는 고객을 공략해야 합니다."),
    }

    if is_approx_equal(a, c) and a > b: return situations["SINKING_MARKET"]
    if is_approx_equal(a, b) and a > c: return situations["DECLINING_INDUSTRY"]
    if a > b and a > c and is_approx_equal(b, c): return situations["HYPER_COMPETITION"]
    if b > a and b > c and is_approx_equal(a, c): return situations["STRONG_MARKET_SURVIVOR"]
    if c > a and c > b and is_approx_equal(a, b): return situations["STRONG_INDUSTRY_SURVIVOR"]
    if b > a and c > a and is_approx_equal(b, c): return situations["MARKET_DOMINATOR"]
    if a > c > b: return situations["CRITICAL_SITUATION"]
    if is_approx_equal(a, b) and is_approx_equal(b, c):
        return situations["TOTAL_RECESSION"] if (a + b + c) / 3 > 0.1 else situations["STABLE_GROWTH"]

    return "복합 상황", "여러 요인이 복합적으로 작용하고 있는 상황입니다.", "가장 높은 폐업률 지표에 집중하여 개선 전략을 수립해야 합니다."

def analyze_customer_composition(rates: dict, sector: str) -> tuple | None:
    resident = rates.get('resident_user', 0)
    floating = rates.get('floating_user', 0)

    if abs(resident - floating) > 0.2:
        if resident > floating:
            return (
                "온라인 강화형 고객 구조",
                "매장에 직접 방문하는 거주/직장인 고객 비중이 높아 안정적인 매출 기반을 갖추고 계십니다. 다만, 이는 역으로 온라인 잠재 고객을 놓치고 있다는 신호일 수 있습니다.",
                f"사장님의 업종인 '{sector}'의 특성을 살려, 온라인 인지도를 높이고 새로운 고객을 유치할 수 있는 인스타그램 광고 문구 예시를 2가지 추천해주세요."
            )
        else:
            return (
                "오프라인 강화형 고객 구조",
                "배달, 포장 등 비대면/유동 고객의 비중이 높아 온라인 채널을 잘 활용하고 계십니다. 하지만 매장 방문 고객이 적어 단골 확보나 객단가 상승에 어려움을 겪을 수 있습니다.",
                f"사장님의 업종인 '{sector}'의 특성을 살려, 매장 방문을 유도하고 재방문 고객을 늘릴 수 있는 오프라인 이벤트 아이디어와 홍보 문구를 2가지 추천해주세요."
            )
    return None

class crisis_prompt_builder:
    def build_user_prompt(self, store_info: dict, crisis_data: dict, analysis_result: tuple, customer_analysis: tuple | None) -> str:
        title, diagnosis, solution = analysis_result

        customer_block = ""
        llm_extra_instruction = ""
        if customer_analysis:
            cus_title, cus_diagnosis, cus_solution = customer_analysis
            customer_lines = [
                f"- 진단 : {cus_title}",
                f"- 분석 : {cus_diagnosis}"
            ]
            customer_block = "\n\n[고객 구성 진단 및 추가 제안]\n" + "\n".join(customer_lines)
            llm_extra_instruction = f"5. **추가 마케팅 제안**: '{cus_solution}' 라는 요청에 맞춰, 구체적인 광고 문구나 이벤트 아이디어를 제안해주세요."

        return f"""당신은 대한민국 소상공인을 위한 최고의 상권 및 업종 분석 전문가이자 위기관리 컨설턴트입니다.
아래 데이터를 바탕으로 사장님이 이해하기 쉬운 최종 보고서를 작성해주세요.

[가게 기본 정보]
- 상호명: {store_info.get('name', 'N/A')}
- 주소: {store_info.get('addr', 'N/A')}
- 업종: {store_info.get('sector', 'N/A')}

[상권 위기 진단 지표]
- 주변 동일업종 폐업률 (A): {crisis_data.get('close_rate_A', 0):.1%} (가게의 직접적 경쟁 환경)
- 전국 동일업종 폐업률 (B): {crisis_data.get('close_rate_B', 0):.1%} (업종 자체의 전망)
- 주변 전체업종 폐업률 (C): {crisis_data.get('close_rate_C', 0):.1%} (가게가 속한 상권의 활력)
- 고객 구성 (거주/유동): {crisis_data.get('resident_user', 0):.1%} / {crisis_data.get('floating_user', 0):.1%}

[분석 결과 요약]
- 진단명: {title}
- 현재 상황: {diagnosis}

[사장님을 위한 맞춤 솔루션]
- 핵심 전략: {solution}{customer_block}

[최종 보고서 작성 가이드]
1.  **진단 요약**: "[진단명]"을 바탕으로, 사장님의 가게가 현재 어떤 상황인지 1~2문장으로 쉽고 부드럽게 설명해주세요.
2.  **상세 분석**: '상권 위기 진단 지표'의 A, B, C 수치를 "사장님 가게 주변 동일 업종 가게 100곳 중 {crisis_data.get('close_rate_A', 0) * 100:.0f}곳이 문을 닫았어요." 와 같이 구체적인 예시를 들어 설명해주세요. '현재 상황' 진단을 이 수치들과 연결하여 왜 그런 진단이 나왔는지 근거를 설명해주세요.
3.  **맞춤 솔루션 제안**: '핵심 전략'을 바탕으로, 사장님이 지금 당장 실행할 수 있는 액션 아이템 2~3가지를 구체적으로 제안해주세요. (예: "배달 앱에서 '우리동네 맛집' 광고를 시작해보세요.", "점심시간 직장인들을 위한 할인 세트 메뉴를 개발해보세요.")
4.  **응원 메시지**: 마지막으로, 사장님께 희망과 격려의 메시지를 전달해주세요.
{llm_extra_instruction}"""

def generate_crisis_prompts(user_query: str | None = None, mct_id: str | None = None):
    eng = get_engine()
    builder = crisis_prompt_builder()

    with eng.connect() as cx:
        store_info = None

        if mct_id:
            store_info_row = cx.execute(text(SQL_FIND_STORE_BY_ID), {"mct_id": mct_id}).mappings().first()
            if store_info_row:
                store_info = dict(store_info_row)
        elif user_query:
            input_len = len(user_query)
            if input_len < 2:
                return {"status": "error", "message": "검색어는 두 글자 이상 입력해주세요."}
            search_prefix = user_query[0] if input_len == 2 else user_query[:2]
            all_matches_df = pd.read_sql(text(SQL_FIND_STORE_BY_PREFIX), cx, params={"search": f"{search_prefix}%"})
            if all_matches_df.empty:
                return {"status": "not_found", "message": f"'{user_query}'와(과) 일치하는 가게를 찾을 수 없습니다."}
            exact_len_matches = all_matches_df[all_matches_df['name'].str.len() == input_len]
            if len(exact_len_matches) == 1:
                store_info = exact_len_matches.iloc[0].to_dict()
            elif len(exact_len_matches) > 1:
                choices = exact_len_matches[['mct_id', 'name', 'addr']].to_dict('records')
                return {"status": "multiple_choices", "choices": choices}
            else:
                all_matches_list = all_matches_df.to_dict('records')
                if len(all_matches_list) == 1:
                    store_info = all_matches_list[0]
                else:
                    choices = all_matches_df[['mct_id', 'name', 'addr']].to_dict('records')
                    return {"status": "multiple_choices", "choices": choices}

        if not store_info:
            return {"status": "not_found", "message": "가게 정보를 확정할 수 없습니다."}

        final_mct_id = store_info['mct_id']
        crisis_data_row = cx.execute(text(SQL_CRISIS_DATA), {"mct_id": final_mct_id}).mappings().first()
        if not crisis_data_row:
            return {"status": "error", "message": "가게의 위기 진단 데이터가 존재하지 않습니다."}

        crisis_data = {k: float(v) for k, v in crisis_data_row.items()}

        analysis_result = analyze_crisis_situation(crisis_data)
        customer_analysis_result = analyze_customer_composition(crisis_data, store_info['sector'])

        user_prompt = builder.build_user_prompt(store_info, crisis_data, analysis_result, customer_analysis_result)

        bundle = {
            "mct_id": final_mct_id,
            "title": f"{store_info['name']} 상권 위기 진단 보고서",
            "user_prompt": user_prompt,
            "assets": {}
        }

        return {"status": "success", "bundle": bundle}

    
