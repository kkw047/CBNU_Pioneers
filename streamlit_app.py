import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import streamlit as st
import multiprocessing
import uvicorn
import time
import httpx
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ==============================================================================
# 벡엔드 서버 관리 함수들
# ==============================================================================

def run_api_server():
    """uvicorn을 사용하여 FastAPI 서버를 실행하는 함수"""
    from api_server import app
    print("[API Server Process] Starting Uvicorn server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)


def is_server_running():
    """서버가 응답하는지 확인하여 실행 여부를 판단"""
    try:
        with httpx.Client() as client:
            response = client.get("http://127.0.0.1:8000/")
            return response.status_code == 404
    except httpx.ConnectError:
        return False


# ==============================================================================
# 앱 전체에서 사용될 상수들
# ==============================================================================

system_prompt = (
    "당신은 대한민국 소상공인을 위한 최고의 마케팅 전문가입니다. "
    "제공된 데이터 분석 요약 프롬프트를 바탕으로, 사용자가 이해하기 쉬운 최종 보고서를 작성합니다."
)

# ==============================================================================
# Streamlit 앱의 메인 UI 및 로직
# ==============================================================================

def main_app():
    """Streamlit 앱의 전체 UI와 로직을 포함하는 메인 함수"""

    st.set_page_config(page_title="신한카드 소상공인 비밀상담소", layout="wide")

    ASSETS = Path(__file__).parent / "assets"
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")

    @st.cache_data(show_spinner=False)
    def load_image(name: str) -> Optional[Image.Image]:
        p = ASSETS / name
        return Image.open(p) if p.exists() else None

    def clear_chat_history():
        analysis_type = st.session_state.get("analysis_type", "propensity")
        greeting = (
            "사장님, 반갑습니다! 저는 사장님의 마케팅 고민을 해결해 드릴 비밀상담소의 AI 컨설턴트입니다. "
            "성공적인 마케팅의 첫걸음, 바로 고객을 아는 것이죠. 분석하고 싶은 가맹점 상호명을 알려주세요."
            "\n(예: 동대문엽기떡볶이, 유유커피, 희망분식 등)"
        )
        if analysis_type == "crisis":
            greeting = (
                "사장님, 반갑습니다! 저는 사장님의 마케팅 고민을 해결해 드릴 비밀상담소의 AI 컨설턴트입니다. "
                "성공적인 마케팅의 첫걸음, 바로 환경을 아는 것이죠. 분석하고 싶은 가맹점 상호명을 알려주세요."
                "\n(예: 동대문엽기떡볶이, 유유커피, 희망분식 등)"
            )
        st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]
        st.session_state.analysis_complete = False
        st.session_state.crisis_choices = None


    with st.sidebar:
        img = load_image("shc_ci_basic_00.png")
        if img:
            st.image(img, width='stretch')
        st.markdown("<p style='text-align:center;'>2025 Big Contest</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
        st.button("Clear Chat History", on_click=clear_chat_history)

    st.title("신한카드 소상공인 🔑 비밀상담소")
    st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 .. 🤤")
    hero = load_image("image_gen3.png")
    if hero:
        st.image(hero, width='stretch', caption="🌀 머리아픈 마케팅 📊 어떻게 하면 좋을까?")
    st.write("")

    analysis_type_display = st.selectbox(
        "어떤 분석을 도와드릴까요?",
        ("고객 성향 분석", "환경 및 업종 분석"),
        key = "analysis_type_selector",
        help = "분석 종류를 변경하면 채팅 기록이 초기화됩니다!"
    )
    analysis_type_key = "propensity" if "고객 성향" in analysis_type_display else "crisis"

    if "messages" not in st.session_state or st.session_state.get("analysis_type") != analysis_type_key:
        st.session_state.analysis_type = analysis_type_key
        clear_chat_history()
        st.rerun()

    # --- 여기서 화면에 있는 모든 채팅 메시지를 그려줌 ---
    for m in st.session_state.messages:
        if isinstance(m, SystemMessage):
            continue
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            # AIMessage이고 'images' 정보가 추가된 특별한 메시지인지 확인
            if isinstance(m, AIMessage) and m.additional_kwargs.get("images"):
                reply_text = m.content
                image_paths = m.additional_kwargs["images"]

                # --- 이미지 분리 및 렌더링 로직 ---
                split_keyword_base = "우리 가게 현황 요약"
                found_split = False
                for keyword_variant in [
                    f"### {split_keyword_base}",
                    f"**{split_keyword_base}**", split_keyword_base
                ]:
                    if keyword_variant in reply_text:
                        parts = reply_text.split(keyword_variant, 1)
                        st.markdown((parts[0] + keyword_variant).replace('\n', '  \n'))
                        if image_paths:
                            col1, col2 = st.columns(2)
                            with col1:
                                if len(image_paths) > 0 and Path(image_paths[0]).exists():
                                    st.image(image_paths[0], use_container_width=True, caption="연령 믹스")
                            with col2:
                                if len(image_paths) > 1 and Path(image_paths[1]).exists():
                                    st.image(image_paths[1], use_container_width=True, caption="오디언스 신호")
                        st.markdown(parts[1].replace('\n', '  \n'))
                        found_split = True
                        break

                if not found_split:
                    st.markdown(reply_text.replace('\n', '  \n'))
            else:
                # 일반 메시지 (사용자 입력, 첫 인사, 체크리스트 등)
                st.markdown(m.content.replace('\n', '  \n'))

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1,
    )

    async def call_analysis_api(payload: Dict[str, Any]) -> Dict[str, Any]:
        analysis_type = st.session_state.analysis_type
        if analysis_type == "propensity":
            ANALYSIS_SERVER_URL = "http://127.0.0.1:8000/analyze"
        else:
            ANALYSIS_SERVER_URL = "http://127.0.0.1:8000/analyze_crisis"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    ANALYSIS_SERVER_URL, json=payload, timeout=300.0
                )
                response.raise_for_status()
                # analysis_result = response.json()
                return response.json()
        except httpx.ConnectError:
            return {"status": "error", "message": "분석 서버에 연결할 수 없습니다. 앱을 재시작해주세요."}
        except Exception as e:
            return {"status": "error", "message": f"오류 발생: {e!r}"}

    if "analysis_pending_mct_id" not in st.session_state:
        st.session_state.analysis_pending_mct_id = None

    if st.session_state.get("crisis_choices"):
        choices = st.session_state.crisis_choices
        options = ["선택해주세요..."] + [f"{c['name']} ({c['addr']})" for c in choices]

        selected_option = st.selectbox("여러 가게가 검색되었습니다. 하나를 선택해주세요.", options, key = "store_selector")

        if selected_option != "선택해주세요...":
            selected_index = options.index(selected_option) - 1
            st.session_state.analysis_pending_mct_id = choices[selected_index]['mct_id']
            st.session_state.crisis_choices = None
            st.rerun()

    query = st.chat_input("가맹점 상호명을 입력하거나 대화를 이어가세요...")
    if query:
        # 사용자 메시지를 기억 장소에 추가
        st.session_state.messages.append(HumanMessage(content=query))
        st.session_state.analysis_pending_mct_id = None
        st.rerun()

    last_message_is_user = isinstance(st.session_state.messages[-1], HumanMessage)
    analysis_needed = not st.session_state.analysis_complete and (last_message_is_user or st.session_state.analysis_pending_mct_id)

    if analysis_needed:
        # AI 응답을 기다리기 전에, 사용자 메시지를 화면에 "즉시" 그려줌
        with st.chat_message("assistant"):
            with st.spinner(f"사장님의 가게에 대한 '{analysis_type_display}'을(를) 진행 중입니다..."):
                payload = {"analysis_type": st.session_state.analysis_type}
                if st.session_state.analysis_pending_mct_id:
                    payload["mct_id"] = st.session_state.analysis_pending_mct_id
                else:
                    payload["user_query"] = st.session_state.messages[-1].content

                api_result = asyncio.run(call_analysis_api(payload))
                st.session_state.analysis_pending_mct_id = None

                if api_result.get("status") == "success":
                    bundle = api_result.get("bundle", {})
                    prompt = bundle.get("user_prompt")
                    reply_text = asyncio.run(llm.ainvoke([HumanMessage(content=prompt)])).content if prompt else "보고서 생성에 실패했습니다."

                    images = []
                    if st.session_state.analysis_type == 'propensity':
                        assets = bundle.get("assets", {})
                        if assets.get("age_png"): images.append(assets["age_png"])
                        if assets.get("aud_png"): images.append(assets["aud_png"])

                    st.session_state.messages.append(AIMessage(content = reply_text, additional_kwargs = {"images": images}))
                    st.session_state.analysis_complete = True
                elif api_result.get("status") == "multiple_choice":
                    st.session_state.crisis_choices = api_result.get("choices", [])
                    st.session_state.messages.append(AIMessage(content = "여러 가게가 검색되었습니다. 아래 목록에서 분석하고 싶은 가게를 선택해주세요."))
                else:
                    error_msg = api_result.get("message", "알 수 없는 오류가 발생했습니다.")
                    st.session_state.messages.append(AIMessage(content = error_msg))

                st.rerun()

    elif last_message_is_user and st.session_state.analysis_complete:
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중입니다..."):
                llm_response = llm.invoke(st.session_state.messages)
                st.session_state.messages.append(AIMessage(content = llm_response.content))
                st.rerun()

# ==============================================================================
# 메인 실행 블록
# ==============================================================================

if __name__ == '__main__':
    # 멀티프로세싱 관련 설정 (Windows 사용자를 위해 필요)
    multiprocessing.freeze_support()

    if not is_server_running():
        print("[Main] API server is not running. Starting it now...")
        server_process = multiprocessing.Process(target=run_api_server, daemon=True)
        server_process.start()
        time.sleep(5)
        if is_server_running():
            print("[Main] API server has started successfully.")
        else:
            print("[Main] Error: API server failed to start.")
            st.error("백엔드 서버를 시작하는 데 실패했습니다. 앱을 다시 로드해주세요.")
            st.stop()
    else:
        print("[Main] API server is already running.")

    main_app()