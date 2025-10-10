import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

import asyncio
from pathlib import Path
from typing import Optional
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
            response = client.get("http://127.0.0.1:8000/docs")
            return response.status_code == 200
    except httpx.ConnectError:
        return False


# ==============================================================================
# 앱 전체에서 사용될 상수들
# ==============================================================================

system_prompt = (
    "당신은 대한민국 소상공인을 위한 최고의 마케팅 전문가입니다. "
    "제공된 데이터 분석 요약 프롬프트를 바탕으로, 사용자가 이해하기 쉬운 최종 보고서를 작성합니다."
)

greeting = (
    "사장님, 반갑습니다! 저는 사장님의 마케팅 고민을 해결해 드릴 비밀상담소의 AI 컨설턴트입니다. "
    "성공적인 마케팅의 첫걸음, 바로 고객을 아는 것이죠. 분석하고 싶은 가맹점 상호명을 알려주세요."
    "\n(예: 동대문엽기떡볶이, 유유커피, 희망분식 등)"
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
        st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]
        st.session_state.analysis_complete = False

    with st.sidebar:
        img = load_image("shc_ci_basic_00.png")
        if img:
            st.image(img, use_container_width=True)
        st.markdown("<p style='text-align:center;'>2025 Big Contest</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
        st.button("Clear Chat History", on_click=clear_chat_history)

    st.title("신한카드 소상공인 🔑 비밀상담소")
    st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 .. 🤤")
    hero = load_image("image_gen3.png")
    if hero:
        st.image(hero, use_container_width=True, caption="🌀 머리아픈 마케팅 📊 어떻게 하면 좋을까?")
    st.write("")

    if "messages" not in st.session_state:
        clear_chat_history()

    for m in st.session_state.messages:
        if isinstance(m, SystemMessage):
            continue
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(m.content.replace('\n', '  \n'))

    def render_chat_message(role: str, content: str):
        with st.chat_message(role):
            st.markdown(str(content).replace("<br>", "  \n"))

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # 모델명을 최신 버전으로 변경하는 것을 권장합니다.
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )

    ANALYSIS_SERVER_URL = "http://127.0.0.1:8000/analyze"

    async def process_user_input(user_query: str) -> dict:
        output = {"images": [], "reply": "", "error": ""}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    ANALYSIS_SERVER_URL, json={"user_query": user_query}, timeout=300.0
                )
                response.raise_for_status()
                analysis_result = response.json()

            if not analysis_result.get("success"):
                output["error"] = analysis_result.get("message", "분석 실패")
                return output

            assets = analysis_result.get("assets", {})
            if assets.get("age_png"): output["images"].append(assets["age_png"])
            if assets.get("aud_png"): output["images"].append(assets["aud_png"])

            generated_prompt = analysis_result.get("prompt")
            if generated_prompt:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=generated_prompt)
                ]
                llm_response = await llm.ainvoke(messages)
                output["reply"] = llm_response.content
            else:
                output["error"] = "프롬프트 생성 실패"

        except httpx.ConnectError:
            output["error"] = "분석 서버에 연결할 수 없습니다. 앱을 재시작해주세요."
        except Exception as e:
            output["error"] = f"오류 발생: {e!r}"

        return output

    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False

    query = st.chat_input("가맹점 상호명을 입력하거나 대화를 이어가세요...")
    if query:
        # 사용자 메시지를 기록하고 화면에 표시
        st.session_state.messages.append(HumanMessage(content=query))
        with st.chat_message("user"):
            st.markdown(query)

        # 1. 첫 분석 요청인 경우 (가맹점명 입력)
        if not st.session_state.analysis_complete:
            with st.spinner("사장님의 가게를 분석하고 마케팅 전략을 수립하는 중입니다..."):
                result_data = asyncio.run(process_user_input(query))
                with st.chat_message("assistant"):
                    if result_data.get("error"):
                        error_msg = result_data["error"]
                        st.error(error_msg)
                        st.session_state.messages.append(AIMessage(content=error_msg))
                    else:
                        reply_text = result_data.get("reply", "응답을 생성하지 못했습니다.")
                        image_paths = result_data.get("images", [])

                        # --- 이미지 분리 로직 ---
                        split_keyword_base = "우리 가게 현황 요약"
                        found_split = False
                        for keyword_variant in [
                            f"### {split_keyword_base}", f"**{split_keyword_base}**", f"{split_keyword_base}", split_keyword_base
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
                            st.markdown(reply_text)
                            if image_paths:
                                col1, col2 = st.columns(2)
                                with col1:
                                    if len(image_paths) > 0 and Path(image_paths[0]).exists():
                                        st.image(image_paths[0], use_container_width=True, caption="연령 믹스")
                                with col2:
                                    if len(image_paths) > 1 and Path(image_paths[1]).exists():
                                        st.image(image_paths[1], use_container_width=True, caption="오디언스 신호")

                        st.session_state.messages.append(AIMessage(content=reply_text))
                        st.session_state.analysis_complete = True

        # 2. 분석 완료 후의 후속 질문인 경우
        else:
            with st.spinner("답변을 생성하는 중입니다..."):
                llm_response = llm.invoke(st.session_state.messages)
                reply_text = llm_response.content

                with st.chat_message("assistant"):
                    st.markdown(reply_text.replace('\n', '  \n'))
                st.session_state.messages.append(AIMessage(content=reply_text))


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