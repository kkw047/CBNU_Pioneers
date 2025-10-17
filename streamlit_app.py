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
    "저의 주 특기는 사람들의 성향을 분석하여 사장님 가게에 필요한 마케팅을 도와드릴 수 있어요."
    "\n 그러기 위해 꼭 필요한 사장님의 가게 정보를 입력해주세요. 주소와 가게명으로도 충분해요"
    "\n(예: 가게이름, 주소, 업종 등 2가지 이상을 제공해주세요)"
)
POSITIVE_RESPONSES = ["네", "네 ", "응", "예", "부탁", "해줘", "생성", "만들어줘"]


# ==============================================================================
# Streamlit 앱의 메인 UI 및 로직
# ==============================================================================

def main_app():
    st.set_page_config(
        page_title="AI 비밀 상담사",
        layout="wide",
        initial_sidebar_state="auto"
    )

    # --- CSS 주입 ---
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
        .stApp { font-family: 'Noto Sans KR', sans-serif; }

        [data-testid="stSidebar"] {
            width: 350px !important;
        }

        .user-message-container {
            display: flex;
            justify-content: flex-end;
        }
        .user-message-container [data-testid="stChatMessage"] {
            width: 80%;
        }

        [data-testid="stHeader"] {
            background-color: transparent;
        }

        footer { visibility: hidden; }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 120px;
        }
    </style>
    """, unsafe_allow_html=True)

    ASSETS = Path(__file__).parent / "assets"
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")

    @st.cache_data(show_spinner=False)
    def load_image(name: str) -> Optional[Image.Image]:
        p = ASSETS / name
        return Image.open(p) if p.exists() else None

    def clear_chat_history():
        st.session_state.clear()

    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]
        st.session_state.analysis_complete = False

    # --- 사이드바 UI ---
    with st.sidebar:
        st.markdown("<h2 style='text-align:center; font-weight: bold;'>내 가게를 살리는 AI  비밀 상담사</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>Team CBNU_Pioneers</p>", unsafe_allow_html=True)
        st.divider()
        if st.button("새로운 분석 시작", use_container_width=True):
            clear_chat_history()
            st.rerun()
        st.divider()
        img = load_image("shc_ci_basic_00.png")
        if img:
            st.image(img, use_container_width=True)
        st.markdown("<p style='text-align:center; color: #888;'>2025 Big Contest</p>", unsafe_allow_html=True)


    # --- 메인 대화창 UI ---
    if len(st.session_state.messages) <= 2:
        st.title("AI 마케팅 대시보드 💬")
        st.markdown("데이터 기반 분석으로 사장님의 가게에 꼭 맞는 마케팅 전략을 찾아보세요.")
        hero_img = load_image("main_Image.png")
        if hero_img:
            st.image(hero_img, use_container_width=True)

    # --- 채팅 메시지 렌더링 ---
    for m in st.session_state.messages:
        if isinstance(m, SystemMessage): continue
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        avatar = ":material/person:" if role == "user" else ":material/smart_toy:"
        with st.chat_message(role, avatar=avatar):
            if isinstance(m, AIMessage) and m.additional_kwargs.get("images"):
                reply_text = m.content
                image_paths = m.additional_kwargs["images"]
                split_keyword = "### 우리 가게 현황 요약"

                if split_keyword in reply_text:
                    parts = reply_text.split(split_keyword, 1)
                    st.markdown(parts[0])
                    st.markdown(split_keyword)

                    col1, col2 = st.columns(2)
                    if len(image_paths) > 0 and Path(image_paths[0]).exists():
                        with col1:
                            st.image(image_paths[0], use_container_width=True, caption="주요 고객 연령 분포")
                    if len(image_paths) > 1 and Path(image_paths[1]).exists():
                        with col2:
                            st.image(image_paths[1], use_container_width=True, caption="고객 행동 패턴(오디언스 신호)")

                    st.markdown(parts[1])
                else:
                    st.markdown(reply_text)
            elif isinstance(m, AIMessage) and "요청하신 마케팅 실행 체크리스트를 생성해 드렸습니다." in m.content:
                # 체크리스트 메시지를 인사말과 체크리스트 내용으로 분리
                parts = m.content.split("\n\n---\n\n", 1)
                if len(parts) == 2:
                    st.markdown(parts[0])  # 인사말 부분 출력
                    # 체크리스트 부분을 복사 버튼이 있는 코드 블록으로 출력
                    st.code(parts[1], language="markdown")
                else:
                    # 분리에 실패할 경우, 원본 메시지 전체를 출력
                    st.markdown(m.content)

            else:
                st.markdown(m.content)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
    ANALYSIS_SERVER_URL = "http://127.0.0.1:8000/analyze"

    async def process_user_input(user_query: str) -> dict:
        output = {"images": [], "reply": "", "error": ""}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(ANALYSIS_SERVER_URL, json={"user_query": user_query}, timeout=300.0)
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
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=generated_prompt)]
                llm_response = await llm.ainvoke(messages)
                output["reply"] = llm_response.content
            else:
                output["error"] = "프롬프트 생성 실패"
        except httpx.ConnectError:
            output["error"] = "분석 서버에 연결할 수 없습니다. 앱을 재시작해주세요."
        except Exception as e:
            output["error"] = f"오류 발생: {e!r}"
        return output

    # --- 채팅 입력 및 AI 응답 처리 ---
    if st.session_state.analysis_complete:
        placeholder = "마케팅 전략에 대해 더 궁금한 점을 질문하시거나, '체크리스트 만들어줘'라고 요청해보세요."
    else:
        placeholder = "주소,가게이름,업종 2가지 이상을 입력하여 대화를 이어나가세요."
    query = st.chat_input(placeholder)
    if query:
        st.session_state.messages.append(HumanMessage(content=query))

        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        last_ai_message = ""
        for msg in reversed(st.session_state.messages[:-1]):
            if isinstance(msg, AIMessage):
                last_ai_message = msg.content
                break

        if not st.session_state.analysis_complete:
            with st.spinner("사장님의 가게를 분석하고 마케팅 전략을 수립하는 중입니다..."):
                result_data = asyncio.run(process_user_input(query))
                ai_message = AIMessage(content="오류가 발생했습니다.")
                if not result_data.get("error"):
                    ai_message = AIMessage(
                        content=result_data.get("reply", "결과를 생성하지 못했습니다."),
                        additional_kwargs={"images": result_data.get("images", [])}
                    )
                else:
                    ai_message = AIMessage(content=result_data["error"])

                st.session_state.messages.append(ai_message)
                st.session_state.analysis_complete = True

        elif "체크리스트" in last_ai_message and any(word in query.lower() for word in POSITIVE_RESPONSES):
            with st.spinner("마케팅 실행 체크리스트를 생성하는 중입니다..."):
                checklist_prompt = """
                이전의 마케팅 전략 분석 내용을 바탕으로, 사용자가 순차적으로 따라 할 수 있는 구체적인 '마케팅 실행 로드맵'을 Markdown 형식으로 작성하십시오.
                [지침]
                1. 인사말은 따로 작성하지 않고 오로지 체크리스트만을 작성하십시오.
                2. 단계별 구성: 전체 과정을 '1단계: 준비', '2단계: 실행', '3단계: 확산 및 분석'과 같이 논리적인 단계로 나누어 구성하십시오.
                3. 실행 항목: 각 단계 아래에는 사용자가 완료 여부를 체크할 수 있도록 모든 실행 항목을 ` - [ ] ` 형식으로 작성해야 합니다.
                4. 구체적인 내용: 각 항목은 누가, 무엇을, 어떻게 해야 하는지 명확히 알 수 있도록 구체적이고 실행 가능한 내용으로 작성하십시오.
                """
                context_messages = st.session_state.messages.copy()
                context_messages.append(HumanMessage(content=checklist_prompt))
                llm_response = llm.invoke(context_messages)

                final_reply = f"""네, 사장님! 요청하신 마케팅 실행 체크리스트를 생성해 드렸습니다. 우측 상단 버튼을 통해 복사 가능합니다!
                \n\n---\n\n{llm_response.content}"""
                st.session_state.messages.append(AIMessage(content=final_reply))

        else:
            with st.spinner("답변을 생성하는 중입니다..."):
                llm_response = llm.invoke(st.session_state.messages)
                st.session_state.messages.append(AIMessage(content=llm_response.content))

        st.rerun()


if __name__ == '__main__':
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