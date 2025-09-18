# streamlit_app.py
# (선택) gRPC 경고 숨기기
import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

import sys
import asyncio
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
try:
    from langchain_mcp_adapters.tools import load_mcp_tools
except ImportError:
    from langchain_mcp.tools import load_mcp_tools  # 호환
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ─────────────────────────────
# 기본 UI/상태
# ─────────────────────────────
st.set_page_config(page_title="신한카드 소상공인 비밀상담소", layout="wide")

ASSETS = Path(__file__).parent / "assets"
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")

system_prompt = (
    "당신은 친절한 마케팅 상담사입니다. 가맹점명을 받아 해당 가맹점의 방문 고객 현황을 분석하고, "
    "적절한 마케팅 방법/채널/메시지를 간결하게 추천합니다. 표를 활용해 알아보기 쉽게 설명하세요."
)
greeting = (
    "마케팅이 필요한 가맹점을 알려주세요  \n"
    "(조회가능 예시: 동대*, 유유*, 똥파*, 본죽*, 본*, 원조*, 희망*, 혁이*, H커*, 케키*)"
)

@st.cache_data(show_spinner=False)
def load_image(name: str) -> Optional[Image.Image]:
    p = ASSETS / name
    return Image.open(p) if p.exists() else None

def clear_chat_history():
    st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]

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
    st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]

for m in st.session_state.messages:
    role = "user" if isinstance(m, HumanMessage) else ("assistant" if isinstance(m, AIMessage) else "system")
    with st.chat_message(role):
        st.write(m.content)

def render_chat_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(str(content).replace("<br>", "  \n"))

# ─────────────────────────────
# LLM & MCP
# ─────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

# 현재 venv의 python.exe로 MCP 서버 실행 (Windows/PyCharm 안전)
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str((Path(__file__).parent / "mcp_server.py").resolve())],
    env=None,
)

async def process_user_input():
    """사용자 질의를 MCP+LLM로 처리"""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(llm, tools)
            result = await agent.ainvoke({"messages": st.session_state.messages})
            # 마지막 AI 메시지
            msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
            return msgs[-1].content if msgs else "(응답 없음)"

# ─────────────────────────────
# 입력창
query = st.chat_input("가맹점 이름을 입력하세요")
if query:
    st.session_state.messages.append(HumanMessage(content=query))
    render_chat_message("user", query)

    with st.spinner("Thinking..."):
        try:
            reply = asyncio.run(process_user_input())
            st.session_state.messages.append(AIMessage(content=reply))
            render_chat_message("assistant", reply)
        except* Exception as eg:
            # 여러 예외 묶음 처리 (Python 3.11+)
            for i, exc in enumerate(eg.exceptions, 1):
                msg = f"오류가 발생했습니다 #{i}: {exc!r}"
                st.session_state.messages.append(AIMessage(content=msg))
                render_chat_message("assistant", msg)
