# api_server.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path
import traceback

# 'tasks' 모듈을 임포트하기 위해 프로젝트 루트를 시스템 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="Marketing Analysis API Server",
    description="Streamlit 앱의 백엔드 요청을 처리하여 고객 성향 분석을 수행합니다.",
)


# HTTP 요청 본문(body)의 형식을 정의하는 모델
class QueryRequest(BaseModel):
    user_query: str


@app.post("/analyze")
def analyze_store(request: QueryRequest):
    """
    HTTP POST 요청을 받아 고객 성향 분석을 수행하고,
    성공 시 분석 결과(프롬프트, 이미지 경로 등)를 JSON으로 반환합니다.
    """
    # 함수가 호출될 때만 임포트하여 시작 시간 단축 (지연 임포트)
    from tasks.propensity_prompt_generate import generate_propensity_prompts

    print(f"[API Server] Received query for analysis: '{request.user_query}'")

    try:
        # 핵심 분석 로직 호출
        bundle = generate_propensity_prompts(
            k=None,
            top_media=5,
            top_kw=10,
            make_plots=True,
            assets_dir="assets",
            user_input=request.user_query,
            mct_id=None,
            area=None,
            sector=None,
            per_store=True
        )

        # 분석 결과에 따라 성공/실패 응답 구성
        if bundle:
            result = {
                "success": True,
                "prompt": bundle.get("user_prompt"),
                "assets": bundle.get("assets", {})
            }
            print("[API Server] Analysis successful. Sending response.")
        else:
            result = {
                "success": False,
                "message": f"'{request.user_query}'에 대한 분석 데이터를 찾을 수 없습니다. 다른 가맹점명으로 시도해보세요."
            }
            print("[API Server] Analysis failed: No data found.")

        return result

    except Exception:
        # 분석 과정에서 예상치 못한 오류 발생 시 500 에러 반환
        print("[API Server] !!! An internal error occurred during analysis !!!")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"분석 중 서버 내부 오류 발생: {traceback.format_exc()}"
        )


# if __name__ == "__main__":
#     # 이 파일을 직접 실행하면 (python api_server.py) FastAPI 서버가 시작됩니다.
#     print("Starting FastAPI server on http://127.0.0.1:8000")
#     uvicorn.run(app, host="127.0.0.1", port=8000)