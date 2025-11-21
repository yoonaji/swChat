# app_generator.py

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os, time
import httpx  # ⭐️ 다른 서버와 통신하기 위한 라이브러리 (필수)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from contextlib import asynccontextmanager

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
         "너는 대학 소프트웨어융합대학(소융대)의 규정 및 교과과정 담당 조교야.\n"
        "주어진 자료(context) 내의 내용만 근거로 답변해.\n"
        "추측이나 일반적 지식으로 답하지 마. 근거가 없으면 '자료 내에서 확인 불가'라고 명시해.\n"
        "답변 마지막 줄에는 반드시 근거 출처를 [시트명(행번호)] 형태로 표기해.\n\n"
        "──────────────────────────────\n"
        "【자료】\n{context}\n"
        "──────────────────────────────\n"
        "【질문】\n{question}\n"
        "──────────────────────────────\n"
        "【답변】\n"
        "※ 참고:\n"
        "- 이 단과대의 공식 명칭은 '소프트웨어융합대학'이며, 줄여서 '소융대'라고 부른다.\n"
        "- 소융대에는 세 개의 학과가 있으며, 약칭은 다음과 같다:\n"
        "  • 컴퓨터공학과 → '컴공'\n"
        "  • 인공지능학과 → '인지' 또는 '인지과'\n"
        "  • 소프트웨어융합학과 → '소융'\n"
        "- 답변 시 '소융대'(단과대)와 '소융과'(학과)를 반드시 구분해서 사용해."

    ),
)

load_dotenv()
RETRIEVER_API_URL = os.getenv("RETRIEVER_API_URL", "http://retriever:8001")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not found in .env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 코드가 시작될 때 실행 ---
    print(f"🔗 [Generator] HTTP Client for {RETRIEVER_API_URL} ready.")
    app.state.httpx_client = httpx.AsyncClient() # 외부 api를 호출해야할 때 멈추지 않고 서버를 실행하기 위해서 사용
    
    yield  # 👈 이 시점에서 FastAPI 앱이 실행됩니다.
    
    # --- 코드가 종료될 때 실행 ---
    print("🛑 [Generator] HTTP Client closed.")
    await app.state.httpx_client.aclose()

# --- FastAPI 앱 초기화 ---
app = FastAPI(
  title="Generator Chatbot API",
  lifespan=lifespan
  )

# --- CORS 설정 (동일) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LLM 초기화 ---
# Qdrant/Retriever 관련 코드는 여기서 모두 삭제
try:
    print("🔗 [Generator] Initializing LLM...")
    llm = ChatOpenAI(model=LLM_MODEL)
    print("✅ [Generator] LLM initialized.")
except Exception as e:
    raise RuntimeError(f"🚨 [Generator] LLM Initialization failed: {e}")


# --- 헬스체크 ---
@app.get("/")
def home():
    return {"message": "✅ Generator Chatbot API is running!"}

# --- ⭐️ API 엔드포인트 (로직 변경) ---
@app.get("/ask")
async def ask(query: str = Query(..., description="사용자 질문")):
    """
    사용자의 질문을 받아 Retriever API와 LLM을 순차적으로 호출하여 답변합니다.
    """
    client = app.state.httpx_client # app.state에 저장해 두었던 httpx.AsyncClient 객체(object) 그 자체가 들어갑니다. 정확히는 그 객체를 가리키는 '참조(reference)' 또는 **'메모리 주소'**가 복사되어 client 변수에 저장
    
    # 1. ⭐️ (통신 1) Retriever API 호출
    try:
        response = await client.get(
            f"{RETRIEVER_API_URL}/retrieve", 
            params={"query": query},
            timeout=20.0
        )
        response.raise_for_status() # HTTP 오류가 200 OK가 아니면 예외 발생
        documents = response.json()
        
        # 검색된 문서 내용을 하나의 문자열로 합치기
        context = "\n\n".join([doc['page_content'] for doc in documents])
        # 출처(sources) 정보 파싱
        sources = [
            {
                "sheet": doc['metadata'].get("sheet"),
                "row_idx": doc['metadata'].get("row_idx"),
                "table_title": doc['metadata'].get("table_title"),
            }
            for doc in documents
        ]

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Retriever API ({e.request.url}) 통신 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever API 처리 오류: {e}")

    # 2. ⭐️ (통신 2) LLM API 호출 (LangChain Chain 수동 구성)
    try:
        # RetrievalQA 대신 수동으로 체인 구성
        rag_chain = (
            {"context": (lambda x: context), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 비동기로 LLM 호출
        answer = await rag_chain.ainvoke(query)
        
        return {
            "question": query,
            "answer": answer,
            "sources": sources # Retriever에서 받아온 출처 정보
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM API 처리 오류: {e}")

# --- 실행 (uvicorn용) ---
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Generator FastAPI on http://localhost:8000")
    uvicorn.run("app_generator:app", host="0.0.0.0", port=8000, log_level="info")