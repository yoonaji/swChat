# ==============================================
# 🎓 Regulations Chatbot — FastAPI Single File
# ==============================================

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os, time

# ✅ LangChain / Qdrant imports
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

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



# ✅ .env 환경변수 로드
load_dotenv()

# --- 환경 설정값 ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLL = os.getenv("QDRANT_COLL_TABLE", "regs_tables")
EMB_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 기본 확인 ---
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not found in .env")

# ==============================================
# 🚀 FastAPI 앱 초기화
# ==============================================
app = FastAPI(
    title="KHU SW edu Chatbot",
    description="소융대 교육과정 질의응답 챗봇 API",
    version="1.0"
)

# ==============================================
# 🌐 CORS 설정 (프론트엔드 연결용)
# ==============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ 배포 시 특정 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================
# 🧠 LLM + Qdrant 연결 초기화
# ==============================================
try:
    print("🔗 Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OpenAIEmbeddings(model=EMB_MODEL)
    llm = ChatOpenAI(model=LLM_MODEL)
    
    

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLL,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5 })
  
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
    )

    print("✅ Qdrant + LLM initialized successfully.")

except Exception as e:
    raise RuntimeError(f"🚨 Initialization failed: {e}")

# ==============================================
# 🧩 요청 로깅 미들웨어
# ==============================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    dur = (time.time() - start) * 1000
    print(f"[{request.method}] {request.url.path} -> {response.status_code} ({dur:.1f}ms)")
    return response

# ==============================================
# 🩺 헬스체크
# ==============================================
@app.get("/")
def home():
    return {"message": "✅ Regulations Chatbot API is running!"}

# ==============================================
# 💬 질의응답 API
# ==============================================
@app.get("/ask")
async def ask(query: str = Query(..., description="사용자 질문")):
    """
    사용자의 질문을 받아 Qdrant 기반 검색 + LLM 생성으로 답변합니다.
    """
    try:
        result = qa({"query": query})
        sources = [
            {
                "sheet": d.metadata.get("sheet"),
                "row_idx": d.metadata.get("row_idx"),
                "table_title": d.metadata.get("table_title"),
            }
            for d in result.get("source_documents", [])
        ]

        return {
            "question": query,
            "answer": result.get("result", ""),
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {e}")

# ==============================================
# 🔥 실행 진입점 (uvicorn 실행용)
# ==============================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
