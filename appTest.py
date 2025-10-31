from fastapi import FastAPI, Query
from dotenv import load_dotenv
import os

# ✅ 새 import 경로 (중요!)
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL = os.getenv("QDRANT_COLL_TABLE", "regs_tables")
EMB_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

app = FastAPI(title="University Regulations Chatbot")

# --- Qdrant Client 생성 (Cloud 쓰면 api_key, prefer_grpc, https 등 추가) ---
# 예: QdrantClient(url="https://xxxx.qdrant.cloud", api_key="...")  # 클라우드
client = QdrantClient(url=QDRANT_URL)  # 로컬 기본

# --- Embedding / LLM ---
embeddings = OpenAIEmbeddings(model=EMB_MODEL)
llm = ChatOpenAI(model=LLM_MODEL)

# --- 기존 컬렉션을 벡터스토어로 래핑 ---
# (이미 인덱싱 끝난 상태이므로 from_texts가 아니라 '기존 컬렉션'에 연결)
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLL,
    embedding =embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

@app.get("/ask")
async def ask(query: str = Query(..., description="사용자 질문")):
    result = qa({"query": query})
    sources = [
        {
            "sheet": d.metadata.get("sheet"),
            "row_idx": d.metadata.get("row_idx"),
            "table_title": d.metadata.get("table_title"),
        }
        for d in result["source_documents"]
    ]
    return {"question": query, "answer": result["result"], "sources": sources}

@app.get("/")
def home():
    return {"message": "✅ Regulations Chatbot API is running!"}
  
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")  
