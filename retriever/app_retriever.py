# app_retriever.py

from fastapi import FastAPI, Query, HTTPException
from dotenv import load_dotenv
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333") # docker-compose용
COLL = os.getenv("QDRANT_COLL_TABLE", "regs_tables")
EMB_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not found in .env")

app = FastAPI(title="Retriever API")

try:
    print("🔗 [Retriever] Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OpenAIEmbeddings(model=EMB_MODEL)
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLL,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("✅ [Retriever] Qdrant Retriever initialized.")

except Exception as e:
    raise RuntimeError(f"🚨 [Retriever] Initialization failed: {e}")

@app.get("/")
def home():
    return {"message": "✅ Retriever API is running!"}

@app.get("/retrieve")
async def retrieve_documents(query: str = Query(..., description="검색할 질문")):
    """
    질문을 받아 Qdrant에서 관련 문서를 검색하여 반환합니다.
    """
    try:
      
        docs = await retriever.ainvoke(query)

        results = [
            {
                "page_content": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]
        return results
    
    except Exception as e:
        print(f"🚨 [Retriever] /retrieve CRITICAL ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"❌ Error: {e}")

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Retriever FastAPI on http://localhost:8001")
    uvicorn.run("app_retriever:app", host="0.0.0.0", port=8001, log_level="info")