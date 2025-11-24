# app_retriever.py
import asyncio
from fastapi import FastAPI, Query, HTTPException
from dotenv import load_dotenv
import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
import grpc
import retriever_pb2
import retriever_pb2_grpc

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

class RetrieverServicer(retriever_pb2_grpc.RetrieverServiceServicer):
    async def Search(self, request, context):
        query = request.query
        print(f"grpc 검색 요청: {query}")
        
        try:
      
            docs = await retriever.ainvoke(query)

            proto_docs = []
            for d in docs:
                meta = d.metadata
                proto_docs.append(retriever_pb2.Document(
                    content=d.page_content,
                    metadata=retriever_pb2.Metadata(
                        sheet=str(meta.get("sheet", "")),
                        row_idx=str(meta.get("row_idx", "")),
                        table_title=str(meta.get("table_title", ""))
                    )
                ))
    
            return retriever_pb2.SearchResponse(documents=proto_docs)
            
        except Exception as e:
            print(f"🚨 Error: {e}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return retriever_pb2.SearchResponse()


async def serve():
    server = grpc.aio.server()
    retriever_pb2_grpc.add_RetrieverServiceServicer_to_server(RetrieverServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("🚀 Retriever gRPC Server running on port 50051")
    await server.start()
    await server.wait_for_termination()
    
if __name__ == "__main__":
    asyncio.run(serve())