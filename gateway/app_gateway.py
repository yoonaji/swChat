import sys
import os
from fastapi import FastAPI
import grpc
from contextlib import asynccontextmanager
import retriever_pb2_grpc, retriever_pb2
import generator_pb2_grpc, generator_pb2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../retriever')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../generator')))

RETRIEVER_HOST = os.getenv("RETRIEVER_HOST", "localhost:50051")
GENERATOR_HOST = os.getenv("GENERATOR_HOST", "localhost:50052")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🔗 [Gateway] Connecting to Retriever at {RETRIEVER_HOST}")
    print(f"🔗 [Gateway] Connecting to Generator at {GENERATOR_HOST}")

    retriever_conn = grpc.aio.insecure_channel(RETRIEVER_HOST)
    app.state.retriever = retriever_pb2_grpc.RetrieverServiceStub(retriever_conn)
    
    generator_conn = grpc.aio.insecure_channel(GENERATOR_HOST)
    app.state.generator = generator_pb2_grpc.GeneratorServiceStub(generator_conn)
    
    yield
    
    await retriever_conn.close()
    await generator_conn.close()

app = FastAPI(lifespan=lifespan)

@app.get("/ask")
async def ask(query: str):
    retriever_res = await app.state.retriever.Retrieve(
            retriever_pb2.RetrieveRequest(query=query)
        )
    
    contexts = [doc.page_content for doc in retriever_res.documents]
    
    sources = []
    for d in retriever_res.documents:
        sources.append({
            "sheet": d.metadata.sheet,
            "row": d.metadata.row_idx,
            "title": d.metadata.table_title
        })
        
    gen_res = await app.state.generator.Generate(
        generator_pb2.GenerateRequest(query=query, contexts=contexts)
    )
    
    return {
        "question": query,
        "answer": gen_res.answer,
        "sources": sources
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)