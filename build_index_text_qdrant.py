# build_index_text_qdrant.py
import os, json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

load_dotenv() #env파일 로드
JSONL = os.getenv("TEXT_JSONL", "sw_pdf_text.cleaned.jsonl")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL = os.getenv("QDRANT_COLL_TEXT", "regs_text")
EMB_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

emb = OpenAIEmbeddings(model=EMB_MODEL)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)

texts, metas = [], []
with open(JSONL, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        chunks = splitter.split_text(r["text"])
        texts.extend(chunks)
        metas.extend([{"source": r["source"], "page": r["page"]}] * len(chunks))

# Qdrant에 생성/업서트
vs = Qdrant.from_texts(
    texts=texts,
    embedding=emb,
    metadatas=metas,
    url=QDRANT_URL,
    collection_name=COLL,
)
print(f"✅ Text indexed to Qdrant collection: {COLL} (chunks={len(texts)})")
