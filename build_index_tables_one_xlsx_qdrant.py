# build_index_tables_one_xlsx_qdrant.py
import os, re
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

load_dotenv()
XLSX = Path(os.getenv("TABLE_XLSX", "realTable.xlsx"))  # ← 네 파일 경로
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLL = os.getenv("QDRANT_COLL_TABLE", "regs_tables")
EMB_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def row_to_text(row: pd.Series) -> str:
    parts = []
    for k, v in row.items():
        v = "" if pd.isna(v) else str(v)
        if v.strip() == "":
            continue
        parts.append(f"{k}: {v}")
    return " / ".join(parts)

emb = OpenAIEmbeddings(model=EMB_MODEL)
xls = pd.ExcelFile(XLSX)

all_texts, all_metas = [], []
for sheet in xls.sheet_names:
    df = xls.parse(sheet, dtype=str).fillna("")
    df = df.loc[~(df == "").all(axis=1)]
    title = sheet.strip()
    title_norm = norm(title)
    for i, r in df.iterrows():
        t = row_to_text(r)
        if not t:
            continue
        all_texts.append(t)
        all_metas.append({
            "table_file": XLSX.name,
            "sheet": sheet,
            "table_title": title,
            "table_title_norm": title_norm,
            "row_idx": int(i),
        })

Qdrant.from_texts(
    texts=all_texts,
    embedding=emb,
    metadatas=all_metas,
    url=QDRANT_URL,
    collection_name=COLL,
)
print(f"✅ Tables indexed to Qdrant collection: {COLL} (rows={len(all_texts)})")
