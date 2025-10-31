# pdf.py — 표 텍스트 제외하고 텍스트만 추출 (sw_pdf.pdf 전용)
from pathlib import Path
import pdfplumber, json, uuid, re

PDF = Path("sw_pdf.pdf")
OUT = Path("sw_pdf_text.jsonl")

def clean(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x0c", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def in_bbox(word, bbox):
    # bbox = (x0, top, x1, bottom)
    return (word["x0"] >= bbox[0] and word["x1"] <= bbox[2]
            and word["top"] >= bbox[1] and word["bottom"] <= bbox[3])

def is_in_any_table(word, table_bboxes):
    for b in table_bboxes:
        if in_bbox(word, b):
            return True
    return False

with pdfplumber.open(PDF) as doc, open(OUT, "w", encoding="utf-8") as out:
    for page_num, page in enumerate(doc.pages, start=1):
        # 표 영역(bbox) 탐지
        # 필요하면 table_settings를 조정해서 탐지 민감도 튜닝 가능
        tables = page.find_tables()  # 기본 설정으로도 대부분 잡힘
        table_bboxes = [t.bbox for t in tables]

        # 모든 단어(좌표 포함) 추출
        words = page.extract_words()
        # 표 영역에 속하지 않는 단어만 사용
        kept = [w for w in words if not is_in_any_table(w, table_bboxes)]

        # (y, x) 순으로 정렬하여 자연스러운 읽기 순서로 합치기
        kept.sort(key=lambda w: (round(w["top"], 0), w["x0"]))
        page_text = " ".join(w["text"] for w in kept)
        page_text = clean(page_text)

        if not page_text:
            continue

        rec = {
            "id": uuid.uuid4().hex,
            "source": PDF.name,
            "page": page_num,
            "text": page_text,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Saved: {OUT}")
