# clean_jsonl.py
import json, re

def clean(s: str) -> str:
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

inp = "sw_pdf_text.jsonl"          # 현재 파일
out = "sw_pdf_text.cleaned.jsonl"  # 정리본

with open(inp, "r", encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
    for line in src:
        rec = json.loads(line)
        rec["text"] = clean(rec.get("text", ""))
        dst.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"✅ Saved: {out}")
