
from pathlib import Path
import pandas as pd

from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_ollama import OllamaEmbeddings  # pip install -U langchain-ollama
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.vectorstores import FAISS


EXCEL_PATH = Path("data") / "sample.xlsx"   # adjust if needed
SHEET = 0
INDEX_DIR = Path("indexes") / "safety_faiss"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBED_MODEL = "nomic-embed-text"


def two(x):
    """Return a zero-padded 2-digit string (e.g., 1 -> '01', 59 -> '59')."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    # handle Excel floats like '3.0'
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(2)

print(f"Reading: {EXCEL_PATH.resolve()} (sheet={SHEET})")
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET)


print("Raw columns from Excel:", list(df.columns))
df.columns = [str(c).strip() for c in df.columns]
print("Normalized columns    :", list(df.columns))

required = {"F", "Function", "HG", "Hazard", "CU", "Cause", "M", "Safety Message", "Hazard Key"}
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Expected columns not found: {missing}. Present columns: {list(df.columns)}"
    )

print("Loaded rows:", len(df))

docs = []
kept_rows = 0
missing_msg_rows = 0

for i, row in df.iterrows():
    # Text fields (what we embed for retrieval)
    function_txt = str(row.get("Function", "")).strip()
    hazard_txt   = str(row.get("Hazard", "")).strip()
    cause_txt    = str(row.get("Cause", "")).strip()
    msg_txt      = str(row.get("Safety Message", "")).strip()  # exact header

    if not msg_txt:
        missing_msg_rows += 1

  
    if not any([function_txt, hazard_txt, cause_txt, msg_txt]):
        continue
    kept_rows += 1

    
    parts = []
    if function_txt: parts.append(f"Function: {function_txt}")
    if hazard_txt:   parts.append(f"Hazard: {hazard_txt}")
    if cause_txt:    parts.append(f"Cause: {cause_txt}")
    if msg_txt:      parts.append(f"Safety Message: {msg_txt}")
    page_content = "\n".join(parts)

  
    F_code = two(row.get("F", ""))
    G_code = two(row.get("HG", ""))
    U_code = two(row.get("CU", ""))
    M_code = two(row.get("M", ""))
    hazard_key = str(row.get("Hazard Key", "")).strip()  # e.g. hz_ga_F01G01U03M03

    
    metadata = {
        "row": int(i),
        "source": EXCEL_PATH.name,
        "function": function_txt,
        "hazard": hazard_txt,
        "cause": cause_txt,
        "safety_message": msg_txt,   # exact text for UI
        "hazard_key": hazard_key,    # sheet key if present (no braces)
        "F_code": F_code,
        "G_code": G_code,
        "U_code": U_code,
        "M_code": M_code,
    }

    docs.append(Document(page_content=page_content, metadata=metadata))

print(f"Documents created: {len(docs)} (kept {kept_rows} non-empty rows)")
print(f"Rows with empty 'Safety Message': {missing_msg_rows}")


splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(docs)
print("Chunks created:", len(chunks))


emb = OllamaEmbeddings(model=EMBED_MODEL)
vectordb = FAISS.from_documents(chunks, emb)


INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
vectordb.save_local(str(INDEX_DIR))
print(f"Index saved to: {INDEX_DIR}")
