from pathlib import Path

try:
    from langchain_ollama import OllamaEmbeddings  # preferred
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.vectorstores import FAISS

INDEX_DIR   = Path("indexes") / "safety_faiss"
EMBED_MODEL = "nomic-embed-text"

def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def exact_match(doc, f, h, c) -> bool:
    md = doc.metadata if isinstance(doc.metadata, dict) else {}
    return (
        norm(md.get("function","")) == norm(f)
        and norm(md.get("hazard","")) == norm(h)
        and norm(md.get("cause","")) == norm(c)
    )

def make_query(function: str, hazard: str, cause: str) -> str:
    return (
        f"Function: {function}\n"
        f"Hazard: {hazard}\n"
        f"Cause: {cause}\n"
        "Return concise safety messages."
    )

def safety_message(doc) -> str:
    md = doc.metadata if isinstance(doc.metadata, dict) else {}
    msg = (md.get("safety_message") or "").strip()
    if msg:
        return msg
    # fallback: show page_content if metadata missing
    return (doc.page_content or "").strip()

def show(doc, i):
    md = doc.metadata
    print(f"\n=== Candidate #{i} ===")
    print("Function  :", md.get("function"))
    print("Hazard    :", md.get("hazard"))
    print("Cause     :", md.get("cause"))
    print("Hazard Key:", md.get("hazard_key"))
    print("Codes     :", md.get("F_code"), md.get("G_code"), md.get("U_code"), md.get("M_code"))
    print("Safety Message :\n", safety_message(doc)[:600])

def verify(function, hazard, cause, k=8):
    print("Loading index:", INDEX_DIR)
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vs = FAISS.load_local(str(INDEX_DIR), embeddings=emb, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": k})

    q = make_query(function, hazard, cause)
    docs = retriever.get_relevant_documents(q)

    if not docs:
        print("\nNo results. Check that you rebuilt the index after editing build_index.py.")
        return

    # 1) Show top-k semantic results
    print(f"\nTop-{min(k,len(docs))} semantic candidates:")
    for i, d in enumerate(docs[:k], 1):
        show(d, i)

    # 2) Try exact metadata match
    exact = [d for d in docs if exact_match(d, function, hazard, cause)]
    if exact:
        print("\n\n✅ Exact metadata match found:")
        for i, d in enumerate(exact[:2], 1):
            show(d, i)
    else:
        print("\n⚠️ No exact metadata match. The UI will fall back to the semantic top candidates.")

if __name__ == "__main__":
    # >>> Put the row you want to test here (copy EXACTLY from Excel) <<<
    F_txt = "archiving"
    H_txt = "Data lost"
    C_txt = "Low-quality storage media are used."

    verify(F_txt, H_txt, C_txt, k=8)
