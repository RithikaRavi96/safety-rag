from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

# load the same embedding model used to build the index
emb = OllamaEmbeddings(model="nomic-embed-text")


db = FAISS.load_local(
    "indexes/safety_faiss",
    embeddings=emb,
    allow_dangerous_deserialization=True,  # required for FAISS load
)


queries = [
    "wrong patient orientation risk and countermeasure",
    "radiation exposure hazard controls",
    "incorrect diagnosis basis",
]

for q in queries:
    print("\n=== QUERY:", q)
    docs = db.similarity_search(q, k=3)  # top-3 results
    for i, d in enumerate(docs, 1):
        meta = d.metadata
        snippet = d.page_content[:220].replace("\n", " ")
        print(f"  {i}. row={meta.get('row')}  key={meta.get('Hazard Key')}  func={meta.get('Function')}")
        print(f"     {snippet}...")
