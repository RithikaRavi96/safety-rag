print("Starting RAG test...")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate


emb = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.load_local(
    "indexes/safety_faiss",
    embeddings=emb,
    allow_dangerous_deserialization=True,
)
retriever = db.as_retriever(search_kwargs={"k": 5})

template = """
You are a safety & compliance assistant. Answer ONLY using the context.
- If the answer is not in the context, say: "Insufficient context."
- Be concise and factual.
- If available, cite Hazard Key and Function from metadata.

Question: {question}

Context:
{context}

Answer:
"""
PROMPT = PromptTemplate.from_template(template)


def format_context(docs):
    lines = []
    for d in docs:
        meta = d.metadata or {}
        tag = f"[row={meta.get('row')}, key={meta.get('Hazard Key')}, func={meta.get('Function')}]"
        text = d.page_content.replace("\n", " ")
        lines.append(f"{tag} {text}")
    return "\n\n".join(lines)


question = "What countermeasure should be taken for incorrect diagnosis basis due to wrong patient orientation?"


docs = retriever.get_relevant_documents(question)
ctx = format_context(docs)

llm = Ollama(model="llama3.1", temperature=0.1, num_ctx=4096)
prompt = PROMPT.format(question=question, context=ctx)

answer = llm.invoke(prompt)


print("\n=== QUESTION ===")
print(question)

print("\n=== ANSWER ===")
print(answer.strip())

print("\n=== SOURCES (top-k) ===")
for i, d in enumerate(docs, 1):
    meta = d.metadata or {}
    print(f"{i}. row={meta.get('row')}, key={meta.get('Hazard Key')}, func={meta.get('Function')}")
