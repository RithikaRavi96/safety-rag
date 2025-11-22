from langchain.embeddings import OllamaEmbeddings

emb = OllamaEmbeddings(model="nomic-embed-text")

vec = emb.embed_query("hazard controls for CT equipment")

print("Vector dimension:", len(vec))
print("First 10 values:", vec[:10])

