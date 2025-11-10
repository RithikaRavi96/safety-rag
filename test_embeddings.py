from langchain.embeddings import OllamaEmbeddings

# Use the embedding model from Ollama
emb = OllamaEmbeddings(model="nomic-embed-text")

# Test with a sample query
vec = emb.embed_query("hazard controls for CT equipment")

print("Vector dimension:", len(vec))
print("First 10 values:", vec[:10])

