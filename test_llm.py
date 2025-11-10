print("Starting LLM test...")

from langchain_community.llms import Ollama

llm = Ollama(model="llama3.1", temperature=0.1, num_ctx=4096)

answer = llm.invoke("In one sentence, what are hazard controls in medical devices?")
print("LLM reply:", answer)
