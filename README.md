# Safety-RAG (Safety Chatbot Prototype)

A lightweight RAG-based prototype that assists with **safety documentation** in regulated healthcare contexts.  
The system retrieves validated safety messages by **Function–Hazard–Cause (FHC)** triplets and generates traceable responses with governance-friendly logs.

> Thesis context: GenAI governance, operational compliance, and traceability for safety documentation.

---

## Key Features
- **Semantic retrieval** over curated safety data (FAISS index)
- **Streamlit UI** for interactive querying and evidence view
- **Audit-friendly**: logs, deterministic pipelines, and export options
- Test scripts for embeddings, retrieval, and model connectivity

---

## Repository Structure
safety-rag/
├─ data/ # Input data (Excel) used to build the index
├─ images/ # Figures for README / thesis
├─ indexes/ # Generated FAISS index (created by build step)
├─ build_index.py # Build FAISS index from data/ Excel
├─ matchref_chat_app.py # Streamlit app
├─ preview_excel.py # Quick data preview helper
├─ test_chat_ollama.py # (Optional) Local LLM chat connectivity
├─ test_embeddings.py # Sanity checks for embedding model
├─ test_llm.py # Model connectivity tests
├─ test_rag.py # End-to-end RAG test
├─ test_retrieval.py # Retrieval-only test
├─ verify_retrieval.py # Determinism / verification checks
└─ .gitignore

yaml
Copy code

> **Note:** The `.venv/` virtual environment and large artifacts are intentionally **not** committed.

---

## 🛠️ Prerequisites
- **Python 3.10+** (3.11 recommended)
- **git**
-  **Ollama** if you want to run local LLM tests: <https://ollama.ai/>

If you use managed APIs (e.g., OpenAI/others), add a `.env` file with your keys:
OPENAI_API_KEY=your_key_here

or other provider keys if your config uses them
yaml
Copy code

---

##  Quick Start

### 1) Clone the repo
```bash
git clone https://github.com/RithikaRavi96/safety-rag.git
cd safety-rag
2) Create & activate a virtual environment
Windows (PowerShell / Git Bash):

bash
Copy code
python -m venv .venv
.venv\Scripts\activate
macOS / Linux:

bash
Copy code
python3 -m venv .venv
source .venv/bin/activate
3) Install dependencies
If you have a requirements.txt, run:

bash
Copy code
pip install -r requirements.txt
If not, typical libs used for this project include:

bash
Copy code
pip install streamlit faiss-cpu pandas numpy scikit-learn pydantic python-dotenv langchain
# add model-specific libs your environment needs (e.g., openai, ollama, sentence-transformers)
4) Prepare data (Excel)
Place your validated safety dataset in ./data/ (e.g., safety_messages.xlsx) with the 5 columns:
Function, Hazard, Cause, Safety Message, Hazard Key.
Each row should contain a complete F–H–C triplet + message.

5) Build the FAISS index
bash
Copy code
python build_index.py
Creates/updates ./indexes/

Prints how many rows were loaded and how many chunks were indexed

6) Run the Streamlit app
bash
Copy code
streamlit run matchref_chat_app.py
Open the app in your browser: http://localhost:8501

 What to Expect in the UI
Input FHC information (or select via controls)

See retrieved references and the generated safety message

View trace logs / evidence for audit and reproducibility

Useful Test Scripts
Run any of the following to validate parts of the pipeline:

bash
Copy code
python preview_excel.py         # Quick preview of the Excel input
python test_embeddings.py       # Checks embedding model and vector shape
python test_retrieval.py        # Retrieval sanity check
python verify_retrieval.py      # Determinism/consistency checks
python test_rag.py              # End-to-end RAG flow test
python test_llm.py              # Verifies LLM connectivity
python test_chat_ollama.py      # Simple chat via Ollama (if installed)
Troubleshooting
Large files / .venv committed by mistake

Ensure .venv/ is in .gitignore

Remove from history if already pushed (or re-init repo as done in setup)

Port already in use

bash
Copy code
streamlit run matchref_chat_app.py --server.port 8502
Excel path errors

Confirm the file exists in ./data/ and the filename in build_index.py matches.

Model keys

If using an API model, ensure a .env (or environment variable) with the proper key is set.

 Data & Privacy
The repository assumes validated, non-sensitive example data.

Do not commit any patient data or regulated content.

 License
Unless otherwise specified by your organization, you can start with MIT:

css
Copy code
MIT License — Copyright (c) 2025 Rithika
See LICENSE file for details.
(Create a LICENSE file if needed.)

Author / Contact
Rithika Ravichandran
For questions or collaboration, please open a GitHub issue or reach out directly.
