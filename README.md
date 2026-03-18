# 🛡️ QBE Policy Assistant: Agentic RAG System

An enterprise-grade, Agentic Retrieval-Augmented Generation (RAG) system built to autonomously route, retrieve, and synthesize information from complex QBE's insurance policies and general industry standards.

This project was developed to demonstrate advanced AI engineering techniques, including **Semantic Intent Routing**, **Dual-Pipeline Data Ingestion**, and strict **Hallucination Guardrails**.

## ✨ Key Engineering Features

* **Semantic Intent Routing (LangGraph):** Replaces naive keyword searching with an LLM-driven decision engine. The agent analyzes the semantic intent of the user's prompt and autonomously routes granular contractual queries to the QBE database and broad definitional queries to the General Industry database.
* **Dual-Pipeline Ingestion (PyMuPDF):** * *Hierarchical Parser:* Extracts QBE legal documents while preserving heading paths, clause structures, and page numbers to maintain strict legal context.
* **Semantic Recursive Splitter:**  Processes general industry standards into clean, overlapping knowledge chunks.
* **Zero-Hallucination "Pre-Baked" Citations:** Shifts citation formatting logic to the Python backend. By pre-formatting the exact citation string before passing it to the LLM, the system mathematically prevents the model from hallucinating metadata labels or fabricating document names.
* **Transactional Memory Management:** Implements a dual-state memory architecture in Streamlit. The UI retains a fluid visual chat history, while the agent's internal memory is deterministically wiped after every tool execution to completely eliminate search vector pollution and context bleed.

## 🛠️ Tech Stack

* **Orchestration:** LangChain, LangGraph
* **LLM Inference:** Ollama (Llama 3.1 8B) 
* **Vector Database:** ChromaDB
* **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Data Processing:** PyMuPDF (`fitz`), Regex
* **Frontend UI:** Streamlit


## 🚀 Setup & Installation
1. Create a Virtual Environment

```bash
python -m venv venv
.venv\Scripts\activate
```


2. Install Dependencies

```bash
pip install -r requirements.txt
# Note: Requires numpy<2.0 for compatibility with standard Windows AppLocker policies
```

3. Install and Start Ollama
* Ensure you have Ollama installed and running locally, then pull the required model:

```bash
ollama run llama3.1
```

## ⚙️ Running the Pipeline
Step 1: Ingest the Data
* Extracts text from the PDFs in the data/ folder and creates structured JSON chunks.

```bash
python src/ingestion.py
```

Step 2: Index the Vector Database
* Embeds the chunks and stores them in local ChromaDB collections.

```bash
python src/indexing.py
```
Step 3: Launch the User Interface
* Starts the interactive Agentic RAG chat application.

```bash
streamlit run app.py
```

## 🧪 Quality Assurance & Testing
This system was rigorously tested against common local LLM failure modes:

* Semantic Routing Tests: Verified that broad questions correctly map to general knowledge tools without explicit keyword triggers.
* Context Bleed Tests: Verified that consecutive, unrelated queries do not pollute the retrieval vector space (solved via transactional memory wiping).
* Negative Restraint Tests: Verified that out-of-scope queries (e.g., "Does this cover alien invasions?") result in a firm, accurately cited rejection rather than a hallucinated policy clause.

