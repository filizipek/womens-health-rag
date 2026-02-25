🧬 Women’s Health RAG — Corpus Starter (Ollama + FAISS)
This repository is a starter scaffold for building a Women’s Health RAG system. It focuses on a clean, interview-friendly pipeline designed for local execution to ensure medical data privacy and auditability.

🏷️ Tagging Philosophy
We preserve a consistent tag structure across all documents and chunks to enable granular filtering during retrieval:

Tier: tier:A_patient_info (General triage/layperson info) vs tier:B_peer_reviewed (Clinical evidence).

Source: source:NHS, source:MedlinePlus, source:Journal, …

Type: type:patient_page, type:guideline_page, type:journal_article, …

Optional: topic:*, sx:* (symptoms), ctx:* (context), possible:*, triage:*

📁 Folder Layout
Organize your workspace as follows:

🚀 Local Setup (Ollama)
This project is Ollama-only for generation. No third-party API keys are required.

Install Ollama: Download from .

Pull the Model: ```bash
ollama pull llama3.1:8b

Start the Service:

🛠️ Workflow
1. Curate & Validate
Fill corpus/metadata/manifest.csv with your sources. Ensure tags follow tag_vocab.yaml.

2. Collect (Download Snapshots)
Behavior: Automatically detects PDFs via extension or content; otherwise saves raw HTML.

Integrity: Records SHA-256 hashes and timestamps.

3. Extract & Chunk
Prepare the text for the vector database.

4. Index & Retrieve
Generate the FAISS index and test the search relevance:

5. Generate Answers (RAG)
Inject the retrieved context into the Ollama local model:

🌐 API + Swagger Interface
Optionally serve the system as a local microservice:

Run this command:
touch scripts/__init__.py
python3 -m uvicorn scripts.api:app --reload --port 8000 --host 127.0.0.1

Access the Swagger UI at: http://127.0.0.1:8000/docs

⚠️ Medical Safety Notes
This project is an educational prototype and must not be treated as medical advice.

Safety Policy: Prefer Tier A for patient-facing guidance and triage cues. Use Tier B as background evidence only.

Generator Logic: Answers must be non-diagnostic, cite sources, and include urgent-care guidance for severe symptoms.

⚙️ Reproducibility
No API keys required.

Configurable ENV:

OLLAMA_MODEL: Default llama3.1:8b

OLLAMA_URL: Default http://127.0.0.1:11434/api/generate