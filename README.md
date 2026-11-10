# 📄 Paperless-ngx RAG Assistant & Auto-Classifier

A production-grade Retrieval-Augmented Generation (RAG) assistant and document intelligence integration for **Paperless-ngx**. This module connects directly to the Paperless-ngx REST API, extracts OCR-processed document text, handles paginated fetching, generates vector embeddings with OpenAI, stores them in ChromaDB with chunk deduplication, and answers questions using ChatOpenAI (`gpt-4o-mini`).

---

## 🏗️ Project Overview & Architecture

### Architecture Diagram

```
┌──────────────────────────┐      REST API (Paginated)      ┌───────────────────────────┐
│                          ├───────────────────────────────►│                           │
│   Paperless-ngx Server   │                                │   paperless_rag.py        │
│   (PostgreSQL + Redis)   │◄───────────────────────────────┤   (RAG Ingestion Engine)  │
│                          │      OCR Content Extraction    │                           │
└──────────────────────────┘                                └─────────────┬─────────────┘
                                                                          │
                                                                          │ Split Chunks & Embed
                                                                          ▼
                                                            ┌───────────────────────────┐
┌──────────────────────────┐      Context-Aware Prompt      │                           │
│   OpenAI API             │◄───────────────────────────────┤   Chroma Vector DB        │
│   (gpt-4o-mini +         │                                │   (./paperless_db)        │
│    text-embedding-3)     ├───────────────────────────────►│                           │
└──────────────────────────┘      Synthesized Answer        └───────────────────────────┘
```

### Key Workflow Steps
1. **Environment Validation**: Validates `PAPERLESS_TOKEN` and `OPENAI_API_KEY` before execution.
2. **Paginated Document Retrieval**: Connects to `{PAPERLESS_URL}/api/documents/` and iterates through all page results via the `next` URL.
3. **Chunk Splitting & Deduplication**: Splits OCR document content using `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 150) and generates deterministic chunk IDs (`doc_{id}_chunk_{index}`) to eliminate duplicate vector embeddings in ChromaDB across re-indexing runs.
4. **Context-Aware QA Chain**: Constructs a LangChain retrieval chain, retrieving the top $k=4$ most relevant chunks and inserting them into an XML safety-delimited system prompt (`<documents_context>`) to insulate against indirect prompt injection.

---

## ⚙️ Setup & Installation Instructions

### Prerequisites
- Python 3.10+
- A running instance of Paperless-ngx (via Docker Compose)
- OpenAI API Key

### Installation

1. Install required dependencies:
```bash
pip install requests langchain-community langchain-openai chromadb python-dotenv langchain-core langchain-classic
```

2. Start Paperless-ngx (if using Docker Compose):
```bash
docker compose up -d
```

3. Obtain your Paperless-ngx API Token from the Paperless-ngx web interface (`http://localhost:8000` -> **Settings** -> **API Tokens**).

---

## 🔑 Environment Variables Required

Create a `.env` file in the root directory with the following configuration:

| Variable Name | Required | Default Value | Description |
|---|---|---|---|
| `PAPERLESS_URL` | Optional | `http://localhost:8000` | Base URL of your running Paperless-ngx instance. |
| `PAPERLESS_TOKEN` | **Yes** | N/A | API Authentication Token generated in Paperless-ngx. |
| `OPENAI_API_KEY` | **Yes** | N/A | OpenAI API Key for embeddings and Chat model execution. |

Example `.env` file:
```env
PAPERLESS_URL=http://localhost:8000
PAPERLESS_TOKEN=your_paperless_api_token_here
OPENAI_API_KEY=sk-proj-your_openai_api_key_here
```

---

## 🧪 How to Run the Test Suite

The project includes a comprehensive unit and integration test suite using Python's built-in `unittest` framework.

### Execute All Tests
Run the test runner from the root directory:
```bash
python3 -m unittest test_paperless_rag.py
```

### Test Coverage Highlights
- **Environment Validation**: Tests for missing or blank `PAPERLESS_TOKEN` and `OPENAI_API_KEY`.
- **API Pagination**: Tests single-page and multi-page paginated responses from Paperless-ngx.
- **Resilience & Fault Tolerance**: Tests handling of HTTP 401/403 errors, invalid JSON payloads, and network timeouts/connection failures.
- **Vector Store Ingestion**: Tests empty document handling and deterministic chunk ID generation.
- **RAG QA Chain**: Mocks retriever and ChatOpenAI chain execution to verify context insertion and answer formatting.

---

## 🛡️ Security Considerations & Vulnerabilities Resolved

The following security and logic vulnerabilities were identified and remediated during code auditing:

| Vulnerability / Issue | Risk Level | Resolution / Remediation |
|---|---|---|
| **Missing API Pagination** | High | Added a `while url:` loop in `fetch_paperless_documents()` to follow the `next` pagination links from Paperless-ngx, ensuring all documents are retrieved instead of only the first 25. |
| **Duplicate Embeddings on Re-indexing** | Medium | Implemented deterministic chunk ID generation (`doc_{source_id}_chunk_{idx}`) passed to `Chroma.from_documents(..., ids=chunk_ids)` to prevent vector database inflation and retrieval skew across multiple runs. |
| **Indirect Prompt Injection** | High | Enclosed retrieved document content within strict `<documents_context>` XML delimiters in the system prompt with explicit instructions telling the LLM to treat document text strictly as reference content and ignore internal commands. |
| **Unbounded HTTP Request Blocking** | Medium | Enforced explicit `timeout=30` on all `requests.get()` calls to prevent process hanging and resource exhaustion during network disruptions. |
| **Unvalidated API Tokens** | Medium | Added runtime validation (`validate_environment()`) to verify token and key availability before making network calls, preventing invalid authorization headers. |
