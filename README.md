# RAG System

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline exposed through a **FastAPI** service. It ingests a local collection of PDF papers, indexes them with a FAISS vector store, and answers natural-language questions by retrieving the most relevant passages before generating a response with a local LLM.

## How it works

1. **Document loading** – PDFs in `data/papers/` are loaded with LangChain's `PyPDFLoader`.
2. **Chunking** – Documents are split into overlapping chunks (`chunk_size=1000`, `chunk_overlap=200`) using `RecursiveCharacterTextSplitter`, so the retriever works with bite-sized, searchable pieces.
3. **Embedding & indexing** – Each chunk is embedded with the `sentence-transformers/all-mpnet-base-v2` model and stored in a **FAISS** vector index for fast similarity search.
4. **Retrieval** – On a query, the top-3 most similar chunks are retrieved (`search_type="similarity"`, `k=3`).
5. **Generation** – The retrieved context and the question are combined into a prompt and passed to a local **TinyLlama-1.1B-Chat** model (via a Hugging Face `pipeline`) to produce the final answer.

## Tech stack

- **FastAPI** + **Uvicorn** – REST API server
- **LangChain** (`langchain`, `langchain-community`, `langchain-text-splitters`) – document loading, splitting, and RAG orchestration
- **FAISS** (`faiss-cpu`) – vector similarity search
- **Sentence-Transformers** – embedding model (`all-mpnet-base-v2`)
- **Hugging Face Transformers** – local LLM inference (TinyLlama-1.1B-Chat)
- **Docker** – containerized deployment

## Project structure

```
rag_system/
├── app/
│   ├── main.py         # FastAPI app entry point
│   ├── endpoints.py     # API routes (/query/)
│   └── rag.py            # RAG pipeline: loading, chunking, embedding, retrieval, generation
├── data/
│   ├── papers/           # PDF source documents (indexed at startup)
│   └── text.txt
├── Dockerfile
├── requirements.txt
└── README.md
```

## Getting started

### Prerequisites

- Python 3.11+
- ~2 GB of free disk space (for the embedding and LLM model weights, downloaded on first run)

### Option 1 – Run locally

```bash
# Clone the repository
git clone https://github.com/lniango/rag_system.git
cd rag_system

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) add your own API keys / tokens
echo "OPENAI_API_KEY=your_key_here" > .env
echo "HF_TOKEN=your_token_here" >> .env

# Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

### Option 2 – Run with Docker

```bash
docker build -t rag-system .
docker run -p 8000:8000 rag-system
```

## Usage

Send a GET request to the `/query/` endpoint with your question:

```bash
curl "http://localhost:8000/query/?query=What are the differences between DiT and LaVin-DiT?"
```

Example response:

```json
{
  "query": "What are the differences between DiT and LaVin-DiT?",
  "response": "..."
}
```

Other example questions you can try (based on the diffusion-model papers included in `data/papers/`):
- *What datasets are used by these models?*
- *How does 3D RoPE work?*

## Notes & limitations

- The vector index is rebuilt **in memory** from the PDFs in `data/papers/` every time the app starts (no persistence between runs).
- The default generator is a small local model (TinyLlama-1.1B), chosen for CPU-friendly experimentation rather than answer quality. Swapping in `OpenAI`/`Ollama` (already imported in `rag.py`) is straightforward if you want stronger generation.
- `OPENAI_API_KEY` and `HF_TOKEN` are optional depending on which embedding/LLM backend you use.

## Possible improvements

- Persist the FAISS index to disk to avoid re-embedding documents on every restart
- Add a `/documents/` endpoint to upload new PDFs dynamically
- Add unit tests for chunking, retrieval, and the API layer
- Swap TinyLlama for a stronger local or API-based LLM for production use


