# DocMind — Ask your PDF

A RAG (Retrieval-Augmented Generation) pipeline built from scratch in Python. Upload any PDF and ask questions — get grounded answers with source citations.

No LangChain. No shortcuts. Every component built manually.

## How it works

1. **Ingest** — PDF is loaded, split into overlapping chunks, embedded via OpenAI, and stored in ChromaDB
2. **Retrieve** — user's question is embedded, top-8 similar chunks fetched from ChromaDB
3. **Rerank** — Cohere reranker scores all 8 chunks, keeps top-3 most relevant
4. **Generate** — GPT-4o-mini answers using only the retrieved chunks as context
5. **Cite** — sources shown alongside every answer

## Tech Stack

| Library | Purpose |
|---|---|
| PyMuPDF | PDF text extraction |
| OpenAI text-embedding-3-small | Convert text chunks to vectors |
| ChromaDB | Local vector database |
| Cohere rerank-v3.5 | Two-stage reranking |
| GPT-4o-mini | Grounded answer generation |
| Streamlit | Web UI |

## Project Structure

```
docmind/
├── data/              # drop your PDFs here
├── ingest.py          # Phase 1: load, chunk, embed, store
├── query.py           # Phase 2: embed query, retrieve, rerank, generate
├── app.py             # Streamlit UI
├── eval.py            # evaluation set — 10 Q&A pairs, scored 10/10
└── requirements.txt
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Priyanshu312003/docmind.git
cd docmind

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
# Create a .env file with:
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key

# 5. Run the app
streamlit run app.py
```

## What I learned

- Chunk size directly affects retrieval quality — tuned from 500 → 250 words after measuring failures
- Cosine similarity and reranking solve different problems — retrieval vs relevance
- Eval sets matter — built 10 Q&A pairs to measure pipeline quality objectively instead of guessing
- Prompt engineering can fix generation failures without touching retrieval at all