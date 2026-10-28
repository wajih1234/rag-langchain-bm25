# RAG Chat: hybrid search (BM25 + vectors)

A chat application to ask questions about your own documents. The user uploads files in the sidebar, the app indexes them, and answers are generated from the content of those files only.

Built with **LangChain**, **Chroma**, **BM25**, **OpenRouter** and **Streamlit**.

## Features

- Upload one or several files (`.pdf`, `.txt`) from the sidebar
- Hybrid retrieval: BM25 (exact keywords) + embeddings (meaning), merged with `EnsembleRetriever`
- Answers grounded in the documents: the model says it does not know when the answer is not in the context
- Chat interface with several conversations, each with its own files
- In-memory Chroma store: every upload is isolated in its own collection

## How it works

```
Upload  ->  Load  ->  Split (1000 / 200)  ->  Embeddings  ->  Chroma
                                    \                              |
                                     \->  BM25 index               |
                                              \                    |
Question  ->  Hybrid retriever (BM25 + vectors, 0.5 / 0.5)  <------/
          ->  Prompt (context + question)  ->  LLM  ->  Answer
```

## Project structure

| File | Role |
|---|---|
| `ingest.py` | Loads `.pdf` and `.txt` files and splits them into chunks |
| `embed.py` | Embedding model and Chroma vector store |
| `retrievers.py` | BM25 retriever and hybrid retriever |
| `llm.py` | LLM connection through OpenRouter |
| `rag.py` | RAG chain: retriever, prompt, LLM, output parser |
| `app.py` | Streamlit interface |
| `data/` | Test documents (optional) |
| `.env` | API key (not committed) |

## Requirements

- Python 3.10 or newer
- An [OpenRouter](https://openrouter.ai) API key

## Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install langchain langchain-community langchain-text-splitters langchain-classic
pip install rank_bm25 pypdf
pip install langchain-chroma langchain-huggingface sentence-transformers
pip install langchain-openai python-dotenv
pip install streamlit
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

## Run

```bash
streamlit run app.py
```

Then drop your files in the sidebar and ask your questions in the chat.

## Configuration

| What | Where | Current value |
|---|---|---|
| LLM | `llm.py` | `meta-llama/llama-3.3-70b-instruct` (temperature 0.1, max 500 tokens) |
| Embedding model | `embed.py` | `sentence-transformers/all-MiniLM-L6-v2` |
| Chunk size / overlap | `ingest.py` | 1000 / 200 |
| Chunks per retriever | `retrievers.py` | `k=4` |
| BM25 / vector weights | `retrievers.py` | 0.5 / 0.5 |

For documents in French, replace the embedding model with a multilingual one.

## Test each part on its own

Each file has a small test at the bottom. Replace the file name in it, then run:

```bash
python ingest.py       # number of chunks
python embed.py        # content of the Chroma store
python retrievers.py   # chunks returned for a question
python llm.py          # test the LLM connection
python rag.py          # full answer from the chain
```

## Limitations

- Conversations are kept in the browser session: a page refresh erases them
- No memory between questions: follow-ups like "tell me more" do not use earlier messages
- Images (`.png`, `.jpg`) are not supported yet (would need OCR, for example Tesseract)
- The BM25 index is rebuilt in memory at each upload
