# 📚 RAG Pipeline: Chat with Your PDFs

A small Retrieval-Augmented Generation (RAG) project built while learning how RAG works. It loads PDF research papers, splits them into chunks, embeds them, stores them in a local vector database, and answers questions using an LLM grounded in the retrieved context.

Everything lives in a single Jupyter notebook: `Rag_pipline.ipynb`.

---

## How it works

```
PDFs ──► Load ──► Chunk ──► Embed ──► ChromaDB
                                          │
 Question ──► Embed ──► Top-k search ◄────┘
                              │
                  Context + Question ──► LLM ──► Answer
```

| Stage | What happens | Tool |
|---|---|---|
| **Ingestion** | Every PDF in the `data/` folder is loaded page by page | `PyPDFLoader` (LangChain) |
| **Chunking** | Pages are split into 500-character chunks with 50-character overlap | `RecursiveCharacterTextSplitter` |
| **Embedding** | Chunks are converted to 384-dim vectors | `all-MiniLM-L6-v2` (Sentence-Transformers) |
| **Vector store** | Embeddings, text and metadata are persisted locally | ChromaDB (`data/vector_store`, collection `pdf_documents`) |
| **Retrieval** | The query is embedded and the top-k most similar chunks are returned with similarity scores (`1 - distance`) | `RAGRetriever` class |
| **Generation** | Retrieved chunks are placed in a prompt together with the question and sent to the LLM | `ChatOpenAI` (LangChain) |

### Main classes and functions

- `load_all_pdfs()`: loads all PDFs from `data/` into LangChain `Document` objects
- `split_docs()`: chunks documents (`chunk_size=500`, `chunk_overlap=50`)
- `EmbeddingManager`: wraps the SentenceTransformer model
- `VectorStoreManager`: creates or loads the persistent Chroma collection and adds documents
- `RAGRetriever`: semantic search with optional score threshold
- `generate_output()`: retrieves context, builds the prompt and calls the LLM

---

## Knowledge base

The pipeline was tested on three papers:

1. **Attention Is All You Need** (Vaswani et al., 2017), the original Transformer paper
2. **Retrieval-Augmented Generation for Large Language Models: A Survey** (Gao et al., 2023)
3. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (DeepSeek-AI, 2025)

You can swap in any PDFs you like.

---

## Project structure

```
.
├── Rag_pipline.ipynb      # The full pipeline
├── data/
│   ├── research1.pdf      # Attention Is All You Need
│   ├── research2.pdf      # RAG survey
│   ├── research3.pdf      # DeepSeek-R1
│   └── vector_store/      # ChromaDB persistent store (auto-created)
├── Python.txt             # Sample text file used to try TextLoader
└── README.md
```

> The notebook reads PDFs from a folder named `data/`. Put your PDFs there before running.

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install langchain langchain-core langchain-community langchain-text-splitters \
            langchain-openai pypdf pymupdf sentence-transformers chromadb jupyter
```

### 3. Add your PDFs

Place them in the `data/` folder.

### 4. Set your OpenAI API key

In the notebook the key is currently a placeholder variable:

```python
API_KEY_OPENAI = "ENTER YOUR OPEN API KEY HERE"
```

⚠️ **Never commit a real key to GitHub.** Prefer an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

and in the notebook:

```python
import os
llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model="<your-model-name>",
    temperature=0.1,
    max_tokens=1024,
)
```

### 5. Run the notebook

```bash
jupyter notebook Rag_pipline.ipynb
```

Run the cells top to bottom. The first run builds the vector store; later runs reuse it from `data/vector_store`. Avoid re-running the "add documents" cell on an existing store, since it adds the chunks again as duplicates.

---

## Example usage

```python
# Retrieval only
rag_retriever.retrieve("What is encoder decoder")

# Full RAG answer
answer = generate_output("what is RAG?", rag_retriever, llm)
print(answer)
```

---

## Configuration

| Setting | Where | Default |
|---|---|---|
| Chunk size / overlap | `split_docs()` | `500` / `50` |
| Embedding model | `EmbeddingManager` | `all-MiniLM-L6-v2` |
| Vector store path | `VectorStoreManager` | `data/vector_store` |
| Collection name | `VectorStoreManager` | `pdf_documents` |
| Top-k chunks | `generate_output()` | `3` |
| LLM temperature / max tokens | `ChatOpenAI` | `0.1` / `1024` |

---

## Ideas for improvement

Based on what I learned from the RAG survey paper included in this project:

- [ ] Re-ranking retrieved chunks before generation
- [ ] Query rewriting / expansion (e.g. HyDE, multi-query)
- [ ] Hybrid retrieval (BM25 + dense vectors)
- [ ] Metadata filtering (by paper, page number)
- [ ] Show source citations (file and page) in answers
- [ ] A "not found in context" fallback so the model doesn't hallucinate
- [ ] Evaluation with RAGAS (context relevance, faithfulness, answer relevance)
- [ ] Simple UI with Streamlit or Gradio

---

## Tech stack

Python · LangChain · Sentence-Transformers · ChromaDB · OpenAI API · Jupyter

---

## Acknowledgements

- Vaswani et al., *Attention Is All You Need*, 2017
- Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey*, 2023
- DeepSeek-AI, *DeepSeek-R1*, 2025

---

## License

This project is for learning purposes. Add a license of your choice (e.g. MIT) before publishing.
