# 06 · RAG (retrieval-augmented generation)

Answer questions from your own documents: split them into chunks, embed and index the chunks,
retrieve the relevant ones per question, and have the LLM answer from them with citations.

Data: [data/](data/) holds a small handbook for a fictional company, "Northwind Robotics"
(vacation, expenses, IT security, product FAQ).

| File | Covers |
|------|--------|
| [01_load_and_split.py](01_load_and_split.py) | A custom `BaseLoader` · `RecursiveCharacterTextSplitter` · `MarkdownHeaderTextSplitter` (headers → metadata) (no LLM calls; sliders for chunk size and overlap) |
| [02_vector_store.py](02_vector_store.py) | Embeddings · `InMemoryVectorStore` · scores · metadata filters · retrievers: similarity vs MMR · add/delete |
| [03_rag_chain.py](03_rag_chain.py) | Classic LCEL RAG chain · "I don't know" when the answer is missing · returning sources · structured answer with cited chunk ids · streaming |
| [04_advanced_retrieval.py](04_advanced_retrieval.py) | Multi-query retrieval · conversational RAG with history-aware query rewriting |
| [05_agentic_rag.py](05_agentic_rag.py) | Retriever as an agent tool: the agent decides when and what to search, filters by document, combines with a calculator |
| [rag_common.py](rag_common.py) | Shared loading, splitting, indexing and `format_docs` used by 02–05 |

## Embeddings

`get_embeddings()` returns `RemoteEmbeddings`: a multilingual embedding model (1024 dimensions) served by an inference
service at `EMBEDDINGS_URL`. It's multilingual, so a Russian question can find English text.
The gateway's own `/embeddings` endpoint is broken (HTTP 500), so it isn't used.

The embeddings service may be on a private network. Without VPN, `get_embeddings()` prints a notice and falls
back to `HashingEmbeddings`, a local word-and-trigram hasher that matches **words, not meaning**.

To use another embedding provider, return it from `get_embeddings()` (e.g. `OllamaEmbeddings`,
`HuggingFaceEmbeddings`). For production, swap `InMemoryVectorStore` for pgvector, Qdrant, Chroma, etc.;
they share the same interface.
