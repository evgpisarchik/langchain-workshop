import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore
    from rag_common import load_documents, split_documents

    from shared import get_embeddings


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # RAG step 2: embed chunks, store them in a vector store, search

    Embeddings map text to vectors; texts with similar meaning get nearby vectors.
    `get_embeddings()` returns a multilingual embedding model behind an HTTP service (see `shared/llm.py`),
    or a local lexical fallback when that service is unreachable. Any `Embeddings`
    class (OpenAI, HuggingFace, Ollama...) can be swapped in without touching this code.

    Open: `uv run marimo edit 06_rag/02_vector_store.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Embeddings
    """)
    return


@app.cell
def _():
    embeddings = get_embeddings()
    vector = embeddings.embed_query("How many vacation days do I get?")
    {"embedder": type(embeddings).__name__, "dimensions": len(vector), "first values": vector[:5]}
    return (embeddings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Build the store

    `InMemoryVectorStore` is perfect for demos. For real apps use Chroma, FAISS,
    pgvector, Qdrant, Elasticsearch... - they share the same `VectorStore` interface.
    """)
    return


@app.cell
def _(embeddings):
    chunks = split_documents(load_documents())
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    {"indexed chunks": len(chunks)}
    return (store,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `similarity_search_with_score`

    Type any question - paraphrases and other languages work with a multilingual model (cross-lingual search).
    """)
    return


@app.cell
def _():
    search_query = mo.ui.text(
        value="How many days of sick leave without a doctor's note?", label="Query", full_width=True
    )
    search_query
    return (search_query,)


@app.cell
def _(search_query, store):
    mo.ui.table(
        [
            {"score": round(score, 3), "source": doc.metadata["source"], "section": doc.metadata["section"],
             "text": doc.page_content}
            for doc, score in store.similarity_search_with_score(search_query.value, k=3)
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Semantic and cross-lingual search (paraphrases, other languages)
    """)
    return


@app.cell
def _(store):
    mo.ui.table(
        [
            {"query": q, "top hit": f"{top.metadata['source']} > {top.metadata['section']}"}
            for q in [
                "Can I be reimbursed for wine with dinner?",  # no shared keywords with "Alcohol is not reimbursable"
                "My kid is about to be born, what time off do I get?",
                "Сколько дней отпуска положено при рождении ребёнка?",  # Russian question, English docs
            ]
            for top in store.similarity_search(q, k=1)
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Metadata filtering

    `InMemoryVectorStore` takes a Python predicate; other stores take a dict filter.
    """)
    return


@app.cell
def _(store):
    only_it = store.similarity_search(
        "What should I do if I lose my laptop?", k=2, filter=lambda d: d.metadata["source"] == "it_security.md"
    )
    mo.ui.table([{"section": d.metadata["section"], "text": d.page_content} for d in only_it])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Retrievers: similarity vs MMR

    A retriever is the Runnable interface over a store: `str` in -> `list[Document]` out.
    MMR (maximal marginal relevance) trades a bit of relevance for more diverse results.
    """)
    return


@app.cell
def _(store):
    similarity = store.as_retriever(search_kwargs={"k": 3})
    mmr = store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
    retriever_question = "hotel budget on business trips"
    {
        "similarity": [d.metadata["section"] for d in similarity.invoke(retriever_question)],
        "mmr": [d.metadata["section"] for d in mmr.invoke(retriever_question)],
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Updating the store

    Add / delete documents later.
    """)
    return


@app.cell
def _(store):
    new_ids = store.add_documents(
        [Document("The office cafeteria is open 8:00-16:00.", metadata={"source": "memo", "section": "Cafeteria"})]
    )
    _found = store.similarity_search("When is the cafeteria open?", k=1)[0].page_content
    store.delete(new_ids)
    {"found new doc": _found}
    return


if __name__ == "__main__":
    app.run()
