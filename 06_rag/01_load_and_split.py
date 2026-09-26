import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import Document
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    from rag_common import DATA_DIR, split_documents


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # RAG step 1: load documents and split them into chunks

    Retrieval works on chunks, not whole files: small enough to be specific,
    large enough to carry meaning. No LLM calls in this notebook.

    Open: `uv run marimo edit 06_rag/01_load_and_split.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. A custom loader

    Loaders turn files (or web pages, PDFs, DB rows...) into `Document` objects:
    `page_content` (str) + `metadata` (dict). Ready-made loaders (`PyPDFLoader`,
    `WebBaseLoader`, `CSVLoader`, ...) live in langchain-community - now being sunset -
    and partner packages. Writing your own is a few lines: implement `lazy_load()`.
    """)
    return


@app.class_definition
class MarkdownFolderLoader(BaseLoader):
    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def lazy_load(self):  # yields Documents one by one; .load() collects them into a list
        for path in sorted(self.folder.glob("*.md")):
            yield Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})


@app.cell
def _():
    docs = MarkdownFolderLoader(DATA_DIR).load()
    mo.ui.table([{"file": Path(d.metadata["source"]).name, "chars": len(d.page_content)} for d in docs])
    return (docs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `RecursiveCharacterTextSplitter`

    The default choice. Tries to split on paragraphs, then lines, then sentences,
    then words to respect `chunk_size`. Play with the sliders.
    """)
    return


@app.cell
def _():
    chunk_size = mo.ui.slider(100, 1000, step=50, value=300, label="chunk_size", show_value=True)
    chunk_overlap = mo.ui.slider(0, 200, step=10, value=30, label="chunk_overlap", show_value=True)
    mo.hstack([chunk_size, chunk_overlap], justify="start")
    return chunk_overlap, chunk_size


@app.cell
def _(chunk_overlap, chunk_size, docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size.value, chunk_overlap=chunk_overlap.value)
    chunks = splitter.split_documents(docs)
    mo.vstack([
        mo.md(f"**{len(chunks)} chunks.** Metadata is inherited from the source document."),
        mo.ui.table(
            [{"source": Path(c.metadata["source"]).name, "chars": len(c.page_content), "text": c.page_content} for c in chunks]
        ),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `MarkdownHeaderTextSplitter`

    Structure-aware splitting: headers go into metadata, which you can later use
    for filtering or citations.
    """)
    return


@app.cell
def _(docs):
    md_splitter = MarkdownHeaderTextSplitter([("#", "doc_title"), ("##", "section")])
    sections = md_splitter.split_text(docs[0].page_content)
    mo.ui.table(
        [{"section": s.metadata.get("section", "(intro)"), "text": s.page_content[:120] + "..."} for s in sections]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Combined splitting (`rag_common.split_documents`)

    The pipeline used by the other RAG notebooks: header split -> size split ->
    prefix the header path to each chunk.
    """)
    return


@app.cell
def _(docs):
    combined = split_documents(docs)
    mo.vstack([
        mo.md(f"**{len(combined)} chunks.** Example:"),
        mo.md(f"```\n{combined[3].page_content}\n```"),
        combined[3].metadata,
    ])
    return


if __name__ == "__main__":
    app.run()
