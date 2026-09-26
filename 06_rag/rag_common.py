"""Shared helpers for the RAG examples: load the handbook and build a vector store.

Notebooks in this folder import it directly (`from rag_common import ...`) because
both Python and marimo put the notebook's own directory on sys.path.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from shared import get_embeddings

DATA_DIR = Path(__file__).with_name("data")


def load_documents() -> list[Document]:
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})
        for path in sorted(DATA_DIR.glob("*.md"))
    ]


def split_documents(docs: list[Document]) -> list[Document]:
    """Split by markdown headers first, then by size; keep the header path in the text."""
    header_splitter = MarkdownHeaderTextSplitter([("#", "doc_title"), ("##", "section")])
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks: list[Document] = []
    for doc in docs:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata["source"] = Path(doc.metadata["source"]).name
            title = f"{section.metadata.get('doc_title', '')} > {section.metadata.get('section', '')}"
            for piece in size_splitter.split_documents([section]):
                # Prefixing the header path gives each chunk context ("Vacation policy > Sick leave").
                piece.page_content = f"{title}\n{piece.page_content}"
                chunks.append(piece)
    return chunks


def build_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore.from_documents(split_documents(load_documents()), get_embeddings())


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] (source: {d.metadata['source']})\n{d.page_content}" for i, d in enumerate(docs, 1))
