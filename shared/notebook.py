"""Small marimo helpers shared by the notebooks."""

from collections.abc import AsyncIterable, Iterable

import marimo as mo


def stream_text(chunks: Iterable[str]) -> str:
    """Render text chunks live in the current cell's output; return the full text."""
    text = ""
    for chunk in chunks:
        text += chunk
        mo.output.replace(mo.md(text))
    return text


async def astream_text(chunks: AsyncIterable[str]) -> str:
    """Async twin of stream_text."""
    text = ""
    async for chunk in chunks:
        text += chunk
        mo.output.replace(mo.md(text))
    return text
