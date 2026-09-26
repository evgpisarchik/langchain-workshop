import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import asyncio
    import time

    import marimo as mo

    from shared import astream_text, get_llm, stream_text

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Streaming tokens and the async API

    Every LangChain Runnable (models, prompts, chains, agents) exposes the same
    interface: `invoke / batch / stream` and their async twins `ainvoke / abatch / astream`.

    Open: `uv run marimo edit 01_basics/02_streaming_and_async.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `stream()` - render tokens as they arrive

    `stream()` yields `AIMessageChunk` objects as tokens arrive.
    """)
    return


@app.cell
def _():
    stream_text(chunk.text for chunk in llm.stream("Write a haiku about unit tests."))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Aggregating chunks

    Chunks can be added together to rebuild the full message (with metadata).
    """)
    return


@app.cell
def _():
    full = None
    for _chunk in llm.stream("List 3 colors, comma separated."):
        full = _chunk if full is None else full + _chunk
    {"aggregated": full.text, "usage": full.usage_metadata}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `ainvoke()` + `asyncio.gather`

    Async: run several calls concurrently. marimo cells can `await` directly.
    """)
    return


@app.cell
async def _():
    topics = ["cats", "rockets", "coffee"]
    _start = time.perf_counter()
    facts = await asyncio.gather(*(llm.ainvoke(f"One fun fact about {topic}, one sentence.") for topic in topics))
    _elapsed = time.perf_counter() - _start
    mo.vstack([
        mo.ui.table([{"topic": t, "fact": f.text} for t, f in zip(topics, facts)]),
        mo.md(f"{len(topics)} calls in **{_elapsed:.1f}s** (concurrent)"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. `astream()`
    """)
    return


@app.cell
async def _():
    await astream_text(chunk.text async for chunk in llm.astream("Count from 1 to 10, separated by spaces."))
    return


if __name__ == "__main__":
    app.run()
