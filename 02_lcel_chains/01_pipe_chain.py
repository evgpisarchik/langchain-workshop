import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    from shared import get_llm, stream_text

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # LCEL: compose Runnables with `|`

    `prompt | model | parser` is itself a Runnable, so it gets `invoke / batch / stream`
    / async for free.

    Open: `uv run marimo edit 02_lcel_chains/01_pipe_chain.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `prompt | llm | StrOutputParser()`

    The canonical chain. The parser turns `AIMessage` -> `str`.
    """)
    return


@app.cell
def _():
    prompt = ChatPromptTemplate.from_template("Explain {concept} to a 10-year-old in two sentences.")
    chain = prompt | llm | StrOutputParser()
    chain.invoke({"concept": "recursion"})
    return (chain,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `batch()` and `stream()` on a chain

    The same chain supports batch and stream with no extra code.
    """)
    return


@app.cell
def _(chain):
    concepts = ["gravity", "DNA"]
    mo.ui.table(
        [{"concept": c, "explanation": t} for c, t in zip(concepts, chain.batch([{"concept": c} for c in concepts]))]
    )
    return


@app.cell
def _(chain):
    stream_text(chain.stream({"concept": "photosynthesis"}))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Custom steps with `RunnableLambda`

    Any Python function becomes a step with `RunnableLambda` (or just a plain
    function/lambda inside a `|` chain - it is wrapped automatically).
    """)
    return


@app.cell
def _(chain):
    shout = RunnableLambda(lambda text: text.upper())
    count_words = lambda text: {"text": text, "words": len(text.split())}  # noqa: E731
    chain_with_steps = chain | shout | count_words
    chain_with_steps.invoke({"concept": "a rainbow"})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Introspection

    Chains are data: inspect the input schema and the graph.
    """)
    return


@app.cell
def _(chain):
    mo.vstack([chain.get_input_jsonschema(), mo.mermaid(chain.get_graph().draw_mermaid())])
    return


if __name__ == "__main__":
    app.run()
