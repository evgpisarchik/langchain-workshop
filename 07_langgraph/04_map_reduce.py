import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import operator
    from typing import Annotated, TypedDict

    import marimo as mo
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Map-reduce with the `Send` API: dynamic fan-out over a list

    The number of parallel branches is decided at runtime. Each `Send(node, state)`
    launches one worker with its own input; results are merged by a reducer.

    Use case: summarize several documents in parallel, then combine the summaries.

    Open: `uv run marimo edit 07_langgraph/04_map_reduce.py`
    """)
    return


@app.class_definition
class OverallState(TypedDict):
    documents: dict[str, str]
    summaries: Annotated[list[str], operator.add]  # each worker appends one summary
    final: str


@app.class_definition
class WorkerState(TypedDict):
    name: str
    text: str


@app.function
def fan_out(state: OverallState) -> list[Send]:
    return [Send("summarize", {"name": name, "text": text}) for name, text in state["documents"].items()]


@app.function
def summarize(state: WorkerState) -> dict:
    summary = llm.invoke(f"Summarize in one sentence:\n\n{state['text']}").text
    print(f"summarized {state['name']}")
    return {"summaries": [f"{state['name']}: {summary}"]}


@app.function
def combine(state: OverallState) -> dict:
    joined = "\n".join(state["summaries"])
    final = llm.invoke(
        f"Write a 3-sentence onboarding intro for a new employee based on these summaries:\n{joined}"
    ).text
    return {"final": final}


@app.cell
def _():
    graph = (
        StateGraph(OverallState)
        .add_node("summarize", summarize)
        .add_node("combine", combine)
        .add_conditional_edges(START, fan_out, ["summarize"])  # returns a list of Send objects
        .add_edge("summarize", "combine")
        .add_edge("combine", END)
        .compile()
    )
    mo.mermaid(graph.get_graph().draw_mermaid())
    return (graph,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Map: all documents in parallel -> Reduce
    """)
    return


@app.cell
def _(graph):
    docs_dir = mo.notebook_dir().parent / "06_rag" / "data"
    documents = {p.name: p.read_text() for p in sorted(docs_dir.glob("*.md"))}
    result = graph.invoke({"documents": documents, "summaries": []})
    mo.vstack([
        mo.md("\n".join(f"- {s}" for s in result["summaries"])),
        mo.md(f"**Onboarding intro:** {result['final']}"),
    ])
    return


if __name__ == "__main__":
    app.run()
