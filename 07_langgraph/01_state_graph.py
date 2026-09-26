import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import operator
    from typing import Annotated, TypedDict

    import marimo as mo
    from langgraph.graph import END, START, StateGraph

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # LangGraph basics: state, nodes, edges

    LangGraph models an LLM application as a graph:

    * **State** - a typed dict shared by all nodes
    * **Nodes** - functions: state -> partial state update
    * **Edges** - which node runs next

    Reducers decide how updates are merged (overwrite by default, or e.g. append).

    Use case: a content pipeline `outline -> draft -> (headline || tags) -> assemble`.

    Open: `uv run marimo edit 07_langgraph/01_state_graph.py`
    """)
    return


@app.class_definition
class ArticleState(TypedDict):
    topic: str
    outline: str
    draft: str
    headline: str
    tags: list[str]
    log: Annotated[list[str], operator.add]


@app.function
def make_outline(state: ArticleState) -> dict:
    outline = llm.invoke(f"Write a 3-bullet outline for a short blog post about: {state['topic']}").text
    return {"outline": outline, "log": ["outline"]}


@app.function
def write_draft(state: ArticleState) -> dict:
    draft = llm.invoke(f"Write a 120-word blog post following this outline:\n{state['outline']}").text
    return {"draft": draft, "log": ["draft"]}


@app.function
def write_headline(state: ArticleState) -> dict:
    headline = llm.invoke(f"Write one catchy headline for this post. Headline only.\n\n{state['draft']}").text
    return {"headline": headline.strip().strip('"'), "log": ["headline"]}


@app.function
def pick_tags(state: ArticleState) -> dict:
    tags = llm.invoke(f"Give 3 lowercase tags for this post, comma separated, nothing else.\n\n{state['draft']}").text
    return {"tags": [t.strip() for t in tags.split(",")], "log": ["tags"]}


@app.function
def assemble(state: ArticleState) -> dict:
    return {"log": ["assemble"]}


@app.cell
def _():
    builder = StateGraph(ArticleState)
    builder.add_node("outline", make_outline)
    builder.add_node("draft", write_draft)
    builder.add_node("headline", write_headline)
    builder.add_node("tags", pick_tags)
    builder.add_node("assemble", assemble)

    builder.add_edge(START, "outline")
    builder.add_edge("outline", "draft")
    builder.add_edge("draft", "headline")  # fan-out: headline and tags run in parallel
    builder.add_edge("draft", "tags")
    builder.add_edge(["headline", "tags"], "assemble")  # fan-in: wait for both
    builder.add_edge("assemble", END)
    graph = builder.compile()
    mo.mermaid(graph.get_graph().draw_mermaid())
    return (graph,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `stream_mode="updates"` - see each node finish
    """)
    return


@app.cell
def _(graph):
    for _update in graph.stream(
        {"topic": "why developers should write tests first", "log": []}, stream_mode="updates"
    ):
        for _node, _change in _update.items():
            mo.output.append(mo.md(f"**[{_node}]** updated keys: `{list(_change)}`"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `invoke()` - final state
    """)
    return


@app.cell
def _():
    topic = mo.ui.text(value="the joy of mechanical keyboards", label="Topic", full_width=True)
    topic
    return (topic,)


@app.cell
def _(graph, topic):
    final = graph.invoke({"topic": topic.value, "log": []})
    mo.vstack([
        mo.md(f"## {final['headline']}\n\n{final['draft']}"),
        {"tags": final["tags"], "execution log (appended by reducer)": final["log"]},
    ])
    return


if __name__ == "__main__":
    app.run()
