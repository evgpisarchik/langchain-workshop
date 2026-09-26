import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import uuid
    from typing import TypedDict

    import marimo as mo
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Persistence superpowers: `interrupt()` for human input, and time travel

    With a checkpointer, every step is saved. That enables:

    * `interrupt()`: pause inside a node, wait for a human, resume with `Command(resume=...)`
    * `get_state_history()`: inspect every past step
    * forking: resume from an old checkpoint with modified state

    Each cell below depends on the previous one, so re-running a step re-runs the rest of the story.

    Open: `uv run marimo edit 07_langgraph/03_interrupt_and_time_travel.py`
    """)
    return


@app.class_definition
class State(TypedDict):
    request: str
    reply: str
    approved: bool


@app.function
def draft_reply(state: State) -> dict:
    reply = llm.invoke(
        f"Write a 2-sentence polite reply from support to this customer message:\n{state['request']}"
    ).text
    return {"reply": reply}


@app.function
def human_review(state: State) -> dict:
    # interrupt() stops the graph here and hands this payload to the caller.
    # When resumed, interrupt() returns whatever was passed in Command(resume=...).
    decision = interrupt({"question": "Send this reply?", "reply": state["reply"]})
    if decision["action"] == "edit":
        return {"reply": decision["text"], "approved": True}
    return {"approved": decision["action"] == "approve"}


@app.function
def send(state: State) -> dict:
    status = "SENT" if state["approved"] else "DISCARDED"
    mo.output.append(mo.md(f"**>>> {status}:** {state['reply']}"))
    return {}


@app.cell
def _():
    graph = (
        StateGraph(State)
        .add_node("draft", draft_reply)
        .add_node("review", human_review)
        .add_node("send", send)
        .add_edge(START, "draft")
        .add_edge("draft", "review")
        .add_edge("review", "send")
        .add_edge("send", END)
        .compile(checkpointer=InMemorySaver())
    )
    mo.mermaid(graph.get_graph().draw_mermaid())
    return (graph,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Run until the interrupt
    """)
    return


@app.cell
def _(graph):
    config = {"configurable": {"thread_id": f"ticket-{uuid.uuid4().hex[:6]}"}}
    paused = graph.invoke({"request": "My robot arrived with a scratched cover. Can I get a discount?"}, config)
    {"paused with": paused["__interrupt__"][0].value, "next node": graph.get_state(config).next}
    return (config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Resume with a human decision (edit)
    """)
    return


@app.cell
def _(config, graph):
    resumed = graph.invoke(
        Command(resume={"action": "edit", "text": "Sorry about the scratch! We'll send a free replacement cover today."}),
        config,
    )
    return (resumed,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Time travel: list checkpoints
    """)
    return


@app.cell
def _(config, graph, resumed):
    history = list(graph.get_state_history(config)) if resumed else []
    mo.ui.table(
        [
            {"step": s.metadata.get("step"), "next": s.next, "reply": str(s.values.get("reply"))[:80]}
            for s in history
        ]
    )
    return (history,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Fork: go back to before review, change the draft, decide differently

    `update_state` on an old checkpoint creates a new branch (the original history is kept).
    """)
    return


@app.cell
def _(config, graph, history):
    before_review = next(s for s in history if s.next == ("review",))
    fork_config = graph.update_state(before_review.config, {"reply": "Sorry! Use code SCRATCH10 for 10% off."})
    fork_paused = graph.invoke(None, fork_config)  # continue the branch -> hits the interrupt again
    mo.output.append(mo.md(f"**Paused again with:** {fork_paused['__interrupt__'][0].value['reply']}"))
    graph.invoke(Command(resume={"action": "reject"}), config)  # the thread now points at the fork
    mo.output.append({"checkpoints in thread": len(list(graph.get_state_history(config)))})
    return


if __name__ == "__main__":
    app.run()
