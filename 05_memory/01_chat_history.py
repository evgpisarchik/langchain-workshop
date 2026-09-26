import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import sqlite3

    import marimo as mo
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import START, MessagesState, StateGraph

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Conversation memory for chains (no agent needed)

    LLMs are stateless: "memory" means sending previous messages again.
    Three ways, from most manual to most managed:

    1. keep a list of messages yourself
    2. LangGraph checkpointer, in memory (the modern way, also used by agents)
    3. the same with SQLite - the conversation survives restarts

    (`RunnableWithMessageHistory` / `InMemoryChatMessageHistory` are deprecated since
    langchain-core 1.6 in favour of LangGraph persistence.)

    Open: `uv run marimo edit 05_memory/01_chat_history.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Manual message list

    You own the list.
    """)
    return


@app.cell
def _():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a friendly assistant. Keep answers to one sentence."),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    history: list = []
    for _text in ["Hi, I'm Dana and I work as a nurse.", "What's my job?"]:
        _reply = chain.invoke({"history": history, "input": _text})
        history += [HumanMessage(_text), AIMessage(_reply.text)]
    mo.ui.table([{"role": m.type, "text": str(m.text)} for m in history])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. LangGraph `MessagesState` + checkpointer

    A one-node graph + checkpointer. State is persisted per `thread_id`,
    survives across calls, and you can inspect or even edit it.
    """)
    return


@app.function
def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


@app.cell
def _():
    graph = (
        StateGraph(MessagesState)
        .add_node("model", call_model)
        .add_edge(START, "model")
        .compile(checkpointer=InMemorySaver())
    )
    thread = {"configurable": {"thread_id": "t-1"}}
    graph.invoke({"messages": [("human", "Remember the number 4817.")]}, thread)
    _result = graph.invoke({"messages": [("human", "What number did I ask you to remember? Digits only.")]}, thread)
    {"answer": _result["messages"][-1].text, "history length": len(graph.get_state(thread).values["messages"])}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Persistent memory with `SqliteSaver`

    `SqliteSaver` writes checkpoints to a file (`PostgresSaver` for prod).
    Re-run the cell (or restart the notebook) - each run remembers the previous ones.
    Delete `05_memory/memory.sqlite` to reset.
    """)
    return


@app.cell
def _():
    db_path = mo.notebook_dir() / "memory.sqlite"
    with sqlite3.connect(db_path, check_same_thread=False) as _conn:
        _durable = (
            StateGraph(MessagesState)
            .add_node("model", call_model)
            .add_edge(START, "model")
            .compile(checkpointer=SqliteSaver(_conn))
        )
        _thread = {"configurable": {"thread_id": "persistent-user"}}
        _previous = _durable.get_state(_thread).values.get("messages", [])
        _words = ["apple", "river", "falcon", "copper", "violet", "tundra"]
        _word = _words[(len(_previous) // 2) % len(_words)]
        _result = _durable.invoke(
            {"messages": [("human", f"New secret word: {_word}. List every secret word I have told you so far.")]},
            _thread,
        )
    {"messages from earlier runs": len(_previous), "new word": _word, "answer": _result["messages"][-1].text}
    return


if __name__ == "__main__":
    app.run()
