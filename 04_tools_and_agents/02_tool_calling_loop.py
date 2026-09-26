import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool

    from shared import get_llm

    ORDERS = {
        "A-100": {"status": "shipped", "carrier": "DHL", "eta": "2026-10-02"},
        "A-101": {"status": "processing", "carrier": None, "eta": None},
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Tool calling by hand: `bind_tools` + your own execution loop

    This is what an agent does internally. Knowing it helps you debug agents and
    build custom control flow when you need it.

    Open: `uv run marimo edit 04_tools_and_agents/02_tool_calling_loop.py`
    """)
    return


@app.cell
def _():
    @tool
    def get_order_status(order_id: str) -> dict:
        """Look up the status of a customer order by its id (format A-123)."""
        return ORDERS.get(order_id, {"error": f"order {order_id} not found"})

    @tool
    def get_carrier_phone(carrier: str) -> str:
        """Get the customer-service phone number of a shipping carrier."""
        return {"DHL": "+49 228 4333112", "UPS": "+1 800 742 5877"}.get(carrier, "unknown carrier")

    tools = [get_order_status, get_carrier_phone]
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = get_llm().bind_tools(tools)
    return get_order_status, llm_with_tools, tools_by_name


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. The model asks for a tool call

    One step: the model returns `tool_calls` instead of (or in addition to) text.
    """)
    return


@app.cell
def _(llm_with_tools):
    first_response = llm_with_tools.invoke("Where is my order A-100?")
    {"text": first_response.text, "tool_calls": first_response.tool_calls}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Manual agent loop

    Call model -> execute requested tools -> append `ToolMessage`s -> repeat.
    """)
    return


@app.cell
def _(llm_with_tools, tools_by_name):
    messages = [
        SystemMessage("You are a customer-support assistant. Use tools to look things up."),
        HumanMessage("Check order A-100 and give me the phone number of its carrier."),
    ]
    trace = []
    for _step in range(5):
        _ai_message = llm_with_tools.invoke(messages)
        messages.append(_ai_message)
        if not _ai_message.tool_calls:
            break
        for _call in _ai_message.tool_calls:
            _result = tools_by_name[_call["name"]].invoke(_call["args"])
            trace.append({"step": _step, "tool": _call["name"], "args": _call["args"], "result": _result})
            messages.append(ToolMessage(content=str(_result), tool_call_id=_call["id"], name=_call["name"]))

    mo.vstack([mo.ui.table(trace, label="Tool calls"), mo.md(f"**Final answer:** {messages[-1].text}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `tool.invoke(tool_call)` -> `ToolMessage`

    Shortcut: invoking a tool with the whole `ToolCall` returns a ready `ToolMessage`.
    """)
    return


@app.cell
def _(get_order_status, llm_with_tools):
    order_call = llm_with_tools.invoke("Status of order A-101?").tool_calls[0]
    get_order_status.invoke(order_call)
    return


if __name__ == "__main__":
    app.run()
