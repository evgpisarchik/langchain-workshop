import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from dataclasses import dataclass
    from typing import Any

    import marimo as mo
    from langchain.agents import AgentState, create_agent
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ModelRequest,
        PIIMiddleware,
        SummarizationMiddleware,
        before_model,
        dynamic_prompt,
        wrap_tool_call,
    )
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.runtime import Runtime

    from shared import get_llm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Agent middleware (LangChain 1.x): hook into every step of the agent loop

    Middleware lets you change prompts, trim history, guard against PII, cap costs,
    log, retry or rewrite tool calls - without touching the agent's core logic.
    Hook logs appear in each cell's console output.

    Open: `uv run marimo edit 04_tools_and_agents/05_middleware.py`
    """)
    return


@app.cell
def _():
    @tool
    def lookup_customer(email: str) -> dict:
        """Look up a customer record by email."""
        return {"email": email, "name": "Jane Doe", "tier": "gold", "open_tickets": 2}

    @tool
    def flaky_inventory(product: str) -> str:
        """Check warehouse stock for a product."""
        raise TimeoutError("inventory service timed out")

    return flaky_inventory, lookup_customer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Middleware definitions

    1. **Dynamic system prompt** - computed per request from the runtime context (e.g. user role).
    2. **Custom hooks** - `before_model` logs every model call; `wrap_tool_call` turns tool
       exceptions into `ToolMessage`s so the model can react instead of crashing.
    3. **Built-ins** - `ModelCallLimitMiddleware` stops runaway loops, `PIIMiddleware` redacts e-mails in user input.
    """)
    return


@app.class_definition
@dataclass
class Ctx:
    role: str  # "customer" | "admin"
    language: str


@app.cell
def _(flaky_inventory, lookup_customer):
    @dynamic_prompt
    def role_based_prompt(request: ModelRequest) -> str:
        ctx: Ctx = request.runtime.context
        base = f"You are a support assistant. Always answer in {ctx.language}."
        if ctx.role == "admin":
            return base + " The user is an admin: you may reveal internal fields like tier."
        return base + " Never reveal internal fields like customer tier."

    @before_model
    def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[before_model] sending {len(state['messages'])} messages to the model")
        return None  # returning a dict would update the state

    @wrap_tool_call
    def tool_errors_to_messages(request, handler):
        """Turn tool exceptions into ToolMessages so the model can react instead of crashing."""
        try:
            return handler(request)
        except Exception as error:  # noqa: BLE001
            print(f"[wrap_tool_call] {request.tool_call['name']} failed: {error}")
            return ToolMessage(
                content=f"Tool failed: {error}. Tell the user to try again later.",
                tool_call_id=request.tool_call["id"],
            )

    agent = create_agent(
        get_llm(),
        tools=[lookup_customer, flaky_inventory],
        context_schema=Ctx,
        middleware=[
            role_based_prompt,
            log_before_model,
            tool_errors_to_messages,
            ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
        ],
    )
    return (agent,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Dynamic prompt: customer vs admin

    `PIIMiddleware` rewrote the e-mail in the user message before the model saw it.
    """)
    return


@app.cell
def _(agent):
    question = {"messages": [{"role": "user", "content": "Look up customer jane@example.com and tell me their tier."}]}
    customer_result = agent.invoke(question, context=Ctx(role="customer", language="German"))
    admin_result = agent.invoke(question, context=Ctx(role="admin", language="English"))
    {
        "customer (de)": customer_result["messages"][-1].text,
        "admin (en)": admin_result["messages"][-1].text,
        "user message as stored": admin_result["messages"][0].text,
        "tool calls": [c["args"] for m in admin_result["messages"] for c in getattr(m, "tool_calls", [])],
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Tool error handled by `wrap_tool_call`
    """)
    return


@app.cell
def _(agent):
    agent.invoke(
        {"messages": [{"role": "user", "content": "Is the Rover R2 in stock?"}]},
        context=Ctx(role="customer", language="English"),
    )["messages"][-1].text
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `SummarizationMiddleware`

    When history grows past the trigger, old messages are replaced by a summary.
    """)
    return


@app.cell
def _():
    chat = create_agent(
        get_llm(),
        tools=[],
        checkpointer=InMemorySaver(),
        middleware=[SummarizationMiddleware(model=get_llm(), trigger=("messages", 6), keep=("messages", 2))],
    )
    chat_config = {"configurable": {"thread_id": "long-chat"}}
    turns = []
    for _text in [
        "My name is Ivan and I'm planning a trip to Japan.",
        "I want to visit Kyoto and Osaka, 10 days in April.",
        "My budget is 3000 EUR without flights.",
        "Remind me: what's my name, destination and budget?",
    ]:
        _answer = chat.invoke({"messages": [{"role": "user", "content": _text}]}, chat_config)["messages"][-1].text
        _stored = chat.get_state(chat_config).values["messages"]
        turns.append({"user": _text, "stored messages": len(_stored), "answer": _answer})
    mo.vstack([
        mo.ui.table(turns),
        mo.md("**First stored message is now a summary:**"),
        mo.md(chat.get_state(chat_config).values["messages"][0].text),
    ])
    return


if __name__ == "__main__":
    app.run()
