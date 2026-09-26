import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import math
    from datetime import date

    import marimo as mo
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    from shared import get_llm, stream_text

    EMPLOYEES = {
        "anna": {"name": "Anna Petrova", "team": "Robotics", "office": "Minsk", "vacation_days_left": 12},
        "mike": {"name": "Mike Chen", "team": "Sales", "office": "Berlin", "vacation_days_left": 3},
        "olga": {"name": "Olga Novak", "team": "Robotics", "office": "Warsaw", "vacation_days_left": 20},
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # `create_agent`: a production-ready tool-calling agent in one call

    `create_agent` (LangChain 1.x) builds a LangGraph graph: model node <-> tools node,
    looping until the model answers without tool calls.

    Open: `uv run marimo edit 04_tools_and_agents/03_create_agent.py`
    """)
    return


@app.cell
def _():
    @tool
    def find_employee(name: str) -> dict:
        """Find an employee by first name. Returns team, office and remaining vacation days."""
        return EMPLOYEES.get(name.lower().split()[0], {"error": "employee not found"})

    @tool
    def list_team(team: str) -> list[str]:
        """List the first names of all employees in a team."""
        return [key for key, e in EMPLOYEES.items() if e["team"].lower() == team.lower()]

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression, e.g. '12 * (3 + 4)' or 'sqrt(16)'."""
        allowed = {k: getattr(math, k) for k in ("sqrt", "pow", "ceil", "floor", "log")}
        return str(eval(expression, {"__builtins__": {}}, allowed))  # demo only - never eval untrusted input

    @tool
    def today() -> str:
        """Return today's date (ISO format)."""
        return date.today().isoformat()

    agent = create_agent(
        model=get_llm(),
        tools=[find_employee, list_team, calculator, today],
        system_prompt="You are an HR assistant. Always use tools to get facts; never guess.",
    )
    return (agent,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The agent is a compiled LangGraph graph
    """)
    return


@app.cell
def _(agent):
    mo.mermaid(agent.get_graph().draw_mermaid())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `invoke()`

    Returns the final state; `state["messages"]` has the whole trajectory.
    """)
    return


@app.cell
def _(agent):
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "How many vacation days do people in the Robotics team have in total?"}]}
    )
    mo.vstack([
        mo.ui.table(
            [
                {
                    "type": m.type,
                    "text": m.text[:200],
                    "tool_calls": [f"{c['name']}({c['args']})" for c in getattr(m, "tool_calls", [])],
                }
                for m in result["messages"]
            ],
            label="Trajectory",
        ),
        mo.md(f"**Answer:** {result['messages'][-1].text}"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `stream(stream_mode="updates")`

    See each step as it happens.
    """)
    return


@app.cell
def _():
    agent_question = mo.ui.text(
        value="Which office is Mike in, and what is 17% of his remaining vacation days?",
        label="Question",
        full_width=True,
    )
    agent_question
    return (agent_question,)


@app.cell
def _(agent, agent_question):
    for _update in agent.stream(
        {"messages": [{"role": "user", "content": agent_question.value}]}, stream_mode="updates"
    ):
        for _node, _state in _update.items():
            _last = _state["messages"][-1]
            _calls = getattr(_last, "tool_calls", None)
            _detail = ", ".join(c["name"] for c in _calls) if _calls else _last.text
            mo.output.append(mo.md(f"**[{_node}]** {_detail}"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `stream(stream_mode="messages")` - tokens

    Token-by-token output of the final answer.
    """)
    return


@app.cell
def _(agent):
    stream_text(
        token.text
        for token, metadata in agent.stream(
            {"messages": [{"role": "user", "content": "Introduce Olga in two sentences."}]}, stream_mode="messages"
        )
        if metadata["langgraph_node"] == "model" and token.text
    )
    return


if __name__ == "__main__":
    app.run()
