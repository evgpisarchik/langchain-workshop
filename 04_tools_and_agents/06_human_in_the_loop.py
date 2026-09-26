import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import uuid

    import marimo as mo
    from langchain.agents import create_agent
    from langchain.agents.middleware import HumanInTheLoopMiddleware
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from shared import get_llm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Human-in-the-loop: pause the agent before sensitive tool calls

    `HumanInTheLoopMiddleware` interrupts the graph when the model wants to call a
    protected tool. A human then approves, edits or rejects the call, and the
    agent resumes from the saved checkpoint.

    Sections 1-3 use scripted decisions; the last section lets **you** decide.

    Open: `uv run marimo edit 04_tools_and_agents/06_human_in_the_loop.py`
    """)
    return


@app.cell
def _():
    @tool
    def read_balance(account: str) -> str:
        """Read the balance of a bank account."""
        return f"{account}: 12400 EUR"

    @tool
    def transfer_money(from_account: str, to_account: str, amount_eur: float) -> str:
        """Transfer money between accounts."""
        return f"Transferred {amount_eur} EUR from {from_account} to {to_account}"

    agent = create_agent(
        get_llm(),
        tools=[read_balance, transfer_money],
        checkpointer=InMemorySaver(),  # required: the paused state must be saved somewhere
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "transfer_money": {"allowed_decisions": ["approve", "edit", "reject"]},
                    "read_balance": False,  # safe - never interrupt
                }
            )
        ],
    )

    def run(request: str, scripted: dict) -> mo.Html:
        """Run the agent, answering every interrupt with the scripted decision."""
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = agent.invoke({"messages": [{"role": "user", "content": request}]}, config)
        log = []
        while "__interrupt__" in result:  # the agent is paused, waiting for us
            actions = result["__interrupt__"][0].value["action_requests"]
            log += [f"- PAUSED: model wants `{a['name']}({a['args']})` -> **{scripted['type']}**" for a in actions]
            result = agent.invoke(Command(resume={"decisions": [scripted] * len(actions)}), config)
        return mo.md("\n".join(log) + f"\n\n**Final answer:** {result['messages'][-1].text}")

    return agent, run


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Approve
    """)
    return


@app.cell
def _(run):
    run(
        "What is the balance of my 'checking' account? If it is above 1000 EUR, "
        "move 500 EUR from 'checking' to 'savings'.",
        {"type": "approve"},
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Edit the arguments before execution
    """)
    return


@app.cell
def _(run):
    run(
        "Transfer 9000 EUR from checking to brokerage.",
        {
            "type": "edit",
            "edited_action": {
                "name": "transfer_money",
                "args": {"from_account": "checking", "to_account": "brokerage", "amount_eur": 900},
            },
        },
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Reject with feedback
    """)
    return


@app.cell
def _(run):
    run(
        "Transfer 5000 EUR from checking to offshore77.",
        {"type": "reject", "message": "Account offshore77 is on a fraud watch list."},
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. You decide

    Send a request; when the agent pauses, review the tool call and submit a decision.
    The paused run lives in `mo.state`, so the decision form resumes the same thread.
    """)
    return


@app.cell
def _():
    get_run, set_run = mo.state(None)  # {"config": ..., "result": ...} of the current run
    hitl_request = mo.ui.text_area(
        value="Check my 'checking' balance, then transfer 2500 EUR from checking to savings.",
        label="Request",
        full_width=True,
    )
    hitl_send = mo.ui.run_button(label="Send to agent")
    mo.vstack([hitl_request, hitl_send])
    return get_run, hitl_request, hitl_send, set_run


@app.cell
def _(agent, hitl_request, hitl_send, set_run):
    mo.stop(not hitl_send.value)
    _config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    set_run(
        {
            "config": _config,
            "result": agent.invoke({"messages": [{"role": "user", "content": hitl_request.value}]}, _config),
        }
    )
    return


@app.cell
def _(get_run):
    current_run = get_run()
    mo.stop(current_run is None, mo.md("_Press **Send to agent** to start._"))
    pending = (
        current_run["result"]["__interrupt__"][0].value["action_requests"]
        if "__interrupt__" in current_run["result"]
        else []
    )
    decision_form = (
        mo.md(
            "{decision}\n\n{amount}\n\n{reason}"
        )
        .batch(
            decision=mo.ui.radio(["approve", "edit", "reject"], value="approve", label="Decision"),
            amount=mo.ui.number(
                value=pending[0]["args"].get("amount_eur", 0) if pending else 0, label="New amount (for edit)"
            ),
            reason=mo.ui.text(value="Not allowed.", label="Reason (for reject)"),
        )
        .form(submit_button_label="Submit decision")
    )
    if pending:
        _view = mo.vstack([
            mo.md("\n".join(f"**PAUSED:** model wants `{a['name']}({a['args']})`" for a in pending)),
            decision_form,
        ])
    else:
        _view = mo.md(f"**Final answer:** {current_run['result']['messages'][-1].text}")
    _view
    return current_run, decision_form, pending


@app.cell
def _(agent, current_run, decision_form, pending, set_run):
    mo.stop(decision_form.value is None)
    _choice = decision_form.value
    if _choice["decision"] == "edit":
        _decisions = [
            {"type": "edit", "edited_action": {"name": a["name"], "args": {**a["args"], "amount_eur": _choice["amount"]}}}
            for a in pending
        ]
    elif _choice["decision"] == "reject":
        _decisions = [{"type": "reject", "message": _choice["reason"]} for _ in pending]
    else:
        _decisions = [{"type": "approve"} for _ in pending]
    set_run(
        {
            "config": current_run["config"],
            "result": agent.invoke(Command(resume={"decisions": _decisions}), current_run["config"]),
        }
    )
    return


if __name__ == "__main__":
    app.run()
