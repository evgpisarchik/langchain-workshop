import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    from shared import get_llm

    STOCK = {"rover r2": 14, "docking station": 40, "battery pack": 3}
    PRICES = {"rover r2": 38000, "docking station": 2500, "battery pack": 1200}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Multi-agent system: a supervisor delegating to specialist agents

    Pattern "agents as tools": each specialist is a full `create_agent` with its own
    prompt and tools. The supervisor sees them as tools, decides who to call (maybe
    several, maybe repeatedly) and combines their answers.

    Use case: a sales assistant that needs both inventory data and pricing math.

    Open: `uv run marimo edit 07_langgraph/05_multi_agent_supervisor.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Specialist agents
    """)
    return


@app.cell
def _():
    @tool
    def check_stock(product: str) -> str:
        """Units in stock for a product (Rover R2, docking station, battery pack)."""
        return f"{product}: {STOCK.get(product.lower(), 0)} units in stock"

    @tool
    def get_list_price(product: str) -> str:
        """List price in EUR for one unit of a product."""
        return f"{product}: {PRICES.get(product.lower(), 'unknown')} EUR"

    @tool
    def volume_discount(quantity: int) -> str:
        """Discount percentage for an order of the given quantity."""
        pct = 0 if quantity < 10 else 5 if quantity < 20 else 10
        return f"{pct}% discount for {quantity} units"

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return str(eval(expression, {"__builtins__": {}}, {}))  # demo only

    inventory_agent = create_agent(
        get_llm(),
        tools=[check_stock],
        system_prompt="You are the warehouse agent. Report stock levels exactly as the tool returns them.",
        name="inventory_agent",
    )
    pricing_agent = create_agent(
        get_llm(),
        tools=[get_list_price, volume_discount, calculator],
        system_prompt="You are the pricing agent. Compute exact totals with the calculator. Show the calculation.",
        name="pricing_agent",
    )
    return inventory_agent, pricing_agent


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Supervisor

    Each agent is wrapped as a tool. Only the final answer goes back to the supervisor,
    which keeps its context small. The tools also record each delegation in `delegations`
    (tools run in worker threads, so they can't write to the cell output directly).
    """)
    return


@app.cell
def _(inventory_agent, pricing_agent):
    delegations: list[dict] = []

    @tool
    def ask_inventory(question: str) -> str:
        """Ask the warehouse agent about stock availability."""
        result = inventory_agent.invoke({"messages": [{"role": "user", "content": question}]})
        delegations.append({"agent": "inventory_agent", "question": question, "answer": str(result["messages"][-1].text)})
        return result["messages"][-1].text

    @tool
    def ask_pricing(question: str) -> str:
        """Ask the pricing agent for prices, discounts and order totals."""
        result = pricing_agent.invoke({"messages": [{"role": "user", "content": question}]})
        delegations.append({"agent": "pricing_agent", "question": question, "answer": str(result["messages"][-1].text)})
        return result["messages"][-1].text

    supervisor = create_agent(
        get_llm(),
        tools=[ask_inventory, ask_pricing],
        system_prompt=(
            "You are a sales assistant supervising two specialists. Delegate stock questions to ask_inventory "
            "and price questions to ask_pricing. Never compute prices or guess stock yourself. "
            "Combine their answers into a short reply for the customer."
        ),
    )
    return delegations, supervisor


@app.cell
def _():
    customer_question = mo.ui.dropdown(
        [
            "Can you deliver 12 Rover R2 units, and what would the total cost be including any discount?",
            "I need 5 battery packs. Do you have them?",
        ],
        value="Can you deliver 12 Rover R2 units, and what would the total cost be including any discount?",
        label="Customer question",
        full_width=True,
    )
    customer_question
    return (customer_question,)


@app.cell
def _(customer_question, delegations: list[dict], supervisor):
    delegations.clear()
    _result = supervisor.invoke({"messages": [{"role": "user", "content": customer_question.value}]})
    mo.vstack([
        mo.ui.table(list(delegations), label="Delegations"),
        mo.md(f"**Supervisor:** {_result['messages'][-1].text}"),
    ])
    return


if __name__ == "__main__":
    app.run()
