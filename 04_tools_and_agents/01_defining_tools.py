import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from typing import Annotated, Literal

    import marimo as mo
    from langchain_core.tools import StructuredTool, ToolException, tool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import BaseModel, Field


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Defining tools: functions the model can decide to call

    A tool = name + description + JSON schema of arguments + the Python function.
    The model only sees name/description/schema, so write them for the model.
    This notebook makes no LLM calls - it shows what the model will receive.

    Open: `uv run marimo edit 04_tools_and_agents/01_defining_tools.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `@tool` on a typed function

    The docstring becomes the description, type hints become the argument schema.
    """)
    return


@app.cell
def _():
    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    {"name": multiply.name, "description": multiply.description, "args": multiply.args}
    return (multiply,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Argument descriptions

    `parse_docstring=True` reads Google-style `Args:`; `Annotated` adds per-argument descriptions inline.
    """)
    return


@app.cell
def _():
    @tool(parse_docstring=True)
    def search_products(
        query: str,
        max_price: float | None = None,
        category: Literal["robots", "accessories", "services"] = "robots",
    ) -> list[dict]:
        """Search the product catalog.

        Args:
            query: Free-text search, e.g. "warehouse robot".
            max_price: Optional upper price limit in EUR.
            category: Product category to search in.
        """
        catalog = [
            {"name": "Rover R2", "category": "robots", "price": 38000},
            {"name": "Docking station", "category": "accessories", "price": 2500},
            {"name": "Extended warranty", "category": "services", "price": 4500},
        ]
        return [
            p for p in catalog
            if p["category"] == category and (max_price is None or p["price"] <= max_price)
        ]

    @tool
    def convert_currency(
        amount: Annotated[float, "Amount of money to convert"],
        to: Annotated[Literal["USD", "GBP", "PLN"], "Target currency"],
    ) -> str:
        """Convert an amount in EUR to another currency."""
        rates = {"USD": 1.09, "GBP": 0.85, "PLN": 4.3}
        return f"{amount * rates[to]:.2f} {to}"

    return convert_currency, search_products


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Explicit Pydantic `args_schema`

    Gives full control (validation, defaults, docs). `ToolException` +
    `handle_tool_error=True` returns the message to the model instead of crashing,
    so the agent can fix its call and retry.
    """)
    return


@app.class_definition
class TicketInput(BaseModel):
    title: str = Field(description="Short summary of the problem")
    severity: Literal["low", "medium", "high", "critical"]
    customer_email: str = Field(pattern=r"^[^@\s]+@[^@\s]+$")


@app.cell
def _():
    def _create_ticket(title: str, severity: str, customer_email: str) -> str:
        if severity == "critical" and "outage" not in title.lower():
            raise ToolException("Critical tickets must describe an outage in the title.")
        return f"Ticket #1042 created for {customer_email} ({severity}): {title}"

    create_ticket = StructuredTool.from_function(
        func=_create_ticket,
        name="create_ticket",
        description="Create a support ticket in the helpdesk system.",
        args_schema=TicketInput,
        handle_tool_error=True,
    )
    return (create_ticket,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What the model sees for each tool
    """)
    return


@app.cell
def _(convert_currency, create_ticket, multiply, search_products):
    tools = [multiply, search_products, convert_currency, create_ticket]
    mo.ui.table([{"name": t.name, "description": t.description.splitlines()[0], "args": t.args} for t in tools])
    return (tools,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Calling tools directly (tools are Runnables)
    """)
    return


@app.cell
def _(convert_currency, create_ticket, multiply, search_products):
    {
        "multiply": multiply.invoke({"a": 6, "b": 7}),
        "search_products": search_products.invoke({"query": "anything", "max_price": 5000, "category": "accessories"}),
        "convert_currency": convert_currency.invoke({"amount": 100, "to": "USD"}),
        "create_ticket ok": create_ticket.invoke(
            {"title": "Robot stuck", "severity": "high", "customer_email": "a@b.co"}
        ),
        "create_ticket error": create_ticket.invoke(
            {"title": "Robot stuck", "severity": "critical", "customer_email": "a@b.co"}
        ),
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## OpenAI-format schema actually sent to the API
    """)
    return


@app.cell
def _(tools):
    tool_picker = mo.ui.dropdown({t.name: t for t in tools}, value="search_products", label="Tool")
    tool_picker
    return (tool_picker,)


@app.cell
def _(tool_picker):
    mo.json(convert_to_openai_tool(tool_picker.value))
    return


if __name__ == "__main__":
    app.run()
