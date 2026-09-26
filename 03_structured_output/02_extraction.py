import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from datetime import date
    from typing import Literal, Optional

    import marimo as mo
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    from shared import get_llm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Information extraction from unstructured text

    Use case: turn meeting notes and emails into structured records (action items,
    dates, people) that you can store in a database or send to a ticketing system.

    Open: `uv run marimo edit 03_structured_output/02_extraction.py`
    """)
    return


@app.class_definition
class ActionItem(BaseModel):
    """A task someone agreed to do."""

    owner: str = Field(description="Person responsible")
    task: str = Field(description="What must be done, imperative mood")
    due: Optional[date] = Field(default=None, description="Deadline as YYYY-MM-DD if stated, else null")
    priority: Literal["low", "medium", "high"] = "medium"


@app.class_definition
class MeetingSummary(BaseModel):
    """Structured summary of a meeting."""

    title: str
    attendees: list[str]
    decisions: list[str] = Field(description="Decisions that were made")
    action_items: list[ActionItem]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Meeting notes -> `MeetingSummary`

    A system prompt + today's date lets the model resolve relative dates ("next Friday").
    Edit the notes and re-run.
    """)
    return


@app.cell
def _():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at extracting structured data from notes. Today is {today}. "
                "Only extract information that is explicitly present. Never invent attendees or dates.",
            ),
            ("human", "{notes}"),
        ]
    )
    extractor = prompt | get_llm().with_structured_output(MeetingSummary)

    notes = mo.ui.text_area(
        value=(
            "Sync on the Q3 launch - Anna, Pavel, Mike and Olga joined.\n"
            "We agreed to move the launch to September 15 and to drop the Android widget from v1.\n"
            "Pavel will finish the payment integration by Sept 5, this is critical.\n"
            "Olga should prepare the press release draft by next Friday.\n"
            "Mike mentioned he might look into the crash reports, no deadline."
        ),
        label="Meeting notes",
        full_width=True,
        rows=6,
    )
    notes
    return extractor, notes


@app.cell
def _(extractor, notes):
    summary = extractor.invoke({"notes": notes.value, "today": "2026-08-20"})
    mo.vstack([
        mo.md(f"### {summary.title}\n\n**Attendees:** {', '.join(summary.attendees)}"),
        mo.md("**Decisions:**\n\n" + "\n".join(f"- {d}" for d in summary.decisions)),
        mo.ui.table([item.model_dump() for item in summary.action_items], label="Action items"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Batch extraction from emails

    Extraction over many documents: `batch()` runs them concurrently.
    """)
    return


@app.class_definition
class Order(BaseModel):
    """A purchase order found in an email."""

    customer: str
    product: str
    quantity: int
    unit_price_eur: Optional[float] = Field(default=None, description="Price per unit in EUR if stated")


@app.cell
def _():
    emails = [
        "Hi, this is Green Farms Ltd. We'd like 12 Rover R2 robots, quoted at 36,500 EUR each.",
        "Dear sales, please send 3 docking stations to BlueShip GmbH. Thanks, Karl",
        "Hello! Acme Retail wants to order forty extended warranties at 4500 euros apiece.",
    ]
    order_extractor = get_llm().with_structured_output(Order)
    mo.ui.table([{"email": e, **o.model_dump()} for e, o in zip(emails, order_extractor.batch(emails))])
    return


if __name__ == "__main__":
    app.run()
