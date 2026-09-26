import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from collections import Counter
    from typing import Literal

    import marimo as mo
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    from shared import get_llm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Classification / tagging with structured output

    Use case: tag incoming customer reviews with sentiment, topic, language and
    urgency so they can be routed and aggregated.

    Open: `uv run marimo edit 03_structured_output/03_classification.py`
    """)
    return


@app.class_definition
class ReviewTags(BaseModel):
    """Tags for a customer review."""

    sentiment: Literal["positive", "neutral", "negative"]
    topics: list[Literal["price", "quality", "delivery", "support", "usability"]] = Field(
        description="All topics the review talks about"
    )
    language: str = Field(description="ISO 639-1 code of the review language, e.g. 'en'")
    urgent: bool = Field(description="True if the customer threatens to leave or reports a safety issue")
    summary_en: str = Field(description="One-sentence summary in English")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tagging reviews in a batch
    """)
    return


@app.cell
def _():
    tagger = (
        ChatPromptTemplate.from_template("Tag this customer review:\n\n{review}")
        | get_llm().with_structured_output(ReviewTags)
    )
    reviews = [
        "Great robot, setup took 10 minutes and support answered in an hour!",
        "Доставка заняла три недели, а цена выше, чем у конкурентов.",
        "The robot almost hit a worker yesterday. If this isn't fixed we cancel the contract.",
        "Es funktioniert. Nichts Besonderes.",
        "Too expensive for what it does, and the app is confusing.",
    ]
    tags = tagger.batch([{"review": r} for r in reviews], config={"max_concurrency": 5})
    mo.ui.table([{"review": r, **t.model_dump()} for r, t in zip(reviews, tags)])
    return (tags,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Aggregates
    """)
    return


@app.cell
def _(tags):
    {
        "sentiment": dict(Counter(t.sentiment for t in tags)),
        "topics": dict(Counter(topic for t in tags for topic in t.topics)),
        "urgent": sum(t.urgent for t in tags),
    }
    return


if __name__ == "__main__":
    app.run()
