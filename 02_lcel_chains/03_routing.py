import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from typing import Literal

    import marimo as mo
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
    from pydantic import BaseModel, Field

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Routing: classify the input, then send it to a specialised chain

    Use case: a support inbox where billing, technical and other questions each
    get a different expert prompt.

    Open: `uv run marimo edit 02_lcel_chains/03_routing.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Step 1 is a classifier with structured output (see 03_structured_output),
    then one expert chain per category.
    """)
    return


@app.class_definition
class Route(BaseModel):
    """Which team should handle the customer message."""

    category: Literal["billing", "technical", "other"] = Field(
        description=(
            "billing = payments, invoices, refunds, subscriptions; "
            "technical = bugs, crashes, errors, how the product works; "
            "other = anything else (company info, offices, jobs, feedback)"
        )
    )


@app.function
def expert(persona: str):
    return (
        ChatPromptTemplate.from_messages(
            [("system", f"You are {persona}. Reply in 2 short sentences."), ("human", "{question}")]
        )
        | llm
        | StrOutputParser()
    )


@app.cell
def _():
    classifier = (
        ChatPromptTemplate.from_template("Classify this customer message:\n{question}")
        | llm.with_structured_output(Route)
    )
    billing_chain = expert("a friendly billing specialist")
    technical_chain = expert("a senior support engineer")
    general_chain = expert("a helpful customer-care agent")

    questions = [
        "I was charged twice for my subscription this month.",
        "The app crashes when I upload a PNG larger than 10 MB.",
        "Do you have an office in Berlin?",
    ]
    return billing_chain, classifier, general_chain, questions, technical_chain


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A. Routing with a function (recommended)

    A `RunnableLambda` that returns a Runnable - the returned chain is invoked.
    The classification is kept in the output so we can see where each question went.
    """)
    return


@app.cell
def _(billing_chain, classifier, general_chain, questions, technical_chain):
    def route(inputs: dict):
        return {"billing": billing_chain, "technical": technical_chain}.get(
            inputs["route"].category, general_chain
        )

    router_chain = RunnablePassthrough.assign(route=classifier) | RunnablePassthrough.assign(
        answer=RunnableLambda(route)
    )
    mo.ui.table(
        [
            {"route": r["route"].category, "question": r["question"], "answer": str(r["answer"])}
            for r in router_chain.batch([{"question": q} for q in questions])
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## B. Routing with `RunnableBranch`

    `(condition, runnable)` pairs plus a default.
    """)
    return


@app.cell
def _(billing_chain, classifier, general_chain, questions, technical_chain):
    branch = RunnableBranch(
        (lambda x: x["route"].category == "billing", billing_chain),
        (lambda x: x["route"].category == "technical", technical_chain),
        general_chain,
    )
    branch_chain = RunnablePassthrough.assign(route=classifier) | branch
    branch_question = mo.ui.dropdown(questions, value=questions[1], label="Question", full_width=True)
    branch_question
    return branch_chain, branch_question


@app.cell
def _(branch_chain, branch_question):
    branch_chain.invoke({"question": branch_question.value})
    return


if __name__ == "__main__":
    app.run()
