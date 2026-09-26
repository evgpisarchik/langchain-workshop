import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from typing import Literal

    import marimo as mo
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from rag_common import build_vector_store, format_docs

    from shared import get_llm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Agentic RAG: the agent decides when and what to search

    Unlike a fixed chain, the agent can search several times, combine knowledge
    from different documents, mix retrieval with other tools, or skip retrieval
    entirely for small talk.

    Open: `uv run marimo edit 06_rag/05_agentic_rag.py`
    """)
    return


@app.cell
def _():
    store = build_vector_store()

    @tool
    def search_handbook(
        query: str,
        document: Literal["any", "vacation_policy.md", "expense_policy.md", "it_security.md", "product_faq.md"] = "any",
    ) -> str:
        """Search the Northwind Robotics employee handbook and product FAQ.

        Use short keyword queries. Optionally restrict the search to one document.
        """
        doc_filter = None if document == "any" else (lambda d: d.metadata["source"] == document)
        docs = store.similarity_search(query, k=3, filter=doc_filter)
        return format_docs(docs) if docs else "No results."

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression like '3 * 150 + 2 * 220'."""
        return str(eval(expression, {"__builtins__": {}}, {}))  # demo only

    agent = create_agent(
        get_llm(),
        tools=[search_handbook, calculator],
        system_prompt=(
            "You are the Northwind Robotics internal assistant. For any question about policies or products, "
            "search the handbook first (possibly several times) and base your answer on the results. "
            "Cite the source file names."
        ),
    )

    def ask(question: str) -> None:
        """Stream the agent's steps into the cell output."""
        mo.output.append(mo.md(f"### {question}"))
        for update in agent.stream({"messages": [{"role": "user", "content": question}]}, stream_mode="updates"):
            for node, state in update.items():
                message = state["messages"][-1]
                for call in getattr(message, "tool_calls", []):
                    mo.output.append(mo.md(f"-> `{call['name']}({call['args']})`"))
                if node == "model" and not getattr(message, "tool_calls", None):
                    mo.output.append(mo.md(f"**Answer:** {message.text}"))

    return (ask,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Several searches + a calculation
    """)
    return


@app.cell
def _(ask):
    ask(
        "I'm flying 8 hours to Tokyo for 3 nights. Can I fly business class, "
        "and what's the max I can spend on hotels in total?"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A policy question
    """)
    return


@app.cell
def _(ask):
    ask("Is it OK to paste a customer's payroll data into ChatGPT to format it?")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Small talk - no search needed
    """)
    return


@app.cell
def _(ask):
    ask("Hi! How are you today?")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Your question
    """)
    return


@app.cell
def _():
    my_question = mo.ui.text(label="Question", full_width=True, placeholder="e.g. What is the laptop password policy?")
    my_question
    return (my_question,)


@app.cell
def _(ask, my_question):
    mo.stop(not my_question.value.strip())
    ask(my_question.value)
    return


if __name__ == "__main__":
    app.run()
