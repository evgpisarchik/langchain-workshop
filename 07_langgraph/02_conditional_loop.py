import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from typing import Literal, TypedDict

    import marimo as mo
    from langgraph.graph import END, START, StateGraph
    from pydantic import BaseModel, Field

    from shared import get_llm

    llm = get_llm()
    MAX_ITERATIONS = 3


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Conditional edges and cycles: the evaluator-optimizer (reflection) pattern

    A generator writes, an evaluator grades with structured output, and a
    conditional edge loops back with feedback until the grade passes or we hit a
    max number of iterations. Cycles are what LCEL chains cannot express.

    Open: `uv run marimo edit 07_langgraph/02_conditional_loop.py`
    """)
    return


@app.class_definition
class State(TypedDict):
    task: str
    draft: str
    feedback: str
    grade: str
    iterations: int


@app.class_definition
class Evaluation(BaseModel):
    """Strict review of a product description."""

    grade: Literal["pass", "fail"] = Field(
        description="pass ONLY if every requirement is met: under 40 words, mentions the price, "
        "mentions battery life, ends with a call to action"
    )
    feedback: str = Field(description="Concrete fixes needed; empty if pass")


@app.function
def generate(state: State) -> dict:
    prompt = f"Task: {state['task']}"
    if state.get("feedback"):
        prompt += f"\n\nPrevious draft:\n{state['draft']}\n\nReviewer feedback to address:\n{state['feedback']}"
    draft = llm.invoke(prompt + "\n\nOutput only the description.").text
    mo.output.append(mo.md(f"**Draft #{state['iterations'] + 1}** ({len(draft.split())} words)\n\n> {draft}"))
    return {"draft": draft, "iterations": state["iterations"] + 1}


@app.function
def evaluate(state: State) -> dict:
    result = llm.with_structured_output(Evaluation).invoke(
        f"Requirements: under 40 words, mentions the price, mentions battery life, ends with a call to action.\n\n"
        f"Description to review:\n{state['draft']}"
    )
    mo.output.append(mo.md(f"**Evaluation: {result.grade}** {result.feedback}"))
    return {"grade": result.grade, "feedback": result.feedback}


@app.function
def route(state: State) -> Literal["generate", "__end__"]:
    if state["grade"] == "pass" or state["iterations"] >= MAX_ITERATIONS:
        return END
    return "generate"


@app.cell
def _():
    graph = (
        StateGraph(State)
        .add_node("generate", generate)
        .add_node("evaluate", evaluate)
        .add_edge(START, "generate")
        .add_edge("generate", "evaluate")
        .add_conditional_edges("evaluate", route)  # the path function returns the next node name
        .compile()
    )
    mo.mermaid(graph.get_graph().draw_mermaid())
    return (graph,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reflection loop
    """)
    return


@app.cell
def _(graph):
    final = graph.invoke(
        {
            "task": "Write a product description for the Rover R2 warehouse robot: price 38,000 EUR, "
                    "10 hour battery, moves 800 kg shelves. Make it exciting and detailed.",
            "iterations": 0,
            "feedback": "",
        }
    )
    mo.output.append(mo.md(f"---\n**Final grade:** {final['grade']} after {final['iterations']} iteration(s)"))
    return


if __name__ == "__main__":
    app.run()
