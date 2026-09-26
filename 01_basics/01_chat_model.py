import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Chat model basics

    `invoke`, message types, response metadata, per-call settings and `batch`.

    Open: `uv run marimo edit 01_basics/01_chat_model.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `invoke()` with a string

    The simplest call: a plain string becomes a single `HumanMessage`.
    `.text` joins all text blocks of the content.
    """)
    return


@app.cell
def _():
    response = llm.invoke("In one sentence, what is LangChain?")
    {"type": type(response).__name__, "text": response.text}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `invoke()` with a list of messages

    A conversation is a list of messages. `SystemMessage` sets behaviour,
    `AIMessage` lets you inject previous model turns (few-shot, history).
    The same thing works with `(role, content)` tuples or OpenAI-style dicts.
    """)
    return


@app.cell
def _():
    messages = [
        SystemMessage("You are a terse assistant that answers with a single word."),
        HumanMessage("Capital of France?"),
        AIMessage("Paris"),
        HumanMessage("Capital of Belarus?"),
    ]
    {
        "message objects": llm.invoke(messages).text,
        "tuples": llm.invoke([("system", "Answer in one word."), ("human", "Capital of Japan?")]).text,
        "dicts": llm.invoke([{"role": "user", "content": "2+2? Digits only."}]).text,
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Response metadata

    Everything besides the text: token usage, model name, reasoning.
    The model is a reasoning model; its chain of thought comes back separately.
    """)
    return


@app.cell
def _():
    _response = llm.invoke("Name three prime numbers.")
    {
        "text": _response.text,
        "usage_metadata": _response.usage_metadata,
        "model": _response.response_metadata.get("model_name"),
        "reasoning": _response.additional_kwargs.get("reasoning", {}),
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Temperature / reasoning effort

    Per-call model settings: any constructor kwarg can be overridden in `get_llm()`.
    """)
    return


@app.cell
def _():
    creative = get_llm(temperature=1.0)
    thinker = get_llm(reasoning={"effort": "high"})
    {
        "creative": creative.invoke("Invent a name for a robot vacuum. Name only.").text,
        "high effort": thinker.invoke("Is 391 prime? Answer yes/no and the factors.").text,
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. `batch()`

    Runs many independent inputs concurrently (thread pool under the hood).
    """)
    return


@app.cell
def _():
    questions = ["Translate 'hello' to German", "Translate 'hello' to Spanish", "Translate 'hello' to Russian"]
    answers = llm.batch(questions, config={"max_concurrency": 3})
    mo.ui.table([{"question": q, "answer": a.text} for q, a in zip(questions, answers)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Playground

    Try your own prompt with different settings.
    """)
    return


@app.cell
def _():
    playground_prompt = mo.ui.text_area(
        value="Explain what a vector database is in two sentences.", label="Prompt", full_width=True
    )
    playground_temperature = mo.ui.slider(0, 1.5, step=0.1, value=0, label="temperature", show_value=True)
    playground_effort = mo.ui.dropdown(["low", "medium", "high"], value="low", label="reasoning effort")
    playground_run = mo.ui.run_button(label="Ask")
    mo.vstack([playground_prompt, mo.hstack([playground_temperature, playground_effort, playground_run])])
    return (
        playground_effort,
        playground_prompt,
        playground_run,
        playground_temperature,
    )


@app.cell
def _(
    playground_effort,
    playground_prompt,
    playground_run,
    playground_temperature,
):
    mo.stop(not playground_run.value, mo.md("_Press **Ask** to run._"))
    _model = get_llm(temperature=playground_temperature.value, reasoning={"effort": playground_effort.value})
    mo.md(_model.invoke(playground_prompt.value).text)
    return


if __name__ == "__main__":
    app.run()
