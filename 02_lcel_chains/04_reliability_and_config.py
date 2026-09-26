import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import random

    import marimo as mo
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import ConfigurableField, RunnableLambda

    from shared import get_llm

    prompt = ChatPromptTemplate.from_template("Give one synonym for '{word}'. Word only.")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Making chains robust and configurable

    Retries, fallbacks, runtime config.

    Open: `uv run marimo edit 02_lcel_chains/04_reliability_and_config.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `with_fallbacks`

    If the primary runnable raises, try the next one. Here the primary model name
    does not exist on the gateway -> 400 -> fallback.
    """)
    return


@app.cell
def _():
    broken = get_llm(model="model-that-does-not-exist", max_retries=0)
    reliable = broken.with_fallbacks([get_llm()])
    (prompt | reliable | StrOutputParser()).invoke({"word": "happy"})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `with_retry`

    Retry flaky steps with exponential backoff.
    """)
    return


@app.cell
def _():
    attempts = {"n": 0}

    def flaky_step(text: str) -> str:
        attempts["n"] += 1
        if random.random() < 0.5:
            raise ConnectionError(f"simulated network error on attempt {attempts['n']}")
        return text

    random.seed(3)
    flaky = RunnableLambda(flaky_step).with_retry(
        retry_if_exception_type=(ConnectionError,), stop_after_attempt=10, wait_exponential_jitter=False
    )
    retry_chain = prompt | get_llm() | StrOutputParser() | flaky
    {"answer": retry_chain.invoke({"word": "fast"}), "attempts needed": attempts["n"]}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `RunnableConfig`

    Tags, metadata, `run_name`, `max_concurrency`. These flow to callbacks/tracing
    (see 08_production) without changing code.
    """)
    return


@app.cell
def _():
    words = [{"word": w} for w in ["big", "smart", "cold", "bright"]]
    (prompt | get_llm() | StrOutputParser()).batch(
        words,
        config={"max_concurrency": 2, "tags": ["synonyms"], "metadata": {"user": "demo"}, "run_name": "synonyms"},
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. `configurable_fields`

    Expose model parameters that callers can change per call.
    """)
    return


@app.cell
def _():
    configurable_llm = get_llm().configurable_fields(
        temperature=ConfigurableField(id="temperature", name="LLM temperature")
    )
    story = ChatPromptTemplate.from_template("Name a new {thing}. Name only.") | configurable_llm | StrOutputParser()
    story_temperature = mo.ui.slider(0, 1.5, step=0.1, value=1.2, label="temperature", show_value=True)
    story_temperature
    return story, story_temperature


@app.cell
def _(story, story_temperature):
    {
        "temperature=0": story.invoke({"thing": "ice-cream flavour"}),
        f"temperature={story_temperature.value}": story.with_config(
            configurable={"temperature": story_temperature.value}
        ).invoke({"thing": "ice-cream flavour"}),
    }
    return


if __name__ == "__main__":
    app.run()
