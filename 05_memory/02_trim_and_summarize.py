import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
    from langchain_core.messages.utils import count_tokens_approximately

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Keeping long conversations inside the context window

    1. `trim_messages`: keep only the last N tokens/messages (cheap, lossy).
    2. Running summary: fold old messages into a summary (costs an LLM call, keeps facts).

    For agents, `SummarizationMiddleware` does this automatically (see `04_tools_and_agents/05_middleware.py`).

    Open: `uv run marimo edit 05_memory/02_trim_and_summarize.py`
    """)
    return


@app.cell
def _():
    history = [
        SystemMessage("You are a helpful travel assistant."),
        HumanMessage("Hi, I'm Sam. I'm allergic to peanuts."),
        AIMessage("Hi Sam! Noted - no peanuts."),
        HumanMessage("I'm going to Thailand in March."),
        AIMessage("Great choice! March is hot and dry."),
        HumanMessage("I'll be in Bangkok for 4 days, then Chiang Mai."),
        AIMessage("Nice route. Bangkok has amazing street food."),
        HumanMessage("I prefer cheap guesthouses over hotels."),
        AIMessage("Understood, budget guesthouses it is."),
    ]
    question = HumanMessage("Suggest one street-food dish for me. Keep it to one sentence and mind my restrictions.")
    return history, question


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `trim_messages`

    Keeps the system message + the most recent messages that fit. Trimmers are
    Runnables, so they slot into a chain: `trimmer | prompt | llm`.
    Move the slider to see how much history survives.
    """)
    return


@app.cell
def _():
    max_tokens = mo.ui.slider(20, 150, step=10, value=60, label="max_tokens", show_value=True)
    max_tokens
    return (max_tokens,)


@app.cell
def _(history, max_tokens, question):
    trimmer = trim_messages(
        max_tokens=max_tokens.value,
        strategy="last",
        token_counter=count_tokens_approximately,  # or pass the llm to count with its tokenizer
        include_system=True,
        start_on="human",  # never start the kept window with an AI message
    )
    trimmed = trimmer.invoke(history)
    mo.vstack([
        mo.ui.table([{"role": m.type, "text": m.text} for m in trimmed], label="Kept"),
        mo.md(f"**Answer with trimmed history (the allergy may be lost!):** {llm.invoke(trimmed + [question]).text}"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Running summary

    Summarize old turns, keep the last few verbatim.
    """)
    return


@app.cell
def _(history, question):
    KEEP_LAST = 2
    _old, _recent = history[1:-KEEP_LAST], history[-KEEP_LAST:]
    summary = llm.invoke(
        [
            SystemMessage("Summarize the conversation in 2-3 bullet points. Keep all personal facts and constraints."),
            HumanMessage("\n".join(f"{m.type}: {m.text}" for m in _old)),
        ]
    ).text
    compact = [history[0], SystemMessage(f"Summary of earlier conversation:\n{summary}"), *_recent]
    mo.vstack([
        mo.md(f"**Summary:**\n\n{summary}"),
        mo.md(f"**Tokens before/after:** {count_tokens_approximately(history)} -> {count_tokens_approximately(compact)}"),
        mo.md(f"**Answer with summary:** {llm.invoke(compact + [question]).text}"),
    ])
    return


if __name__ == "__main__":
    app.run()
