import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.prompts import (
        ChatPromptTemplate,
        FewShotChatMessagePromptTemplate,
        MessagesPlaceholder,
        PromptTemplate,
    )

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Prompt templates

    Reusable, parameterised prompts.

    Open: `uv run marimo edit 01_basics/03_prompt_templates.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `PromptTemplate`

    A plain string template.
    """)
    return


@app.cell
def _():
    template = PromptTemplate.from_template("Give a {adjective} name for a {thing}. Name only.")
    prompt_value = template.invoke({"adjective": "funny", "thing": "pet turtle"})
    {"rendered": prompt_value.to_string(), "answer": llm.invoke(prompt_value).text}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `ChatPromptTemplate`

    A list of role messages with variables.
    """)
    return


@app.cell
def _():
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator. Translate the user's text to {language}. Output only the translation."),
            ("human", "{text}"),
        ]
    )
    translation_messages = chat_template.invoke({"language": "French", "text": "The weather is lovely today."})
    {"messages": translation_messages.to_messages(), "answer": llm.invoke(translation_messages).text}
    return (chat_template,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `MessagesPlaceholder`

    Slot a whole list of messages (e.g. chat history) into a prompt.
    """)
    return


@app.cell
def _():
    with_history = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    history = [HumanMessage("My name is Alex and I love Rust."), AIMessage("Nice to meet you, Alex!")]
    llm.invoke(with_history.invoke({"history": history, "question": "Which language do I love?"})).text
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. `partial()`

    Pre-fill some variables now, the rest later.
    """)
    return


@app.cell
def _(chat_template):
    to_spanish = chat_template.partial(language="Spanish")
    llm.invoke(to_spanish.invoke({"text": "Good morning!"})).text
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. `FewShotChatMessagePromptTemplate`

    Few-shot prompting: teach a format by example.
    """)
    return


@app.cell
def _():
    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "fast", "output": "slow"},
    ]
    few_shot = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
    )
    antonym_prompt = ChatPromptTemplate.from_messages(
        [("system", "Reply with the antonym of the word only."), few_shot, ("human", "{input}")]
    )
    antonym_word = mo.ui.text(value="generous", label="word")
    antonym_word
    return antonym_prompt, antonym_word


@app.cell
def _(antonym_prompt, antonym_word):
    {f"antonym of {antonym_word.value!r}": llm.invoke(antonym_prompt.invoke({"input": antonym_word.value})).text}
    return


if __name__ == "__main__":
    app.run()
