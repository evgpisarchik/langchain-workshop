import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from operator import itemgetter

    import marimo as mo
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    from shared import get_llm

    llm = get_llm()
    to_text = StrOutputParser()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Multi-step chains: parallel branches and data flow

    Use case: from one product description, generate a tagline, a tweet and a
    list of target audiences in parallel, then have a second LLM step pick the best.

    Open: `uv run marimo edit 02_lcel_chains/02_parallel_and_sequential.py`
    """)
    return


@app.function
def make_chain(instruction: str):
    prompt = ChatPromptTemplate.from_template(instruction + "\n\nProduct: {product}")
    return prompt | llm | to_text


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `RunnableParallel`

    Same input -> several chains, results collected in a dict. Branches run concurrently.
    """)
    return


@app.cell
def _():
    product = mo.ui.text_area(
        value="A smart water bottle that reminds you to drink and tracks intake in an app.",
        label="Product",
        full_width=True,
    )
    product
    return (product,)


@app.cell
def _(product):
    marketing = RunnableParallel(
        tagline=make_chain("Write one catchy tagline (max 8 words). Output only the tagline."),
        tweet=make_chain("Write a launch tweet (max 200 characters). Output only the tweet."),
        audiences=make_chain("List 3 target audiences, comma separated. Output only the list."),
    )
    marketing.invoke({"product": product.value})
    return (marketing,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `RunnablePassthrough.assign` - sequential pipeline

    Keep the input and add new keys to it. This is how you thread data through
    a multi-step pipeline.
    """)
    return


@app.cell
def _(marketing, product):
    pick_best = (
        ChatPromptTemplate.from_template(
            "Product: {product}\nTagline A: {tagline}\nTweet: {tweet}\n"
            "Audiences: {audiences}\n\n"
            "Which ONE audience fits the tagline and tweet best? Answer with the audience and one reason."
        )
        | llm
        | to_text
    )
    pipeline = RunnablePassthrough.assign(**marketing.steps__) | RunnablePassthrough.assign(verdict=pick_best)
    pipeline.invoke({"product": product.value})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `itemgetter` for routing inputs

    Pick fields out of the input dict for a sub-chain.
    """)
    return


@app.cell
def _():
    summarize_then_translate = (
        {
            "summary": {"text": itemgetter("text")}
            | ChatPromptTemplate.from_template("Summarize in one sentence:\n{text}")
            | llm
            | to_text,
            "language": itemgetter("language"),
        }
        | ChatPromptTemplate.from_template("Translate to {language}. Output only the translation:\n{summary}")
        | llm
        | to_text
    )
    text = (
        "LangChain provides standard interfaces for models, prompts, retrievers and tools, "
        "plus LCEL to compose them. LangGraph adds stateful, cyclic workflows for agents."
    )
    summarize_then_translate.invoke({"text": text, "language": "German"})
    return


if __name__ == "__main__":
    app.run()
