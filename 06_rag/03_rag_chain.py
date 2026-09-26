import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from pydantic import BaseModel, Field
    from rag_common import build_vector_store, format_docs

    from shared import get_llm, stream_text

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # RAG step 3: retrieve relevant chunks and answer with citations

    Use case: an internal HR / policy assistant over the company handbook.

    Open: `uv run marimo edit 06_rag/03_rag_chain.py`
    """)
    return


@app.cell
def _():
    retriever = build_vector_store().as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions about Northwind Robotics policies using ONLY the context below. "
                "Cite sources as [n]. If the answer is not in the context, say you don't know.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    return prompt, retriever


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Basic RAG chain

    The classic LCEL RAG chain. The last question is not in the docs, so the model
    should say it doesn't know.
    """)
    return


@app.cell
def _(prompt, retriever):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    _questions = [
        "How many vacation days can I carry over, and until when?",
        "What's the hotel limit per night in Paris?",
        "Does the company offer a gym membership?",
    ]
    mo.ui.table([{"question": q, "answer": a} for q, a in zip(_questions, rag_chain.batch(_questions))])
    return (rag_chain,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Answer + source documents

    Return the sources alongside the answer (needed for a UI with citations).
    """)
    return


@app.cell
def _(prompt, retriever):
    rag_with_sources = RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["question"])
    ).assign(
        answer=(lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )
    _result = rag_with_sources.invoke({"question": "How fast can the Rover R2 move and when does it stop?"})
    {
        "answer": _result["answer"],
        "sources": [f"{d.metadata['source']} > {d.metadata['section']}" for d in _result["context"]],
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Structured answer with cited chunk ids

    The model tells us which chunks it actually used.
    """)
    return


@app.class_definition
class CitedAnswer(BaseModel):
    """Answer to the user question, based only on the given sources."""

    answer: str
    citations: list[int] = Field(description="Numbers [n] of the sources that support the answer")
    found_in_context: bool = Field(description="False if the context does not contain the answer")


@app.cell
def _(prompt, retriever):
    structured_rag = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm.with_structured_output(CitedAnswer)
    )
    {
        "in docs": structured_rag.invoke("What do I need to submit with an expense?"),
        "not in docs": structured_rag.invoke("Who is the CEO of Northwind?"),
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Ask the handbook (streaming)

    Streaming works through the whole chain.
    """)
    return


@app.cell
def _():
    handbook_question = mo.ui.text(value="Summarize the rules for sick leave.", label="Question", full_width=True)
    handbook_question
    return (handbook_question,)


@app.cell
def _(handbook_question, rag_chain):
    stream_text(rag_chain.stream(handbook_question.value))
    return


if __name__ == "__main__":
    app.run()
