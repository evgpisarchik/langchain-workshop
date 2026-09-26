import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from pydantic import BaseModel, Field
    from rag_common import build_vector_store, format_docs

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Better retrieval: query rewriting, multi-query and conversational RAG

    The user's words are often a poor search query. Let the LLM fix that.

    Open: `uv run marimo edit 06_rag/04_advanced_retrieval.py`
    """)
    return


@app.cell
def _():
    store = build_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": 3})
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer using ONLY this context, cite [n]. Say 'I don't know' if it's missing.\n\n{context}"),
            MessagesPlaceholder("history", optional=True),
            ("human", "{question}"),
        ]
    )
    answer_chain = answer_prompt | llm | StrOutputParser()
    return answer_chain, retriever


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Multi-query retrieval

    Generate several phrasings, retrieve for each, merge unique docs.
    """)
    return


@app.class_definition
class Queries(BaseModel):
    """Alternative search queries."""

    queries: list[str] = Field(description="3 differently worded search queries for the same need")


@app.cell
def _(answer_chain, retriever):
    query_generator = (
        ChatPromptTemplate.from_template(
            "Rewrite the question into 3 search queries using policy/handbook vocabulary.\nQuestion: {question}"
        )
        | llm.with_structured_output(Queries)
    )
    mq_question = "I'm having a baby soon - how much paid time off do I get?"
    mq_queries = [mq_question, *query_generator.invoke({"question": mq_question}).queries]
    _unique = {}
    for _docs in retriever.batch(mq_queries):
        for _doc in _docs:
            _unique.setdefault(_doc.page_content, _doc)
    mq_docs = list(_unique.values())
    {
        "queries": mq_queries,
        "retrieved sections": [d.metadata["section"] for d in mq_docs],
        "answer": answer_chain.invoke({"context": format_docs(mq_docs), "question": mq_question}),
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Conversational RAG (history-aware query rewriting)

    Follow-up questions like "and in London?" are meaningless as search queries.
    Rewrite them into standalone questions first.
    """)
    return


@app.class_definition
class SearchQuery(BaseModel):
    """A standalone search query."""

    query: str = Field(description="Self-contained search query (max 12 words) that resolves 'it', 'there', etc.")


@app.cell
def _(answer_chain, retriever):
    rewrite_chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "Do NOT answer the question. Rewrite the user's last question as a standalone "
                           "search query, using the chat history to resolve references."),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        | llm.with_structured_output(SearchQuery)
        | (lambda result: result.query)
    )

    history: list = []
    turns = []
    for _turn in [
        "What's the hotel limit per night for business trips?",
        "And what about in London?",
        "Can I expense a drink at dinner there?",
    ]:
        _standalone = rewrite_chain.invoke({"history": history, "question": _turn}) if history else _turn
        _docs = retriever.invoke(_standalone)
        _answer = answer_chain.invoke({"context": format_docs(_docs), "history": history, "question": _turn})
        turns.append({"user": _turn, "search query": _standalone, "bot": str(_answer)})
        history += [HumanMessage(_turn), AIMessage(_answer)]
    mo.ui.table(turns)
    return


if __name__ == "__main__":
    app.run()
