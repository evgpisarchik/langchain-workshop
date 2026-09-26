import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Unit-testing LLM code without calling an LLM

    Fake chat models return scripted responses (including tool calls), so chains
    and agents can be tested deterministically, fast and for free in CI.
    This notebook makes no network calls.

    The code under test and the tests are top-level functions (`@app.function`),
    so pytest collects them straight from the notebook:

    ```bash
    uv run pytest 08_production/03_testing_with_fake_models.py -q
    ```

    Open: `uv run marimo edit 08_production/03_testing_with_fake_models.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Code under test
    """)
    return


@app.function
def build_summary_chain(llm):
    prompt = ChatPromptTemplate.from_template("Summarize: {text}")
    return prompt | llm | StrOutputParser() | (lambda s: s.strip().rstrip(".") + ".")


@app.function
def get_balance(account: str) -> str:
    """Get an account balance."""
    return f"{account}: 100 EUR"


@app.function
def build_agent(llm):
    return create_agent(llm, tools=[tool(get_balance)], system_prompt="You are a banking assistant.")


@app.class_definition
class ToolCallingFake(GenericFakeChatModel):
    """GenericFakeChatModel + a no-op bind_tools, so create_agent accepts it."""

    def bind_tools(self, tools, **kwargs):
        return self


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tests
    """)
    return


@app.function
def test_summary_chain_formats_output():
    fake = GenericFakeChatModel(messages=iter([AIMessage("  A short summary  ")]))
    assert build_summary_chain(fake).invoke({"text": "long text"}) == "A short summary."


@app.function
def test_agent_calls_tool_then_answers():
    fake = ToolCallingFake(
        messages=iter(
            [
                AIMessage("", tool_calls=[{"name": "get_balance", "args": {"account": "main"}, "id": "call_1"}]),
                AIMessage("Your balance is 100 EUR."),
            ]
        )
    )
    result = build_agent(fake).invoke({"messages": [{"role": "user", "content": "balance of main?"}]})
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert tool_messages[0].content == "main: 100 EUR"
    assert result["messages"][-1].text == "Your balance is 100 EUR."


@app.function
def test_streaming_is_token_by_token():
    fake = GenericFakeChatModel(messages=iter([AIMessage("hello brave new world")]))
    chunks = [c.text for c in fake.stream("hi")]
    assert len(chunks) > 1 and "".join(chunks) == "hello brave new world"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run the tests here
    """)
    return


@app.cell
def _():
    _results = []
    for _test in (test_summary_chain_formats_output, test_agent_calls_tool_then_answers, test_streaming_is_token_by_token):
        try:
            _test()
            _results.append({"test": _test.__name__, "result": "PASSED"})
        except AssertionError as _error:
            _results.append({"test": _test.__name__, "result": f"FAILED: {_error!r}"})
    mo.ui.table(_results)
    return


if __name__ == "__main__":
    app.run()
