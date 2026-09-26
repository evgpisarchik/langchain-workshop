import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import time
    from typing import Any

    import marimo as mo
    from langchain.agents import create_agent
    from langchain_core.callbacks import BaseCallbackHandler, get_usage_metadata_callback
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool

    from shared import get_llm

    llm = get_llm()
    chain = ChatPromptTemplate.from_template("One-line fun fact about {topic}.") | llm | StrOutputParser()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Observability: callbacks, token usage tracking, event streaming, tracing

    LangSmith tracing needs no code at all - set these env vars (see `.env.example`):

    ```
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    ```

    and every chain / agent / graph run is traced to smith.langchain.com.

    Open: `uv run marimo edit 08_production/01_callbacks_and_observability.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Custom callback handler

    Hook into start/end/error events of any component. Passed per call via `config`;
    the log appears in the cell's console output.
    """)
    return


@app.class_definition
class TimingLogger(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts: dict[Any, float] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self.starts[run_id] = time.perf_counter()
        print(f"[llm start] {len(messages[0])} message(s), tags={kwargs.get('tags')}")

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        elapsed = time.perf_counter() - self.starts.pop(run_id, time.perf_counter())
        usage = response.generations[0][0].message.usage_metadata or {}
        print(f"[llm end] {elapsed:.2f}s, {usage.get('total_tokens')} tokens")

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        print(f"[tool start] {serialized.get('name')}({input_str})")

    def on_chain_error(self, error, **kwargs) -> None:
        print(f"[error] {error!r}")


@app.cell
def _():
    chain.invoke({"topic": "octopuses"}, config={"callbacks": [TimingLogger()], "tags": ["demo"]})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `get_usage_metadata_callback`

    Token accounting across many calls (e.g. per user request, for billing).
    """)
    return


@app.cell
def _():
    with get_usage_metadata_callback() as _usage_cb:
        chain.batch([{"topic": t} for t in ["bees", "volcanoes", "Saturn"]])
    {"usage by model": _usage_cb.usage_metadata}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Callbacks inside an agent

    Callbacks also see agent tool calls.
    """)
    return


@app.cell
def _():
    @tool
    def get_time(city: str) -> str:
        """Get the current local time in a city."""
        return f"It is 14:05 in {city}"

    agent = create_agent(llm, tools=[get_time])
    agent.invoke(
        {"messages": [{"role": "user", "content": "What time is it in Tokyo?"}]}, config={"callbacks": [TimingLogger()]}
    )["messages"][-1].text
    return (agent,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. `astream_events(version="v2")`

    A single stream of fine-grained events from every nested component - ideal for
    UIs that show "searching...", tokens, tool results.
    """)
    return


@app.cell
async def _(agent):
    _events = []
    _answer = ""
    async for _event in agent.astream_events(
        {"messages": [{"role": "user", "content": "What time is it in Oslo? Answer in 5 words."}]}, version="v2"
    ):
        _kind = _event["event"]
        if _kind == "on_tool_start":
            _events.append(f"- tool `{_event['name']}` started with `{_event['data'].get('input')}`")
        elif _kind == "on_tool_end":
            _events.append(f"- tool `{_event['name']}` returned `{_event['data']['output'].content}`")
        elif _kind == "on_chat_model_stream" and _event["data"]["chunk"].text:
            _answer += _event["data"]["chunk"].text
        mo.output.replace(mo.md("\n".join(_events) + f"\n\n**Streamed answer:** {_answer}"))
    return


if __name__ == "__main__":
    app.run()
