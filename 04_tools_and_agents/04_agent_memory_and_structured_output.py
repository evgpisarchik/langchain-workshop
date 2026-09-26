import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from dataclasses import dataclass

    import marimo as mo
    from langchain.agents import create_agent
    from langchain.tools import ToolRuntime, tool
    from langgraph.checkpoint.memory import InMemorySaver
    from pydantic import BaseModel, Field

    from shared import get_llm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Agent memory, runtime context and structured responses

    Open: `uv run marimo edit 04_tools_and_agents/04_agent_memory_and_structured_output.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Short-term memory with a checkpointer

    A checkpointer saves the graph state after every step. Calls with the same
    `thread_id` continue the same conversation; a new `thread_id` starts fresh.
    `InMemorySaver` is for demos; use `SqliteSaver` / `PostgresSaver` in production.
    """)
    return


@app.cell
def _():
    @tool
    def add_to_cart(item: str, quantity: int = 1) -> str:
        """Add an item to the shopping cart."""
        return f"added {quantity} x {item}"

    cart_agent = create_agent(get_llm(), tools=[add_to_cart], checkpointer=InMemorySaver())
    alice = {"configurable": {"thread_id": "alice"}}
    bob = {"configurable": {"thread_id": "bob"}}

    def ask(config: dict, text: str) -> str:
        result = cart_agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        return result["messages"][-1].text

    mo.ui.table(
        [
            {"thread": "alice", "answer": ask(alice, "Hi, I'm Alice. Please add 2 apples to my cart.")},
            {"thread": "alice", "answer": ask(alice, "Also add one banana. What have I added so far, and what's my name?")},
            {"thread": "bob", "answer": ask(bob, "What's my name and what is in my cart?")},
        ]
    )
    return alice, cart_agent


@app.cell
def _(alice, cart_agent):
    {"messages stored for alice": len(cart_agent.get_state(alice).values["messages"])}
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Runtime context available to tools

    `context_schema` passes per-request data (user id, permissions, DB handles) to
    tools without putting it in the prompt, where the model could alter it.
    """)
    return


@app.class_definition
@dataclass
class UserContext:
    user_id: str
    plan: str


@app.cell
def _():
    @tool
    def get_my_invoices(runtime: ToolRuntime[UserContext]) -> list[dict]:
        """Get the invoices of the current user."""
        user = runtime.context  # injected - not visible to / not settable by the model
        invoices = {
            "u1": [{"id": "INV-1", "amount": 49.0}, {"id": "INV-7", "amount": 49.0}],
            "u2": [{"id": "INV-3", "amount": 199.0}],
        }
        return [dict(inv, plan=user.plan) for inv in invoices.get(user.user_id, [])]

    billing_agent = create_agent(get_llm(), tools=[get_my_invoices], context_schema=UserContext)
    {
        _ctx.user_id: billing_agent.invoke(
            {"messages": [{"role": "user", "content": "What is the total of my invoices?"}]}, context=_ctx
        )["messages"][-1].text
        for _ctx in [UserContext("u1", "basic"), UserContext("u2", "pro")]
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Structured final response

    `response_format` makes the agent end with a typed object in
    `result["structured_response"]`, after it has used tools as needed.

    With a Pydantic class, `create_agent` uses a "tool strategy": `TripAdvice` is exposed
    as one more tool and calling it ends the run. Providers normally *force* that
    call via `tool_choice`; this gateway only accepts `tool_choice="auto"`, so we also
    tell the model explicitly how to finish.
    """)
    return


@app.class_definition
class TripAdvice(BaseModel):
    """Final recommendation for the user."""

    best_city: str = Field(description="The city with the best weather for a beach day")
    temperatures: dict[str, int] = Field(description="City -> temperature in Celsius")
    reason: str


@app.cell
def _():
    @tool
    def get_weather(city: str) -> dict:
        """Get the current weather for a city."""
        data = {"minsk": (14, "rain"), "barcelona": (27, "sunny"), "oslo": (6, "cloudy")}
        temp, sky = data.get(city.lower(), (20, "clear"))
        return {"city": city, "temp_c": temp, "sky": sky}

    weather_agent = create_agent(
        get_llm(),
        tools=[get_weather],
        response_format=TripAdvice,
        system_prompt="Use get_weather for every city. Then give your final answer by calling the TripAdvice tool.",
    )
    weather_result = weather_agent.invoke(
        {"messages": [{"role": "user", "content": "Compare the weather in Minsk, Barcelona and Oslo for a beach day."}]}
    )
    advice = weather_result.get("structured_response")
    if advice is None:
        # Fallback: turn the agent's free-text answer into the schema.
        advice = get_llm().with_structured_output(TripAdvice).invoke(weather_result["messages"][-1].text)
    advice
    return


if __name__ == "__main__":
    app.run()
