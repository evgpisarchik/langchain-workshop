"""Check that the gateway supports everything the examples need.

Run: uv run python -m shared.check
"""

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel

from shared.llm import HashingEmbeddings, get_embeddings, get_llm


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class City(BaseModel):
    name: str
    country: str


def main() -> None:
    llm = get_llm()
    checks = {
        "chat": lambda: llm.invoke("Reply with the word OK.").text,
        "streaming": lambda: "".join(c.text for c in llm.stream("Count 1 to 3.")),
        "tool calling": lambda: llm.bind_tools([add]).invoke("Use the tool to add 2 and 3.").tool_calls,
        "structured output": lambda: llm.with_structured_output(City).invoke("Paris"),
        "agent loop": lambda: create_agent(llm, [add]).invoke(
            {"messages": [{"role": "user", "content": "What is 19 + 23? Use the tool."}]}
        )["messages"][-1].text,
        "embeddings": lambda: (
            "local HashingEmbeddings fallback (embeddings service unreachable)"
            if isinstance(get_embeddings(), HashingEmbeddings)
            else f"{type(get_embeddings()).__name__}, dim={len(get_embeddings().embed_query('ping'))}"
        ),
    }
    for name, check in checks.items():
        try:
            print(f"[ OK ] {name:<18} {str(check())[:80]!r}")
        except Exception as error:  # noqa: BLE001
            print(f"[FAIL] {name:<18} {type(error).__name__}: {str(error)[:150]}")


if __name__ == "__main__":
    main()
