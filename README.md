# LangChain examples (OpenAI-compatible gateway)

A structured, runnable tour of the main LangChain 1.x and LangGraph use cases.
Every example uses a reasoning model on an OpenAI-compatible gateway (`OPENAI_BASE_URL`, `LLM_MODEL`),
and embeddings come from a multilingual embedding service (`EMBEDDINGS_URL`, internal network / VPN).

## Quick start

```bash
cp .env.example .env              # then put your key into OPENAI_API_KEY
uv sync                           # creates .venv with all dependencies
uv run python -m shared.check     # verify the gateway: chat, tools, structured output, agent
uv run marimo edit                # browse and open the notebooks in the browser
```

Every example is a [marimo](https://marimo.io) notebook. marimo notebooks are plain `.py` files, so they
also run as scripts (`uv run python 01_basics/01_chat_model.py`) and diff cleanly in git.
Open a single notebook with `uv run marimo edit 01_basics/01_chat_model.py`.

Notebooks don't run on open, and editing a cell marks its dependents stale instead of re-running
them (see `[tool.marimo]` in `pyproject.toml`), so LLM calls happen only when you ask.
Press "Run all" (or `Ctrl+Shift+R`) the first time you open a notebook.

> If another virtualenv is active (e.g. another project's), uv prints a warning and still uses
> this project's `.venv`. Run `deactivate` first, or prefix with `VIRTUAL_ENV=`, to hide it.

Run every notebook as a script, as a smoke test: `uv run python run_all.py` (or only some sections: `uv run python run_all.py 04 06`).

## Learning path

| # | Folder | What you learn |
|---|--------|----------------|
| 01 | [01_basics](01_basics/) | Chat models, messages, metadata, batch, streaming, async, prompt templates |
| 02 | [02_lcel_chains](02_lcel_chains/) | Composing with `\|`, parallel and sequential steps, routing, retries, fallbacks, runtime config |
| 03 | [03_structured_output](03_structured_output/) | Typed outputs with Pydantic, extraction, classification, output parsers |
| 04 | [04_tools_and_agents](04_tools_and_agents/) | Tools, the tool-calling loop, `create_agent`, memory, runtime context, middleware, human-in-the-loop |
| 05 | [05_memory](05_memory/) | Conversation history, persistent checkpoints, trimming and summarizing |
| 06 | [06_rag](06_rag/) | Loading and splitting, embeddings and vector stores, RAG chains with citations, multi-query, conversational and agentic RAG |
| 07 | [07_langgraph](07_langgraph/) | State graphs, loops, interrupts, time travel, map-reduce with `Send`, multi-agent supervisor |
| 08 | [08_production](08_production/) | Callbacks, token usage, event streaming, tracing, caching, rate limits, testing with fake models |

Each folder has its own README with a file-by-file breakdown. Files are numbered in reading order,
and every file is a standalone notebook: markdown cells explain each step, and many notebooks have
inputs (text boxes, sliders, buttons) so you can try your own prompts, questions and settings.

## Project layout

```
.
├── .env.example        # gateway URL, key, model, reasoning effort, embeddings URL
├── pyproject.toml      # uv project; `shared` is installed as a package
├── run_all.py          # runs every notebook as a script and reports PASS/FAIL
├── shared/
│   ├── llm.py          # get_llm() / get_embeddings() + gateway workarounds
│   ├── notebook.py     # stream_text() / astream_text(): render streamed tokens live in a cell
│   └── check.py        # `python -m shared.check` - gateway capability check
├── 01_basics/ … 08_production/
└── 06_rag/data/        # small fictional company handbook used by the RAG examples
```

All model construction goes through `shared.get_llm()`, so switching to another endpoint or
model means changing only `.env`. Any `ChatOpenAI` argument can be overridden per call, e.g.
`get_llm(temperature=1.0)` or `get_llm(reasoning={"effort": "high"})`.

## About the gateway (important)

The gateway is LiteLLM in front of vLLM serving an open-weight reasoning model. Plain
`ChatOpenAI` breaks on it in several ways. `shared/llm.py` defines `GatewayChatOpenAI`,
a small `ChatOpenAI` subclass that fixes these issues by rewriting requests and responses,
so the example code stays standard LangChain. Main issues found:

| Problem on the gateway | Symptom with plain ChatOpenAI | Workaround in `GatewayChatOpenAI` |
|---|---|---|
| Chat Completions doesn't parse tool calls | tool args come back as text | use the Responses API |
| Responses API drops tool results | agents loop forever calling the same tool | resend tool history as text |
| Only `tool_choice="auto"` is accepted | HTTP 400 for forced tool calls | drop other values |
| Only the first system message is used | later system messages (e.g. summaries) are ignored | merge system messages |
| Structured output: the schema is enforced but not shown to the model | wrong enum picks (e.g. always the first category) | add the schema to the prompt |
| Structured output with a system message or history | empty response | fold everything into user messages |
| Occasional vLLM parse errors or empty turns | random 400s or blank answers | retry up to 3 times |
| `/embeddings` returns HTTP 500 | RAG can't embed | embeddings come from a separate embedding service instead |

Also note:
- The gateway **caches responses**, so identical requests (even at `temperature=1`) may return identical output.
- Reasoning effort defaults to `low` (`LLM_REASONING_EFFORT`) to keep the examples fast.
- `RemoteEmbeddings` (in `shared/llm.py`) adds `passage:` / `query:` role prefixes (configurable), batches 32 texts per
  request, and bypasses `HTTP(S)_PROXY` because the service is internal. If it's unreachable (e.g. off
  VPN), `get_embeddings()` falls back to a lexical `HashingEmbeddings`: word matching, not meaning.

## Versions

Tested with Python 3.14, `marimo` 0.25, `langchain` 1.4, `langchain-core` 1.6, `langchain-openai` 1.6 and `langgraph` 1.2.
`langchain-community` is being sunset upstream and is intentionally not used.
