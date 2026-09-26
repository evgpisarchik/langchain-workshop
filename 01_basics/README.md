# 01 · Basics

The building blocks: a chat model, messages, and prompt templates.

| File | Covers |
|------|--------|
| [01_chat_model.py](01_chat_model.py) | `invoke` with a string, message objects, tuples or dicts · `SystemMessage` / `HumanMessage` / `AIMessage` · `usage_metadata`, `response_metadata`, model reasoning · per-call settings (temperature, reasoning effort) · `batch` |
| [02_streaming_and_async.py](02_streaming_and_async.py) | `stream` token by token · adding chunks together · `ainvoke` + `asyncio.gather` · `astream` |
| [03_prompt_templates.py](03_prompt_templates.py) | `PromptTemplate` · `ChatPromptTemplate` · `MessagesPlaceholder` for history · `partial` · few-shot prompting |

```bash
uv run marimo edit 01_basics/01_chat_model.py
```

## Key ideas

- Everything in LangChain is a **Runnable**, with the same `invoke / batch / stream` interface and async twins (`ainvoke`, `abatch`, `astream`).
- A model takes a list of messages and returns an `AIMessage`. Use `.text` for the plain text; `.content` can be a list of content blocks.
- Prompt templates are Runnables too. They turn a dict of variables into messages, so `template | llm` just works (see 02).
