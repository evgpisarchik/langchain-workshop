# 04 · Tools and agents

An agent is an LLM in a loop: it decides which tool to call, sees the result, and repeats
until it can answer. In LangChain 1.x, `create_agent` builds that loop as a LangGraph graph.

| File | Covers |
|------|--------|
| [01_defining_tools.py](01_defining_tools.py) | `@tool`, `parse_docstring`, `Annotated` argument docs, Pydantic `args_schema`, `StructuredTool`, `ToolException` error handling, the JSON schema sent to the model (no LLM calls) |
| [02_tool_calling_loop.py](02_tool_calling_loop.py) | `bind_tools`, reading `tool_calls`, writing the agent loop by hand with `ToolMessage` |
| [03_create_agent.py](03_create_agent.py) | `create_agent` with several tools · the message trajectory · `stream_mode="updates"` and `"messages"` (tokens) · graph diagram |
| [04_agent_memory_and_structured_output.py](04_agent_memory_and_structured_output.py) | Checkpointer + `thread_id` for conversations · `context_schema` + `ToolRuntime` for per-user data in tools · `response_format` for a typed final answer |
| [05_middleware.py](05_middleware.py) | `@dynamic_prompt` by user role · `@before_model` logging · `@wrap_tool_call` error handling · `ModelCallLimitMiddleware` · `PIIMiddleware` (email redaction) · `SummarizationMiddleware` |
| [06_human_in_the_loop.py](06_human_in_the_loop.py) | `HumanInTheLoopMiddleware`: pause before sensitive tools; approve, edit or reject; resume with `Command(resume=...)`. The last section is an interactive form where you make the decision |

## Mental model

```
            ┌──────────── tool_calls? ───────────┐
user ──▶ [model] ──no──▶ answer              yes │
            ▲                                    ▼
            └──────────── ToolMessages ◀── [tools]
```

Middleware hooks into this loop: `before_model`, `wrap_model_call`, `wrap_tool_call`, `after_model`.

## Gateway notes

- The gateway can't force a tool call (`tool_choice`), so `response_format` relies on the model
  choosing to call the output tool. 04 adds a system-prompt hint and a fallback for this.
- The model likes non-ASCII hyphens (`‑`). Avoid IDs like `ACC-1` in tool arguments; prefer `checking` or `A100`.
