# 05 · Memory

LLMs are stateless. "Memory" means storing messages and sending the relevant ones again,
which also means managing how much of the context window they use.

| File | Covers |
|------|--------|
| [01_chat_history.py](01_chat_history.py) | A manual message list · LangGraph `MessagesState` + `InMemorySaver` per `thread_id` · `SqliteSaver`: memory that survives restarts (re-run the cell) |
| [02_trim_and_summarize.py](02_trim_and_summarize.py) | `trim_messages` (keep the last N tokens, cheap but lossy) · running summary of old turns (keeps the facts) |

## Which to use

| Need | Use |
|------|-----|
| Chat app or agent with sessions | Checkpointer (`InMemorySaver` → `SqliteSaver` / `PostgresSaver`) + `thread_id` |
| Long conversations | `trim_messages`, or `SummarizationMiddleware` for agents (see 04/05) |
| Facts across sessions (user profile, preferences) | A LangGraph `Store` (long-term memory) |

`RunnableWithMessageHistory` and `InMemoryChatMessageHistory` are deprecated since langchain-core 1.6.
Use LangGraph persistence instead.

`memory.sqlite` is created next to the notebook; delete it to reset.
