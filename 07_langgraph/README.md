# 07 · LangGraph

LangGraph is the runtime under `create_agent`, and the tool for custom workflows that need
explicit control: branches, loops, parallelism, persistence and human pauses.

| File | Pattern | Covers |
|------|---------|--------|
| [01_state_graph.py](01_state_graph.py) | Pipeline | `StateGraph`, TypedDict state, nodes and edges, reducers (`Annotated[list, operator.add]`), parallel fan-out and fan-in, streaming updates |
| [02_conditional_loop.py](02_conditional_loop.py) | Evaluator–optimizer | `add_conditional_edges`, cycles, an LLM grader with structured output, iteration cap |
| [03_interrupt_and_time_travel.py](03_interrupt_and_time_travel.py) | Human approval | `interrupt()` inside a node, `Command(resume=...)`, `get_state_history`, forking from an old checkpoint with `update_state` |
| [04_map_reduce.py](04_map_reduce.py) | Map-reduce | `Send` API for a dynamic number of parallel workers; summarize documents, then combine |
| [05_multi_agent_supervisor.py](05_multi_agent_supervisor.py) | Multi-agent | Supervisor agent delegating to specialist agents wrapped as tools |

## When to pick what

| Situation | Tool |
|-----------|------|
| Fixed linear or parallel steps | LCEL chain (02) |
| LLM decides which tools to use | `create_agent` (04) |
| Custom control flow, loops, approvals, durable state | `StateGraph` (here) |
| Several agents with separate prompts and tools | Supervisor with agents as tools (05) |
