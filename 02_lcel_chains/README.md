# 02 · LCEL chains

LangChain Expression Language (LCEL) composes Runnables with `|`. A chain is itself a Runnable,
so it gets streaming, batching, async, retries and tracing for free.

| File | Covers |
|------|--------|
| [01_pipe_chain.py](01_pipe_chain.py) | `prompt \| llm \| StrOutputParser()` · batch and stream on a chain · `RunnableLambda` / plain functions as steps · input schema and graph introspection |
| [02_parallel_and_sequential.py](02_parallel_and_sequential.py) | `RunnableParallel` (concurrent branches) · `RunnablePassthrough.assign` to thread data through multi-step pipelines · `itemgetter` |
| [03_routing.py](03_routing.py) | Classify with structured output, then route to a specialised chain (function router and `RunnableBranch`) |
| [04_reliability_and_config.py](04_reliability_and_config.py) | `with_fallbacks` · `with_retry` · `RunnableConfig` (tags, metadata, `max_concurrency`) · `configurable_fields` |

## Cheat sheet

```python
chain = prompt | llm | StrOutputParser()                          # sequence
RunnableParallel(a=chain_a, b=chain_b)                            # same input, dict of outputs
RunnablePassthrough.assign(summary=summary_chain)                 # keep input, add a key
RunnableLambda(route_fn)                                          # function may return a Runnable
llm.with_fallbacks([backup_llm]); step.with_retry(stop_after_attempt=3)
chain.invoke(x, config={"max_concurrency": 4, "tags": [...], "callbacks": [...]})
```

When you need loops, shared state or human pauses, use LangGraph instead (see 07).
