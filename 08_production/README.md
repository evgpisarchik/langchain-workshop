# 08 · Production concerns

Visibility, cost control and tests.

| File | Covers |
|------|--------|
| [01_callbacks_and_observability.py](01_callbacks_and_observability.py) | Custom `BaseCallbackHandler` (latency, tokens, tool calls) · `get_usage_metadata_callback` for token accounting · callbacks in agents · `astream_events` for UIs · LangSmith tracing via env vars |
| [02_caching_and_rate_limits.py](02_caching_and_rate_limits.py) | `set_llm_cache(InMemoryCache())` and per-model opt-out · `InMemoryRateLimiter` · `max_concurrency` for bulk jobs · async semaphore |
| [03_testing_with_fake_models.py](03_testing_with_fake_models.py) | `GenericFakeChatModel`: deterministic unit tests for chains and agents (including tool calls) with no network |

```bash
uv run pytest 08_production/03_testing_with_fake_models.py -q   # pytest collects the notebook's test_* functions
```

## Tracing with LangSmith

No code changes needed. Add to `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=langchain-examples
```
