import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import asyncio
    import time

    import marimo as mo
    from langchain_core.caches import InMemoryCache
    from langchain_core.globals import set_llm_cache
    from langchain_core.rate_limiters import InMemoryRateLimiter

    from shared import get_llm


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Cost and load control: response caching, rate limiting, concurrency limits

    Open: `uv run marimo edit 08_production/02_caching_and_rate_limits.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. LLM cache

    Identical requests (same messages + same model params) are served from the cache
    without calling the API. `InMemoryCache` lives per process; for a shared cache
    implement `BaseCache` on Redis/SQL (or use a provider package).
    Note: this gateway also has its own server-side cache.

    `cache=False` opts a model out, e.g. for creative generations that should differ every time.
    """)
    return


@app.cell
def _():
    set_llm_cache(InMemoryCache())
    try:
        _llm = get_llm()
        _calls = []
        for _attempt in range(2):
            _start = time.perf_counter()
            _answer = _llm.invoke("Name the largest moon of Jupiter. One word.").text
            _calls.append({"call": _attempt + 1, "answer": _answer, "seconds": round(time.perf_counter() - _start, 3)})
        _uncached = get_llm(cache=False, temperature=1.0).invoke("Invent a pirate name. Name only.").text
    finally:
        set_llm_cache(None)
    mo.vstack([mo.ui.table(_calls), mo.md(f"**uncached:** {_uncached}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `InMemoryRateLimiter` (1 request / second)

    Client-side rate limiting: a token bucket shared by everything using this model.
    """)
    return


@app.cell
def _():
    _limiter = InMemoryRateLimiter(requests_per_second=1, check_every_n_seconds=0.1, max_bucket_size=1)
    _limited_llm = get_llm(rate_limiter=_limiter)
    _start = time.perf_counter()
    for _i in range(3):
        _limited_llm.invoke(f"Say the number {_i}.")
        mo.output.append(mo.md(f"request {_i} done at t={time.perf_counter() - _start:.1f}s"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `max_concurrency` in `batch` / `abatch`

    Concurrency limits for bulk jobs.
    """)
    return


@app.cell
def _():
    llm = get_llm()
    _inputs = [f"Translate the number {n} into French words. Words only." for n in range(1, 9)]
    _rows = []
    for _concurrency in (1, 8):
        _start = time.perf_counter()
        _results = llm.batch(_inputs, config={"max_concurrency": _concurrency})
        _rows.append(
            {
                "max_concurrency": _concurrency,
                "seconds": round(time.perf_counter() - _start, 1),
                "first answers": [r.text for r in _results][:4],
            }
        )
    mo.ui.table(_rows)
    return (llm,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. `asyncio.Semaphore`

    Async + a semaphore gives you the same control in your own async code.
    """)
    return


@app.cell
async def _(llm):
    _semaphore = asyncio.Semaphore(3)

    async def _one(n: int) -> str:
        async with _semaphore:
            return (await llm.ainvoke(f"{n} squared? Digits only.")).text

    {"squares": await asyncio.gather(*(_one(n) for n in range(1, 7)))}
    return


if __name__ == "__main__":
    app.run()
