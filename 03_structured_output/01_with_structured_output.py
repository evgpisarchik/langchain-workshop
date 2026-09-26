import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from typing import Literal, Optional, TypedDict

    import marimo as mo
    from pydantic import BaseModel, Field

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # `with_structured_output`: typed objects instead of free text

    The schema can be a Pydantic model (returns an instance), a `TypedDict` or a
    JSON-schema dict (return a dict). Docstrings and `Field` descriptions are sent
    to the model, so write them like instructions.

    Open: `uv run marimo edit 03_structured_output/01_with_structured_output.py`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Pydantic schema

    Pydantic model -> validated Python object.
    """)
    return


@app.class_definition
class Movie(BaseModel):
    """Information about a movie."""

    title: str = Field(description="Original title")
    year: int = Field(description="Release year")
    director: str
    genres: list[str] = Field(description="1-3 genres, lowercase")
    rating: float = Field(description="Your rating from 0 to 10", ge=0, le=10)


@app.cell
def _():
    movie_title = mo.ui.text(value="Inception", label="Movie")
    movie_title
    return (movie_title,)


@app.cell
def _(movie_title):
    movie = llm.with_structured_output(Movie).invoke(f"Tell me about the movie {movie_title.value}.")
    mo.vstack([mo.md(f"**{movie.title}** ({movie.year})"), movie])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Nested models, enums (`Literal`) and optional fields
    """)
    return


@app.class_definition
class Address(BaseModel):
    city: str
    country: str


@app.class_definition
class Company(BaseModel):
    """A company mentioned in the text."""

    name: str
    industry: Literal["tech", "finance", "retail", "health", "other"]
    headquarters: Address
    founded: Optional[int] = Field(default=None, description="Founding year if known, else null")
    ceo: Optional[str] = Field(default=None, description="CEO name if mentioned, else null")


@app.cell
def _():
    company_text = "Spotify, the Swedish music streaming company from Stockholm, was started in 2006."
    llm.with_structured_output(Company).invoke(company_text).model_dump()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `TypedDict` / JSON schema -> plain dict
    """)
    return


@app.class_definition
class Sentiment(TypedDict):
    """Sentiment of a review."""

    label: Literal["positive", "neutral", "negative"]
    confidence: float


@app.cell
def _():
    json_schema = {
        "title": "Keywords",
        "description": "Keywords extracted from the text",
        "type": "object",
        "properties": {"keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 5}},
        "required": ["keywords"],
    }
    {
        "typeddict": llm.with_structured_output(Sentiment).invoke("The food was cold and the waiter was rude."),
        "json schema": llm.with_structured_output(json_schema).invoke(
            "LangGraph lets you build stateful multi-actor LLM applications with cycles and persistence."
        ),
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. `include_raw=True`

    Also get the raw `AIMessage` (usage, reasoning) and any parsing error.
    """)
    return


@app.cell
def _():
    raw_result = llm.with_structured_output(Movie, include_raw=True).invoke("Tell me about The Matrix.")
    {
        "parsed": raw_result["parsed"],
        "parsing_error": raw_result["parsing_error"],
        "usage": raw_result["raw"].usage_metadata,
    }
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. `method="function_calling"`

    Methods: `"json_schema"` (default here: constrained decoding on the server) and
    `"function_calling"` (the schema is sent as a tool). Both work on this gateway;
    `"json_mode"` is not supported by vLLM here.
    """)
    return


@app.cell
def _():
    llm.with_structured_output(Movie, method="function_calling").invoke("Tell me about Alien (1979).")
    return


if __name__ == "__main__":
    app.run()
