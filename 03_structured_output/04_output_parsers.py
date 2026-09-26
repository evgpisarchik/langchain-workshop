import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from langchain_core.output_parsers import (
        CommaSeparatedListOutputParser,
        JsonOutputParser,
        PydanticOutputParser,
    )
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    from shared import get_llm

    llm = get_llm()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Output parsers: prompt-based structure

    For models without native JSON support. `with_structured_output` is preferred when
    the provider supports it. Parsers are still useful for any model, for streaming
    partial JSON and for simple formats.

    Open: `uv run marimo edit 03_structured_output/04_output_parsers.py`
    """)
    return


@app.class_definition
class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    minutes: int = Field(description="Total cooking time in minutes")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. `PydanticOutputParser`

    Injects format instructions into the prompt and validates the result.
    """)
    return


@app.cell
def _():
    parser = PydanticOutputParser(pydantic_object=Recipe)
    prompt = ChatPromptTemplate.from_template(
        "Give me a simple recipe for {dish}.\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    mo.vstack([
        mo.accordion({"Format instructions sent to the model": mo.md(f"```\n{parser.get_format_instructions()}\n```")}),
        (prompt | llm | parser).invoke({"dish": "pancakes"}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. `JsonOutputParser` - streaming partial JSON

    The parser yields *partial* JSON objects as they are generated. Watch the object grow.
    """)
    return


@app.cell
def _():
    json_parser = JsonOutputParser(pydantic_object=Recipe)
    json_prompt = ChatPromptTemplate.from_template(
        "Give me a recipe for {dish}.\n{format_instructions}"
    ).partial(format_instructions=json_parser.get_format_instructions())
    partial_versions = 0
    final_recipe = {}
    for _partial in (json_prompt | llm | json_parser).stream({"dish": "guacamole"}):
        final_recipe = _partial
        partial_versions += 1
        mo.output.replace(mo.json(_partial))
    mo.output.replace(mo.vstack([mo.md(f"{partial_versions} partial objects streamed; final:"), final_recipe]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. `CommaSeparatedListOutputParser`

    Simple formats.
    """)
    return


@app.cell
def _():
    list_parser = CommaSeparatedListOutputParser()
    list_prompt = ChatPromptTemplate.from_template(
        "List 5 {things}.\n{format_instructions}"
    ).partial(format_instructions=list_parser.get_format_instructions())
    (list_prompt | llm | list_parser).invoke({"things": "programming languages created before 1980"})
    return


if __name__ == "__main__":
    app.run()
