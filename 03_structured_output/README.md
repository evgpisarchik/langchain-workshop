# 03 · Structured output

Get validated Python objects instead of free text: the basis of extraction, classification,
routing and any LLM step whose output feeds into code.

| File | Covers |
|------|--------|
| [01_with_structured_output.py](01_with_structured_output.py) | Pydantic, nested models, `Literal` enums, `Optional` · TypedDict / JSON schema → dict · `include_raw=True` · `method="function_calling"` |
| [02_extraction.py](02_extraction.py) | Use case: meeting notes → decisions and action items (with dates) · batch extraction of orders from emails |
| [03_classification.py](03_classification.py) | Use case: tag reviews (sentiment, topics, language, urgency) and aggregate the results |
| [04_output_parsers.py](04_output_parsers.py) | Prompt-based parsing: `PydanticOutputParser`, streaming partial JSON with `JsonOutputParser`, `CommaSeparatedListOutputParser` |

## Tips

- Docstrings and `Field(description=...)` are sent to the model, so write them as instructions.
- Use `Literal[...]` for closed sets of labels, and `Optional[...] = None` for "if mentioned" fields, to discourage made-up values.
- Put today's date in the prompt if relative dates ("next Friday") must be resolved.
- On this gateway, `json_schema` (the default) and `function_calling` work; `json_mode` doesn't.
  `shared/llm.py` makes sure the model actually sees the schema (see the root README).
