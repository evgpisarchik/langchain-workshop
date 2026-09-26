"""Model factories for the examples, wired to an OpenAI-compatible gateway.

Every example calls `get_llm()` / `get_embeddings()` instead of constructing
models directly, so the endpoint lives in one place (`.env`).

The gateway (LiteLLM -> vLLM serving an open-weight reasoning model) has quirks
that break standard LangChain usage. `GatewayChatOpenAI` hides them so the
examples can stay idiomatic. Found by probing the gateway (see README):

Tools / agents
 1. Tool calls are parsed only on the Responses API -> `use_responses_api=True`.
 2. Tool results (`function_call_output` items) are silently dropped, so an agent
    would call the same tool forever -> past tool calls and results are rewritten
    as plain-text messages before each request. New tool calls from the model stay
    native `tool_calls`, so bind_tools / create_agent / LangGraph work unchanged.
 3. Only `tool_choice="auto"` is accepted -> other values are dropped, so "forced"
    tool calls become strong hints rather than guarantees.

Messages
 4. Assistant turns with `output_text` parts are rejected (no id/status) -> sent as
    plain strings; reasoning items are not replayed.
 5. Only the *first* system message is honoured -> all system messages are merged.

Structured output (json_schema)
 6. The JSON grammar is enforced but the schema is never shown to the model, so
    field descriptions / enum meanings are lost -> the schema is added to the prompt.
 7. Any system message, or any assistant turn, makes the output come back empty,
    and the `instructions` field is ignored -> system prompts, the conversation
    (as a transcript) and the schema are folded into user messages.
 8. The JSON is sometimes mis-filed as an `mcp_call` output item -> moved back.
 (`json_mode` is unsupported by vLLM here; `function_calling` works too.)

Reliability (non-streaming calls are retried up to 3 times)
 9. Transient HTTP 400 "Unexpected token ... expecting start token".
10. Empty turns: reasoning only, no text and no tool call.

Embeddings
11. The gateway's embeddings route returns HTTP 500, so embeddings come from the
    multilingual embedding service instead (EMBEDDINGS_URL). Without it
    (e.g. off VPN) `get_embeddings()` falls back to a local lexical HashingEmbeddings.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"{name} is not set - copy .env.example to .env and fill it in")
    return value


# --------------------------------------------------------------------------- chat


def _flatten_tool_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Replace AI tool-call messages and ToolMessages with plain-text equivalents."""
    result: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            lines = [message.text] if message.text else []
            lines += [
                f"[I called tool `{call['name']}` with arguments "
                f"{json.dumps(call['args'], ensure_ascii=False)} (call id {call['id']})]"
                for call in message.tool_calls
            ]
            result.append(AIMessage(content="\n".join(lines)))
        elif isinstance(message, ToolMessage):
            name = f" `{message.name}`" if message.name else ""
            result.append(
                HumanMessage(
                    content=f"[Result of tool{name} (call id {message.tool_call_id})]:\n{message.text}"
                )
            )
        else:
            result.append(message)
    return result


def _is_transient_parse_error(error: Exception) -> bool:
    # vLLM occasionally fails to parse the model's output under load and returns
    # HTTP 400 "Unexpected token ... while expecting start token". Retrying succeeds.
    return "Unexpected token" in str(error)


def _recover_constrained_json(result: Any) -> Any:
    """vLLM sometimes mis-parses the model's `<|constrain|>json` header and returns the JSON
    as an `mcp_call` output item instead of a message. Move it back into the content."""
    message = result.generations[0].message
    if message.text.strip():
        return result
    outputs = message.additional_kwargs.get("tool_outputs", [])
    for item in outputs:
        if item.get("type") == "mcp_call" and "constrain" in str(item.get("name", "")):
            message.content = item.get("arguments", "")
            try:
                message.additional_kwargs["parsed"] = json.loads(message.content)
            except json.JSONDecodeError:
                pass
            message.additional_kwargs["tool_outputs"] = [o for o in outputs if o is not item]
            break
    return result


def _is_empty(result: Any) -> bool:
    # The model sometimes ends a turn with reasoning only: no text and no tool call.
    message = result.generations[0].message
    return not message.text.strip() and not message.tool_calls


class GatewayChatOpenAI(ChatOpenAI):
    """ChatOpenAI that works around the gateway quirks listed in the module doc."""

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(3):
            try:
                result = _recover_constrained_json(super()._generate(*args, **kwargs))
            except Exception as error:
                if attempt == 2 or not _is_transient_parse_error(error):
                    raise
                continue
            if attempt == 2 or not _is_empty(result):
                return result

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(3):
            try:
                result = _recover_constrained_json(await super()._agenerate(*args, **kwargs))
            except Exception as error:
                if attempt == 2 or not _is_transient_parse_error(error):
                    raise
                continue
            if attempt == 2 or not _is_empty(result):
                return result

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = _flatten_tool_history(self._convert_input(input_).to_messages())
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        if payload.get("tool_choice") not in (None, "auto"):
            # The gateway rejects any tool_choice other than "auto"
            payload.pop("tool_choice")
        if isinstance(payload.get("input"), list):
            payload["input"] = [_simplify_input_item(item) for item in payload["input"]
                                if item.get("type") != "reasoning"]
            schema = _requested_json_schema(payload)
            if schema:
                # 1. The gateway enforces the JSON grammar but does not show the schema to the
                #    model, so field descriptions / enum meanings would be invisible.
                # 2. With a JSON schema, any system/developer item in `input` makes vLLM emit
                #    an empty `mcp_call` instead of a message, and the gateway ignores the
                #    `instructions` field. So both go into a leading user message.
                system_texts = [
                    item["content"] for item in payload["input"]
                    if item.get("role") in ("system", "developer") and isinstance(item.get("content"), str)
                ]
                payload["input"] = [
                    item for item in payload["input"] if item.get("role") not in ("system", "developer")
                ]
                if any(item.get("role") == "assistant" for item in payload["input"]):
                    # 3. ...and assistant turns make the output come back empty. Send the
                    #    conversation as a transcript inside a single user message instead.
                    transcript = "\n".join(
                        f"{item.get('role', 'user')}: {item.get('content')}" for item in payload["input"]
                    )
                    payload["input"] = [{"role": "user", "content": f"Conversation so far:\n{transcript}"}]
                preamble = "\n\n".join(
                    system_texts
                    + ["Respond with a JSON object that matches this JSON schema:\n"
                       + json.dumps(schema, ensure_ascii=False)]
                )
                payload["input"].insert(0, {"role": "user", "content": preamble})
            else:
                payload["input"] = _merge_system_items(payload["input"])
        return payload


def _merge_system_items(items: list[dict]) -> list[dict]:
    """The gateway keeps only the first system message - merge them all into one."""
    system = [i for i in items if i.get("role") in ("system", "developer") and isinstance(i.get("content"), str)]
    if len(system) < 2:
        return items
    merged = {"role": "system", "content": "\n\n".join(i["content"] for i in system)}
    return [merged] + [i for i in items if i not in system]


def _requested_json_schema(payload: dict) -> dict | None:
    text_format = payload.get("text_format")  # a Pydantic class (with_structured_output(Model))
    if isinstance(text_format, type) and issubclass(text_format, BaseModel):
        return text_format.model_json_schema()
    fmt = (payload.get("text") or {}).get("format") or {}  # a dict / TypedDict schema
    if fmt.get("type") == "json_schema":
        return fmt.get("schema")
    return None


def _simplify_input_item(item: dict) -> dict:
    """vLLM rejects assistant items with `output_text` parts but no id/status - send plain text."""
    if item.get("role") == "assistant" and isinstance(item.get("content"), list):
        text = "".join(part.get("text", "") for part in item["content"] if isinstance(part, dict))
        return {"type": "message", "role": "assistant", "content": text}
    return item


def get_llm(**overrides: Any) -> ChatOpenAI:
    """Return the chat model. Any ChatOpenAI kwarg can be overridden, e.g. temperature=0.7."""
    params: dict[str, Any] = {
        "model": _env("LLM_MODEL"),
        "base_url": _env("OPENAI_BASE_URL"),
        "api_key": _env("OPENAI_API_KEY"),
        "temperature": 0,
        "timeout": 120,
        "max_retries": 2,
        "use_responses_api": True,
        # Keep AIMessage.content a list of blocks + reasoning in additional_kwargs.
        "output_version": "v0",
        "reasoning": {"effort": os.getenv("LLM_REASONING_EFFORT", "low")},
    }
    params.update(overrides)
    return GatewayChatOpenAI(**params)


# --------------------------------------------------------------------- embeddings


_STOPWORDS = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being "
    "do does did i me my we our you your he she it its they them their this that these those "
    "what which who whom how when where why can could should would will shall may might must "
    "not no so than too very just about into over under up down out there here have has had".split()
)


class HashingEmbeddings(Embeddings):
    """Tiny dependency-free embedder: hashed word + character-trigram counts.

    It captures lexical overlap only, not meaning. Also a handy example of how
    to implement the `Embeddings` interface yourself.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS]
        trigrams = [w[i : i + 3] for w in words if len(w) > 3 for i in range(len(w) - 2)]
        return words + trigrams

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in self._features(text):
            digest = hashlib.md5(feature.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vector[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class RemoteEmbeddings(Embeddings):
    """A multilingual embedding model behind an HTTP inference endpoint (KServe v2 protocol).

    Some embedding models expect a role prefix on the input, e.g. "passage: " for
    indexed documents and "query: " for search queries. Vectors are L2-normalised.
    """

    def __init__(
        self,
        url: str,
        batch_size: int = 32,
        timeout: float = 30,
        document_prefix: str = "passage: ",
        query_prefix: str = "query: ",
    ) -> None:
        self.url = url
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.batch_size = batch_size  # max batch size accepted by the service
        # trust_env=False: the service is on the internal network, bypass HTTP(S)_PROXY.
        self.client = httpx.Client(timeout=timeout, trust_env=False)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            body = {"inputs": [{"name": "TEXT", "shape": [len(batch), 1], "datatype": "BYTES", "data": batch}]}
            response = self.client.post(self.url, json=body)
            response.raise_for_status()
            output = response.json()["outputs"][0]
            dim = output["shape"][1]
            flat = output["data"]
            for j in range(len(batch)):
                vector = flat[j * dim : (j + 1) * dim]
                norm = math.sqrt(sum(v * v for v in vector)) or 1.0
                vectors.append([v / norm for v in vector])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """RemoteEmbeddings at EMBEDDINGS_URL; HashingEmbeddings if it is unset or unreachable (e.g. off VPN)."""
    url = os.getenv("EMBEDDINGS_URL")
    if url:
        remote = RemoteEmbeddings(url)
        try:
            remote.embed_query("ping")
            return remote
        except Exception as error:  # noqa: BLE001 - any failure means "use the fallback"
            print(f"[shared] embeddings at {url} unavailable ({type(error).__name__}); "
                  "using local HashingEmbeddings")
    return HashingEmbeddings()
