import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
import warnings
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

warnings.filterwarnings("ignore", message=r".*uses fixed sampling defaults.*")
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# USD per 1M tokens (input, output), Gemini API paid tier, September 2026
KNOWN_PRICES = {
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-2.5-flash": (0.30, 2.50),
}

PROVIDERS_WITH_RETRIES = {"google_genai", "openai", "anthropic"}


class ModelError(Exception):
    pass


class QuotaExhausted(ModelError):
    pass


def model_name(spec: str) -> str:
    return spec.split(":", 1)[-1]


def create_model(spec: str, **options: Any) -> BaseChatModel:
    options.setdefault("temperature", 0)
    if spec.split(":", 1)[0] in PROVIDERS_WITH_RETRIES:
        options.setdefault("max_retries", 2)
    try:
        model: BaseChatModel = init_chat_model(spec, **options)
    except (ImportError, ValueError) as error:
        raise ModelError(f"can't create model {spec!r}: {error}") from error
    return model


def cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def get(self, key: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT value FROM responses WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass
class RateLimit:
    retry_after: float | None
    limit: int | None
    per_day: bool


def parse_rate_limit(error: Exception) -> RateLimit | None:
    text = str(error)
    if not re.search(r"\b429\b|RESOURCE_EXHAUSTED|rate.?limit", text, re.IGNORECASE):
        return None
    retry = re.search(r"(?:retry|try again) in ([\d.]+)\s*s", text, re.IGNORECASE)
    retry = retry or re.search(r"retryDelay'?\"?:\s*'?\"?([\d.]+)s", text)
    limit = re.search(r"quotaValue'?\"?:\s*'?\"?(\d+)", text) or re.search(r"limit: (\d+)", text)
    return RateLimit(
        retry_after=float(retry.group(1)) if retry else None,
        limit=int(limit.group(1)) if limit else None,
        per_day="PerDay" in text,
    )


def is_transient(error: Exception) -> bool:
    pattern = r"\b(500|502|503|504)\b|UNAVAILABLE|overloaded|high demand|DEADLINE_EXCEEDED"
    return bool(re.search(pattern, str(error), re.IGNORECASE))


def backoff(attempt: int) -> float:
    return float(min(60, 5 * 2**attempt))


class Pacer:
    def __init__(self, requests_per_minute: float):
        self.interval = 60 / requests_per_minute
        self.next_slot = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_slot - now)
            self.next_slot = max(now, self.next_slot) + self.interval
        if delay:
            await asyncio.sleep(delay)

    def slow_down(self, requests_per_minute: float) -> None:
        self.interval = max(self.interval, 60 / requests_per_minute * 1.05)

    def pause(self, seconds: float) -> None:
        self.next_slot = max(self.next_slot, time.monotonic() + seconds)


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class UsageLog:
    def __init__(self, prices: dict[str, tuple[float, float]] | None = None):
        self.prices = {**KNOWN_PRICES, **(prices or {})}
        self.entries: dict[tuple[str, str], Usage] = defaultdict(Usage)

    def record(self, phase: str, spec: str, message: AIMessage | None) -> None:
        usage = self.entries[(phase, spec)]
        usage.calls += 1
        metadata = message.usage_metadata if message is not None else None
        if metadata:
            usage.input_tokens += metadata["input_tokens"]
            usage.output_tokens += metadata["output_tokens"]

    def record_cached(self, phase: str, spec: str) -> None:
        self.entries[(phase, spec)].cached += 1

    def price(self, spec: str) -> tuple[float, float] | None:
        return self.prices.get(spec) or self.prices.get(model_name(spec))

    def cost(self, spec: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.price(spec)
        if price is None:
            return None
        return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000

    def total_cost(self) -> float | None:
        total = None
        for (_, spec), usage in self.entries.items():
            cost = self.cost(spec, usage.input_tokens, usage.output_tokens)
            if cost is None:
                return None
            total = (total or 0.0) + cost
        return total


class ModelClient:
    def __init__(
        self,
        spec: str,
        usage: UsageLog,
        cache: Cache | None = None,
        requests_per_minute: float = 60,
        max_attempts: int = 8,
        model: BaseChatModel | None = None,
        on_event: Callable[[str], None] | None = None,
        **options: Any,
    ):
        self.spec = spec
        self.usage = usage
        self.cache = cache
        self.model = model or create_model(spec, **options)
        self.pacer = Pacer(requests_per_minute)
        self.max_attempts = max_attempts
        self.on_event = on_event or (lambda message: None)

    async def invoke(self, runnable: Any, messages: list[BaseMessage]) -> Any:
        result, _ = await self.invoke_timed(runnable, messages)
        return result

    async def invoke_timed(self, runnable: Any, messages: list[BaseMessage]) -> tuple[Any, float]:
        last_error = None
        for attempt in range(self.max_attempts):
            await self.pacer.wait()
            started = time.perf_counter()
            try:
                result = await runnable.ainvoke(messages)
                return result, time.perf_counter() - started
            except Exception as error:
                last_error = error
                limit = parse_rate_limit(error)
                if limit is not None:
                    if limit.per_day:
                        per_day = f" ({limit.limit} requests per day)" if limit.limit else ""
                        raise QuotaExhausted(
                            f"the daily quota for {self.spec} is used up{per_day}; "
                            "finished calls are cached, so rerun later or pick another model"
                        ) from error
                    if limit.limit:
                        self.pacer.slow_down(limit.limit)
                    wait = limit.retry_after or backoff(attempt)
                    self.pacer.pause(wait)
                    self.on_event(f"{model_name(self.spec)} is rate limited, waiting {wait:.0f}s")
                elif is_transient(error):
                    wait = backoff(attempt)
                    self.pacer.pause(wait)
                    self.on_event(f"{model_name(self.spec)} is overloaded, retrying in {wait:.0f}s")
                else:
                    raise ModelError(f"{self.spec} failed: {error}") from error
        raise ModelError(
            f"{self.spec} kept failing after {self.max_attempts} attempts: {last_error}"
        )

    def cached_answer[T: BaseModel](self, schema: type[T], prompt: str, phase: str) -> T | None:
        if self.cache is None:
            return None
        hit = self.cache.get(cache_key("ask", self.spec, schema.model_json_schema(), prompt))
        if hit is None:
            return None
        self.usage.record_cached(phase, self.spec)
        return schema.model_validate(hit["parsed"])

    async def ask[T: BaseModel](self, schema: type[T], prompt: str, phase: str) -> T:
        cached = self.cached_answer(schema, prompt, phase)
        if cached is not None:
            return cached

        key = cache_key("ask", self.spec, schema.model_json_schema(), prompt)
        runnable = self.model.with_structured_output(schema, include_raw=True)
        last_error = None
        for _ in range(3):
            result = await self.invoke(runnable, [HumanMessage(prompt)])
            self.usage.record(phase, self.spec, result["raw"])
            parsed = result["parsed"]
            if isinstance(parsed, schema):
                if self.cache is not None:
                    self.cache.put(key, {"parsed": parsed.model_dump()})
                return parsed
            last_error = result.get("parsing_error")
        raise ModelError(f"{self.spec} did not return a valid {schema.__name__}: {last_error}")
