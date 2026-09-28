import asyncio
import logging
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
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


@dataclass
class Usage:
    calls: int = 0
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
        requests_per_minute: float = 60,
        model: BaseChatModel | None = None,
        **options: Any,
    ):
        self.spec = spec
        self.usage = usage
        self.model = model or create_model(spec, **options)
        self.pacer = Pacer(requests_per_minute)

    async def invoke(self, runnable: Any, messages: list[BaseMessage]) -> Any:
        result, _ = await self.invoke_timed(runnable, messages)
        return result

    async def invoke_timed(self, runnable: Any, messages: list[BaseMessage]) -> tuple[Any, float]:
        await self.pacer.wait()
        started = time.perf_counter()
        try:
            result = await runnable.ainvoke(messages)
        except Exception as error:
            raise ModelError(f"{self.spec} failed: {error}") from error
        return result, time.perf_counter() - started

    async def ask[T: BaseModel](self, schema: type[T], prompt: str, phase: str) -> T:
        runnable = self.model.with_structured_output(schema, include_raw=True)
        last_error = None
        for _ in range(3):
            result = await self.invoke(runnable, [HumanMessage(prompt)])
            self.usage.record(phase, self.spec, result["raw"])
            parsed = result["parsed"]
            if isinstance(parsed, schema):
                return parsed
            last_error = result.get("parsing_error")
        raise ModelError(f"{self.spec} did not return a valid {schema.__name__}: {last_error}")
