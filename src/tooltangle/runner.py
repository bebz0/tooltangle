import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from tooltangle.dataset import Query
from tooltangle.models import ModelClient, cache_key
from tooltangle.toolset import Toolset


@dataclass
class Pick:
    query_id: str
    tool: str | None
    calls: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0
    error: str | None = None


class Runner:
    def __init__(
        self,
        client: ModelClient,
        toolset: Toolset,
        system_prompt: str = "",
        concurrency: int = 8,
        on_pick: Callable[[], None] = lambda: None,
    ):
        self.client = client
        self.toolset = toolset
        self.system_prompt = system_prompt
        self.bound = client.model.bind_tools(toolset.function_schemas())
        self.fingerprint = toolset.fingerprint()
        self.semaphore = asyncio.Semaphore(concurrency)
        self.on_pick = on_pick

    def cache_key(self, query: Query) -> str:
        return cache_key("pick", self.client.spec, self.system_prompt, self.fingerprint, query.text)

    def uncached(self, queries: list[Query]) -> list[Query]:
        if self.client.cache is None:
            return list(queries)
        return [query for query in queries if self.client.cache.get(self.cache_key(query)) is None]

    async def run(self, queries: list[Query]) -> dict[str, Pick]:
        picks = await asyncio.gather(*(self.pick(query) for query in queries))
        return {pick.query_id: pick for pick in picks}

    async def pick(self, query: Query) -> Pick:
        key = self.cache_key(query)
        cache = self.client.cache
        if cache is not None and (hit := cache.get(key)):
            self.client.usage.record_cached("select", self.client.spec)
            self.on_pick()
            return Pick(query_id=query.id, **hit)

        messages: list[BaseMessage] = [HumanMessage(query.text)]
        if self.system_prompt:
            messages.insert(0, SystemMessage(self.system_prompt))
        async with self.semaphore:
            try:
                message, seconds = await self.client.invoke_timed(self.bound, messages)
            except Exception as error:
                self.on_pick()
                return Pick(query.id, None, error=str(error)[:500])

        self.client.usage.record("select", self.client.spec, message)
        pick = self.read_pick(query.id, message, seconds)
        if cache is not None:
            stored = asdict(pick)
            del stored["query_id"]
            cache.put(key, stored)
        self.on_pick()
        return pick

    def read_pick(self, query_id: str, message: AIMessage, seconds: float) -> Pick:
        names = [call["name"] for call in message.tool_calls]
        names += [name for call in message.invalid_tool_calls if (name := call.get("name"))]
        calls = [self.toolset.key_for(name) or f"?{name}" for name in names]
        usage = message.usage_metadata
        return Pick(
            query_id=query_id,
            tool=calls[0] if calls else None,
            calls=calls,
            input_tokens=usage["input_tokens"] if usage else 0,
            output_tokens=usage["output_tokens"] if usage else 0,
            latency=round(seconds, 3),
        )
