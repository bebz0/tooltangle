import pytest
from fakes import FakeChatModel
from pydantic import BaseModel

from tooltangle.models import Cache, ModelClient, QuotaExhausted, UsageLog, parse_rate_limit

GEMINI_429 = (
    "Error calling model 'gemini-3.8-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "{'error': {'code': 429, 'message': 'You exceeded your current quota. * Quota exceeded for "
    "metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 5, "
    "model: gemini-3.8-flash\\nPlease retry in 0.02s.', 'status': 'RESOURCE_EXHAUSTED', "
    "'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': "
    "[{'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaValue': '5'}]}, "
    "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '0s'}]}}"
)
GEMINI_DAILY_429 = GEMINI_429.replace("PerMinute", "PerDay")


class Answer(BaseModel):
    value: str


class FlakyRunnable:
    def __init__(self, errors: list[Exception]):
        self.errors = errors
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def make_client() -> ModelClient:
    return ModelClient("fake:model", UsageLog(), requests_per_minute=6000, model=FakeChatModel())


def test_parse_rate_limit():
    limit = parse_rate_limit(RuntimeError(GEMINI_429))
    assert limit.retry_after == pytest.approx(0.02)
    assert limit.limit == 5
    assert not limit.per_day
    assert parse_rate_limit(RuntimeError(GEMINI_DAILY_429)).per_day
    assert parse_rate_limit(RuntimeError("429 - Please try again in 1.5s.")).retry_after == 1.5
    assert parse_rate_limit(RuntimeError("connection reset")) is None


async def test_retries_rate_limits_and_slows_down():
    client = make_client()
    runnable = FlakyRunnable([RuntimeError(GEMINI_429)])
    assert await client.invoke(runnable, []) == "ok"
    assert runnable.calls == 2
    assert client.pacer.interval == pytest.approx(12.6)


async def test_daily_quota_stops_immediately():
    runnable = FlakyRunnable([RuntimeError(GEMINI_DAILY_429)])
    with pytest.raises(QuotaExhausted, match="daily quota"):
        await make_client().invoke(runnable, [])
    assert runnable.calls == 1


async def test_ask_caches_structured_answers(tmp_path):
    model = FakeChatModel(responder=lambda schema, prompt: schema(value=prompt.upper()))
    usage = UsageLog()
    with Cache(tmp_path / "cache.sqlite") as cache:
        client = ModelClient("fake:model", usage, cache, requests_per_minute=6000, model=model)
        assert (await client.ask(Answer, "hi", "generate")).value == "HI"
        assert (await client.ask(Answer, "hi", "generate")).value == "HI"

    assert model.prompts == ["hi"]
    entry = usage.entries[("generate", "fake:model")]
    assert (entry.calls, entry.cached, entry.input_tokens) == (1, 1, 500)
