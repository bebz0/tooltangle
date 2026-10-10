from types import SimpleNamespace

import pytest
from fakes import FakeChatModel
from pydantic import BaseModel

from tooltangle.dataset import Query
from tooltangle.models import (
    Cache,
    FallbackClient,
    ModelClient,
    ModelError,
    QuotaExhausted,
    UsageLog,
    cache_key,
    create_model,
    is_transient,
    parse_rate_limit,
)
from tooltangle.runner import Runner
from tooltangle.settings import DEFAULT_FALLBACKS, Settings
from tooltangle.toolset import Toolset, ToolSpec

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
OPENAI_429 = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for gpt-5.4-mini in "
    "organization org-abc on requests per min (RPM): Limit 500, Used 500, Requested 1. "
    "Please try again in 120ms.', 'type': 'requests', 'code': 'rate_limit_exceeded'}}"
)


class APIError(Exception):
    def __init__(self, message: str, status_code: int, code: str | None = None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.response = SimpleNamespace(headers=headers or {})


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


def test_status_code_wins_over_numbers_in_the_text():
    limit = parse_rate_limit(APIError(OPENAI_429, 429))
    assert (limit.retry_after, limit.limit, limit.per_day) == (pytest.approx(0.12), 500, False)
    assert parse_rate_limit(APIError(OPENAI_429.replace("min (RPM)", "day (RPD)"), 429)).per_day
    assert (
        parse_rate_limit(APIError("slow down", 429, headers={"retry-after": "7"})).retry_after == 7
    )

    confusing = APIError("max_output_tokens must be at most 500, the prompt is 429 tokens", 400)
    assert parse_rate_limit(confusing) is None
    assert not is_transient(confusing)
    assert is_transient(APIError("The server is overloaded", 503))
    assert is_transient(RuntimeError("Request timed out."))


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


async def test_an_account_without_credit_stops_immediately():
    error = APIError("You exceeded your current quota", 429, code="insufficient_quota")
    runnable = FlakyRunnable([error])
    with pytest.raises(QuotaExhausted, match="out of credit"):
        await make_client().invoke(runnable, [])
    assert runnable.calls == 1


def test_create_model_says_what_is_missing(monkeypatch):
    for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ModelError, match="GOOGLE_API_KEY"):
        create_model("google_genai:gemini-3.5-flash-lite")

    def not_installed(spec, **options):
        raise ImportError("install the provider package first")

    monkeypatch.setattr("tooltangle.models.init_chat_model", not_installed)
    with pytest.raises(ModelError, match=r"pip install 'tooltangle\[openai\]'"):
        create_model("openai:gpt-5.4-mini")
    with pytest.raises(ModelError, match="install the provider package first"):
        create_model("ollama:llama3")


def test_temperature_is_zero_unless_the_model_rejects_it(monkeypatch):
    created = []

    def fake_init(spec, **options):
        created.append(options)
        return FakeChatModel(profile={"temperature": "reasoning" not in spec})

    monkeypatch.setattr("tooltangle.models.init_chat_model", fake_init)
    create_model("fake:plain")
    assert created[-1] == {"temperature": 0}
    create_model("fake:reasoning")
    assert created[-1] == {}
    create_model("fake:plain", temperature=0.7)
    assert created[-1] == {"temperature": 0.7}


def test_prices_come_from_the_settings_first_then_from_the_price_list():
    usage = UsageLog({"openai:gpt-5-mini": (1.0, 2.0)})
    assert usage.price("openai:gpt-5-mini") == (1.0, 2.0)
    listed = usage.price("google_genai:gemini-2.5-flash")
    assert listed is not None and listed[0] > 0 and listed[1] > listed[0]
    assert usage.price("fake:model") is None


def test_default_fallbacks_stay_with_the_provider_of_the_generator():
    assert Settings().fallbacks == DEFAULT_FALLBACKS
    assert Settings(generator="openai:gpt-5.4").fallbacks == []
    assert Settings(generator_fallbacks=["openai:gpt-5.4-mini"]).fallbacks == [
        "openai:gpt-5.4-mini"
    ]


def test_model_options_are_part_of_the_cache_key():
    toolset = Toolset([ToolSpec("notes", "search", "Search notes")])
    query = Query(
        id="q", text="find it", tool="notes.search", accepted=["notes.search"], kind="plain"
    )
    plain = Runner(make_client(), toolset)
    tuned = Runner(
        ModelClient("fake:model", UsageLog(), model=FakeChatModel(), temperature=1), toolset
    )
    assert plain.cache_key(query) == cache_key(
        "pick", "fake:model", "", toolset.fingerprint(), "find it"
    )
    assert tuned.cache_key(query) != plain.cache_key(query)


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


async def test_fallback_moves_on_when_a_model_fails():
    def out_of_quota(schema, prompt):
        raise RuntimeError(GEMINI_DAILY_429)

    usage = UsageLog()
    exhausted = FakeChatModel(responder=out_of_quota)
    backup = FakeChatModel(responder=lambda schema, prompt: schema(value="backup"))
    fallback = FallbackClient(
        [
            ModelClient("a", usage, requests_per_minute=6000, model=exhausted),
            ModelClient("b", usage, requests_per_minute=6000, model=backup),
        ]
    )
    assert (await fallback.ask(Answer, "hi", "generate")).value == "backup"
    assert (await fallback.ask(Answer, "hello", "generate")).value == "backup"
    assert fallback.used == ["b"]
    assert exhausted.prompts == ["hi"]
    assert backup.prompts == ["hi", "hello"]
