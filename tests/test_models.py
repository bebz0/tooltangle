from fakes import FakeChatModel
from pydantic import BaseModel

from tooltangle.models import Cache, ModelClient, UsageLog


class Answer(BaseModel):
    value: str


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
