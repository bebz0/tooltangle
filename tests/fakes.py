from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


def usage(input_tokens: int = 100, output_tokens: int = 10) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class FakeChatModel(BaseChatModel):
    picker: Any = None
    responder: Any = None
    tools: list[dict] = Field(default_factory=list)
    prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools": [convert_to_openai_tool(tool) for tool in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = messages[-1].content
        self.prompts.append(text)
        name = self.picker(text, self.tools) if self.picker else None
        tool_calls = []
        if name:
            tool_calls.append({"name": name, "args": {}, "id": "call_0", "type": "tool_call"})
        message = AIMessage(
            content="" if name else "Sure.", tool_calls=tool_calls, usage_metadata=usage()
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        async def respond(messages):
            text = messages[-1].content
            self.prompts.append(text)
            parsed = self.responder(schema, text)
            raw = AIMessage(content="", usage_metadata=usage(500, 200))
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": None}
            return parsed

        return RunnableLambda(respond)
