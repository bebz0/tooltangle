import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from tooltangle.generation import (
    ContrastRequests,
    Labels,
    Requests,
    RequestsByTool,
    ToolPair,
    ToolPairs,
    ToolRequests,
)

PHRASES = {
    "search_notes": "find what I jotted down about",
    "read_note": "open my memo titled",
    "create_note": "write down a fresh entry on",
    "search_files": "locate documents on disk matching",
    "send_email": "shoot a message to",
}
SUBJECTS = [
    "taxes",
    "the garden",
    "my trip to Rome",
    "quarterly budget",
    "dentist visit",
    "birthday party",
    "car repair",
    "the new laptop",
]

CONFUSING = ("taxes", "garden", "Rome", "budget", "birthday", "car repair")


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


def pick_by_phrase(text: str, tools: list[dict]) -> str | None:
    return next((name for name, phrase in PHRASES.items() if phrase in text), None)


def confused_picker(text: str, tools: list[dict]) -> str | None:
    # sends notes requests to search_files until the description of search_notes says "jotted"
    picked = pick_by_phrase(text, tools)
    descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in tools}
    misleading = "jotted" not in descriptions.get("search_notes", "")
    if picked == "search_notes" and misleading and any(word in text for word in CONFUSING):
        return "search_files"
    return picked


def requested_functions(prompt: str) -> list[str]:
    line = prompt.split("For each of these tools:")[1].splitlines()[0]
    return re.findall(r"`(\w+)`", line)


def fake_generator(schema, prompt: str):
    count = re.search(r"write (\d+)", prompt, re.IGNORECASE)
    count = int(count.group(1)) if count else 0
    if schema is ToolPairs:
        return ToolPairs(pairs=[ToolPair(first="search_notes", second="search_files")])
    if schema is ContrastRequests:
        first, second = re.search(r"mix up: `(\w+)` and `(\w+)`", prompt).groups()
        tag = "(more)" if "already exist" in prompt else "(tricky)"
        return ContrastRequests(
            first=[f"{PHRASES[first]} {subject} {tag}" for subject in SUBJECTS[:count]],
            second=[f"{PHRASES[second]} {subject} {tag}" for subject in SUBJECTS[:count]],
        )
    if schema is Requests:
        jokes = [f"tell me a joke about {subject}" for subject in SUBJECTS]
        return Requests(requests=jokes[:count])
    if schema is RequestsByTool:
        items = []
        for function in requested_functions(prompt):
            texts = [f"{PHRASES[function]} {subject}" for subject in SUBJECTS[: count - 1]]
            texts.append(f"please run {function} now")
            items.append(ToolRequests(tool=function, requests=texts))
        return RequestsByTool(tools=items)
    if schema is Labels:
        labels = []
        for number, text in re.findall(r"^(\d+)\. (.+)$", prompt, re.MULTILINE):
            tools = [name for name, phrase in PHRASES.items() if phrase in text]
            if "dentist" in text:
                tools = []
            labels.append({"number": int(number), "tools": tools})
        return Labels(labels=labels)
    raise AssertionError(schema)
