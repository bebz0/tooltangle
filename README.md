# tooltangle

[![ci](https://github.com/bebz0/tooltangle/actions/workflows/ci.yml/badge.svg)](https://github.com/bebz0/tooltangle/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/bebz0/tooltangle/blob/main/LICENSE)

Find out which tools your LLM agent mixes up, and fix their descriptions with proof.

Connect five MCP servers to an agent and the model is choosing between forty tools on every
message. Each server's author tested their own tools, but nobody tested the mix you ended up
with. tooltangle does: it writes realistic user messages for every tool, runs your model on
them with all the tools attached, and shows which tools get confused with which. It can then
rewrite the confusing descriptions, keeping a rewrite only if the model does measurably better
on messages the rewrite was never shown.

Part of a real run against the reference `time` server, which has just two tools:

```
ERROR  time.convert_time → time.get_current_time  44% (7 of 16)
       e.g. "Our CEO wants to host an all-hands at 9:00 AM Eastern Time next Tuesday. What would that time correspond to in Dubai?"

accuracy  0.82 [0.67, 0.91] on 38 messages with gemini-3.5-flash-lite
```

## Install

You need Python 3.12 or newer.

```bash
uvx tooltangle --help     # run it without installing anything
pip install tooltangle    # or install it into a project
```

Set `GOOGLE_API_KEY` in your environment or in a `.env` file in the directory you run from.
Gemini is the default and the only provider tested so far. Other providers go through
LangChain (`pip install langchain-openai`, then `--model openai:<model name>`) and should
work, but treat that as untried.

## Usage

```bash
tooltangle tools claude-desktop   # list your tools and run the checks that need no model
tooltangle check claude-desktop   # measure which tools the model confuses
tooltangle fix claude-desktop     # rewrite confusing descriptions, keep the ones that hold up
tooltangle check claude-desktop   # measure again with the rewrites applied
```

The argument says where the tools come from: `claude-desktop`, `claude-code`, `cursor` or
`vscode`, a path to an MCP config file, or a Python target such as `my_agent/tools.py:TOOLS`
with LangChain tools or plain functions. MCP servers are started only to list their tools.
tooltangle never calls a tool.

The repository has two examples to try first:

```bash
tooltangle tools examples/demo.mcp.json            # five reference MCP servers, 38 tools
tooltangle check examples/support_tools.py:TOOLS   # six LangChain tools of a support agent
```

To pick a model, `compare` runs the same messages through several and names the cheapest one
that a paired test can't show to be worse than the best:

```bash
tooltangle compare mcp.json -m google_genai:gemini-3.8-flash -m google_genai:gemini-3.5-flash-lite
```

## How it works

1. It loads the tools from all servers and runs the checks that need no model: duplicate
   names, missing or very short descriptions, deprecated tools, the token cost of the
   definitions.
2. A generator model writes messages a real user might send: some for each tool, tricky ones
   for pairs that are easy to mix up, and some that need no tool. Messages that name the tool
   or repeat its description are dropped. A second pass labels every message with all the
   tools that would be a reasonable first call, so a message with two valid answers never
   counts as a mistake.
3. The messages are split into dev and holdout halves and saved to
   `.tooltangle/queries.jsonl`, which you can read, edit and commit.
4. The model under test gets each message with all the tools attached, and the first tool it
   calls is recorded.
5. The report lists directed confusions with a rate and an example, needless tool calls, and
   accuracy with a 95% Wilson interval, in the terminal and in `.tooltangle/report.html`.

Looking before acting is not a confusion. Calling a read-only tool from the same server first,
such as `git_status` before `git_commit`, is reported as INFO. This uses the MCP
`readOnlyHint` annotation.

## How `fix` decides what to keep

`fix` asks the generator to rewrite the descriptions of the most confused pairs, then tests
each rewrite:

- The generator sees mistakes only from the dev half. The rewrite is scored on the holdout
  half.
- It is kept only if a one-sided exact McNemar test gives p < 0.05 and, among related tools,
  it broke at most one more message than it fixed.
- The 0.05 is split across attempts, and a retry is told only the counts, never the messages.
- If a pair has too few mistakes to prove anything (fewer than six on holdout by default), it
  first writes more messages for that pair, and says so if that still isn't enough.

Accepted rewrites go to `tooltangle.overrides.yaml`, and the next `check` applies them. To use
them in your own agent:

```python
from tooltangle import apply_overrides

tools = apply_overrides(tools)  # LangChain tools; reads tooltangle.overrides.yaml
```

Apps like Claude Desktop or Cursor take descriptions straight from the server, so there a
rewrite is a suggestion to pass on to the server's author.

## CI

`tooltangle check mcp.json --yes` works as a CI step. Its exit codes:

| Code | Meaning |
|---|---|
| 0 | nothing at ERROR level |
| 1 | a tool loses at least 20% of its messages (3 or more) to one wrong choice, two tools share a name, or a tool has no description |
| 2 | the run failed: a bad config, a model that can't be created, or every request failing |
| 3 | the daily quota ran out |

Commit `.tooltangle/queries.jsonl` so every run scores the same messages, and cache
`.tooltangle/cache.sqlite` so a run with no changes makes no model calls. See
[examples/github-workflow.yml](https://github.com/bebz0/tooltangle/blob/main/examples/github-workflow.yml).

## Configuration

Settings are optional and go in `tooltangle.toml` in the directory you run from. The ones you
are most likely to change, with their defaults:

```toml
model = "google_genai:gemini-3.5-flash-lite"   # the model under test
generator = "google_genai:gemini-3.8-flash"    # writes and labels the test messages
language = "English"                           # language of the test messages
queries_per_tool = 10
requests_per_minute = 120
```

There are also `generator_fallbacks`, `contrast_pairs`, `queries_per_contrast`, `concurrency`,
`system_prompt`, `model_options` and a `[prices]` table; see
[settings.py](https://github.com/bebz0/tooltangle/blob/main/src/tooltangle/settings.py).

## Cost

- Every model answer is cached, so running again without changes is free and an interrupted
  run picks up where it stopped.
- Before making more than 25 model calls it prints an estimate and asks. `--yes` skips the
  question.
- Requests are paced and retried on rate limits. When the generator's quota runs out, the next
  model in `generator_fallbacks` takes over.


## License

[MIT](https://github.com/bebz0/tooltangle/blob/main/LICENSE)
