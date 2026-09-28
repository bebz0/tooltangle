# tooltangle

Find which tools your LLM agent confuses, and fix their descriptions with proof.

An agent with five MCP servers sees forty tools at once, and the model sometimes calls the
wrong one. Server authors test their own tools, not the mix you actually run. tooltangle will
write realistic test messages for every tool, run your model on them with all of your tools
attached, and show exactly which tools it mixes up.

Work in progress.

## Install

```bash
git clone <repo-url>
cd tooltangle
uv sync
```

## Usage

Load the tools and run the checks that don't need a model:

```bash
uv run tooltangle tools claude-desktop            # or claude-code, cursor, vscode
uv run tooltangle tools path/to/mcp.json --list   # --list prints every tool
uv run tooltangle tools my_agent/tools.py:TOOLS   # LangChain tools from a Python module
```

```
servers   time 2 · filesystem 14
tools     16    definitions ≈ 1,971 tokens per request

WARN   filesystem.read_file is marked deprecated but is still exposed

look-alike descriptions (text similarity only)
  0.83  filesystem.list_directory ~ filesystem.list_directory_with_sizes
  0.45  filesystem.read_file ~ filesystem.read_text_file
```

## Models

Only Gemini models are supported for now. Set `GOOGLE_API_KEY` or `GEMINI_API_KEY`.
Other providers are planned.
