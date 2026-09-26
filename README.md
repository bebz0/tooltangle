# tooltangle

Find which tools your LLM agent confuses, and fix their descriptions with proof.

An agent with five MCP servers sees forty tools at once, and the model sometimes calls the
wrong one. Server authors test their own tools, not the mix you actually run. tooltangle will
write realistic test messages for every tool, run your model on them with all of your tools
attached, and show exactly which tools it mixes up.

Work in progress.
