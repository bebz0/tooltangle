PLAIN_QUERIES = """\
You are building a test set for an AI assistant that can call tools.
These are all the tools the assistant can see:

{catalog}

For each of these tools: {functions}
write {count} different messages a real user could send where the best first step \
for the assistant is to call that tool.

- Sound like real people: mix short and long messages, casual and precise ones, \
and messages that explain why the user needs it.
- Never mention any tool by name, and avoid reusing distinctive phrases from the tool \
descriptions.
- Include concrete details when natural: file names, paths, dates, URLs, names.
- The tool a message is written for must fit it better than any other tool in the list.
- Every message must differ from the others in intent or wording.
- Write the messages in {language}."""

NO_TOOL_QUERIES = """\
These are all the tools an AI assistant can see:

{catalog}

Write {count} messages a real user could send where the assistant should answer \
directly without calling any tool. Mix three kinds:
- questions on the same topics as the tools that only need knowledge or an \
explanation, for example how something works;
- writing, reasoning or small-talk requests;
- requests that sound related but that none of these tools can actually do.

Write the messages in {language}."""

LABELS = """\
These are all the tools an AI assistant can see:

{catalog}

For each numbered message below, list every tool that would be a reasonable FIRST call \
for handling it, using the tool names exactly as written above. Don't include tools that \
are merely related. Return an empty list when the assistant should answer directly, or \
when no tool can help.

{messages}"""
