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

CONTRAST_QUERIES = """\
You are building a test set for an AI assistant that can call tools.
These are all the tools the assistant can see:

{catalog}

Two of them are easy to mix up: `{first}` and `{second}`.

Write {count} messages where the best first step is `{first}`, and {count} messages \
where the best first step is `{second}`. Make them tricky: each message should touch \
the domain of the other tool, so that only a careful reading of both descriptions \
routes it correctly. On reflection the right tool must still be clearly better.

- Never mention any tool by name.
- Sound like real users and include concrete details when natural.
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

CONFUSABLE_PAIRS = """\
These are all the tools an AI assistant can see:

{catalog}

Which pairs of tools could a model confuse when deciding which one to call? Only list \
pairs whose purposes genuinely overlap or are easy to mix up from their descriptions, \
most confusable first. Return at most {limit} pairs and use the tool names exactly as \
written above."""

LABELS = """\
These are all the tools an AI assistant can see:

{catalog}

For each numbered message below, list every tool that would be a reasonable FIRST call \
for handling it, using the tool names exactly as written above. Don't include tools that \
are merely related. Return an empty list when the assistant should answer directly, or \
when no tool can help.

{messages}"""

FIX_DESCRIPTIONS = """\
You maintain the descriptions of tools that an AI assistant chooses between.
These are all the tools the assistant can see:

{catalog}

The assistant keeps confusing `{first}` and `{second}`. Messages where it picked the \
wrong one of the two:

{failures}

Messages it routed correctly:

{successes}

{feedback}
Rewrite the description of one or both of these two tools so the boundary between them \
is obvious. Rules:
- Keep every fact the current description states and don't invent capabilities.
- Say plainly when to use the tool and when to use the other one instead.
- At most three sentences per tool.
- Don't quote or paraphrase the example messages; describe the general rule."""

FIX_FEEDBACK = """
Your previous rewrite was tested on messages you haven't seen: it fixed {fixed} and \
broke {broken} of them for this pair, and broke {others_broken} messages for other tools. \
Try a different approach.
"""
