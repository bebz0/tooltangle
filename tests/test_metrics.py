import pytest

from tooltangle.dataset import Dataset, DatasetInfo, Query
from tooltangle.findings import Severity, evaluation_findings
from tooltangle.metrics import allow_looking_first, evaluate
from tooltangle.runner import Pick
from tooltangle.toolset import Toolset, ToolSpec

INFO = DatasetInfo(generators=["g"], language="English", labeled_tools=[])


def query(number: int, tool: str | None, accepted: list[str] | None = None) -> Query:
    accepted = accepted if accepted is not None else ([tool] if tool else [])
    return Query(
        id=f"q{number}",
        text=f"message {number}",
        tool=tool,
        accepted=accepted,
        kind="plain" if tool else "no_tool",
    )


def build(cases: list[tuple[str | None, str | None]], accepted=None):
    queries, picks = [], {}
    for number, (expected, picked) in enumerate(cases):
        queries.append(query(number, expected, (accepted or {}).get(number)))
        picks[f"q{number}"] = Pick(f"q{number}", picked, [picked] if picked else [])
    return queries, picks


def test_counts_confusions_in_both_directions():
    cases = [("a", "a")] * 6 + [("a", "b")] * 4 + [("b", "b")] * 9 + [("b", "a")]
    evaluation = evaluate("m", *build(cases))

    assert (evaluation.total, evaluation.correct) == (20, 15)
    assert evaluation.tools["a"].accuracy == pytest.approx(0.6)
    assert evaluation.tools["b"].precision == pytest.approx(9 / 13)
    top = evaluation.confusions[0]
    assert (top.expected, top.picked, top.count, top.total) == ("a", "b", 4, 10)
    assert top.rate == pytest.approx(0.4)


def test_accepted_alternatives_are_not_errors():
    queries, picks = build([("a", "b")], accepted={0: ["a", "b"]})
    assert evaluate("m", queries, picks).correct == 1


def test_findings_severity_follows_rate_and_count():
    cases = (
        [("a", "a")] * 7
        + [("a", "b")] * 3
        + [("b", "b")] * 18
        + [("b", "c")] * 2
        + [("c", "c")] * 20
        + [("c", "d")]
        + [(None, None)] * 8
        + [(None, "a")] * 2
    )
    findings = evaluation_findings(evaluate("m", *build(cases)))
    summary = [(finding.severity, finding.title) for finding in findings]
    assert summary == [
        (Severity.ERROR, "a → b"),
        (Severity.WARN, "b → c"),
        (Severity.WARN, "a tool was called for a message that needs none"),
    ]
    assert findings[0].detail == "30% (3 of 10)"
    assert findings[0].examples == ['"message 7"']


def test_looking_first_is_fine_but_reported():
    toolset = Toolset(
        [
            ToolSpec("git", "git_status", "Show the working tree status.", read_only=True),
            ToolSpec("git", "git_commit", "Record changes.", read_only=False),
            ToolSpec("git", "git_checkout", "Switch branches.", read_only=False),
            ToolSpec("time", "get_current_time", "Current time.", read_only=True),
        ]
    )
    queries = [query(number, "git.git_commit") for number in range(4)]
    queries.append(query(4, "git.git_status"))
    dataset = allow_looking_first(Dataset(info=INFO, queries=queries), toolset)

    assert dataset.queries[0].accepted == ["git.git_commit", "git.git_status"]
    assert dataset.queries[0].look_first == ["git.git_status"]
    assert dataset.queries[4].accepted == ["git.git_status"]

    picks = {
        "q0": Pick("q0", "git.git_status"),
        "q1": Pick("q1", "git.git_status"),
        "q2": Pick("q2", "git.git_checkout"),
        "q3": Pick("q3", "time.get_current_time"),
        "q4": Pick("q4", "git.git_commit"),
    }
    evaluation = evaluate("m", dataset.queries, picks)
    assert evaluation.correct == 2
    assert [(c.expected, c.picked, c.count) for c in evaluation.looked_first] == [
        ("git.git_commit", "git.git_status", 2)
    ]
    findings = evaluation_findings(evaluation)
    assert any(f.code == "looks-first" and f.severity == Severity.INFO for f in findings)
