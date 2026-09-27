from tooltangle.findings import Severity, lookalike_pairs, static_findings
from tooltangle.toolset import Toolset, ToolSpec


def test_static_checks():
    toolset = Toolset(
        [
            ToolSpec("files", "search", "Search files on disk by glob pattern."),
            ToolSpec("web", "search", "Search the web and return the top results."),
            ToolSpec("files", "read_file", "Read a file. DEPRECATED: use read_text_file."),
            ToolSpec("files", "stat", "File info"),
            ToolSpec("files", "noop", ""),
        ]
    )
    assert sorted((finding.severity, finding.code) for finding in static_findings(toolset)) == [
        (Severity.ERROR, "duplicate-name"),
        (Severity.ERROR, "no-description"),
        (Severity.WARN, "deprecated"),
        (Severity.WARN, "short-description"),
    ]


def test_lookalike_pairs_rank_similar_tools_first():
    toolset = Toolset(
        [
            ToolSpec("fs", "list_directory", "List files and directories in a path."),
            ToolSpec("fs", "list_directory_with_sizes", "List files and directories with sizes."),
            ToolSpec("time", "get_current_time", "Get the current time in a timezone."),
        ]
    )
    pairs = lookalike_pairs(toolset, threshold=0.2)
    assert pairs[0][:2] == ("fs.list_directory", "fs.list_directory_with_sizes")
    assert all("time.get_current_time" not in pair[:2] for pair in pairs)
