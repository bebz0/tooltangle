from tooltangle.findings import Severity, static_findings
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
