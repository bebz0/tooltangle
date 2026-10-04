from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

server = FastMCP("notes")
reading = ToolAnnotations(readOnlyHint=True)


@server.tool(annotations=reading)
def search_notes(query: str) -> str:
    """Search the user's notes by keyword."""
    return ""


@server.tool(annotations=reading)
def read_note(title: str) -> str:
    """Read one note by its title."""
    return ""


@server.tool(annotations=ToolAnnotations(readOnlyHint=False))
def create_note(title: str, body: str) -> str:
    """Create a new note."""
    return ""


@server.tool()
def search_files(pattern: str) -> str:
    """Search files on disk by glob pattern."""
    return ""


if __name__ == "__main__":
    server.run()
