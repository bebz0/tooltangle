from mcp.server.fastmcp import FastMCP

server = FastMCP("notes")

@server.tool()
def search_notes(query: str) -> str:
    """Search the user's notes by keyword."""
    return ""

@server.tool()
def read_note(title: str) -> str:
    """Read one note by its title."""
    return ""

@server.tool()
def create_note(title: str, body: str) -> str:
    """Create a new note."""
    return ""

@server.tool()
def search_files(pattern: str) -> str:
    """Search files on disk by glob pattern."""
    return ""


if __name__ == "__main__":
    server.run()
