from langchain_core.tools import tool


@tool
def find_order(order_id: str) -> str:
    """Look up an order and return its items, total and current status."""
    return ""


@tool
def track_shipment(order_id: str) -> str:
    """Get the delivery status of an order."""
    return ""


@tool
def cancel_order(order_id: str, reason: str) -> str:
    """Cancel an order."""
    return ""


@tool
def refund_order(order_id: str, amount: float) -> str:
    """Refund money for an order."""
    return ""


@tool
def search_help_center(query: str) -> str:
    """Search help center articles about shipping, returns and payments."""
    return ""


@tool
def create_ticket(subject: str, body: str) -> str:
    """Open a ticket for the support team when the customer needs a human."""
    return ""


TOOLS = [
    find_order,
    track_shipment,
    cancel_order,
    refund_order,
    search_help_center,
    create_ticket,
]
