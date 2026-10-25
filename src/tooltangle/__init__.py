from importlib.metadata import version
from typing import Any

__version__ = version("tooltangle")
__all__ = ["__version__", "apply_overrides"]


def __getattr__(name: str) -> Any:
    if name == "apply_overrides":
        from tooltangle.overrides import apply_overrides

        return apply_overrides
    raise AttributeError(f"module 'tooltangle' has no attribute {name!r}")
