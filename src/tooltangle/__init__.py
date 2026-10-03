from importlib.metadata import version

from tooltangle.overrides import apply_overrides

__version__ = version("tooltangle")
__all__ = ["__version__", "apply_overrides"]
