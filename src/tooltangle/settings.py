import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they help with the user's request."
)


class SettingsError(Exception):
    pass


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "google_genai:gemini-3.5-flash-lite"
    generator: str = "google_genai:gemini-3.8-flash"
    model_options: dict[str, Any] = Field(default_factory=dict)
    language: str = "English"
    queries_per_tool: int = 10
    contrast_pairs: int = 8
    queries_per_contrast: int = 6
    concurrency: int = 8
    requests_per_minute: int = 120
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    state_dir: Path = Path(".tooltangle")
    prices: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @property
    def cache_path(self) -> Path:
        return self.state_dir / "cache.sqlite"


def load_settings(path: Path = Path("tooltangle.toml"), **overrides: Any) -> Settings:
    data = {}
    if path.is_file():
        try:
            data = tomllib.loads(path.read_text())
        except tomllib.TOMLDecodeError as error:
            raise SettingsError(f"{path}: {error}") from error
    data.update({key: value for key, value in overrides.items() if value is not None})
    try:
        return Settings(**data)
    except ValidationError as error:
        raise SettingsError(f"{path}: {error}") from error
