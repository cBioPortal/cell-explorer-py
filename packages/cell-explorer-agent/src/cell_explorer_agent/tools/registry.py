"""Tool dataclass and catalog."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

ToolKind = Literal["data", "ui_action"]
ToolFunc = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class Tool:
    name: str
    kind: ToolKind
    description: str
    args_schema: dict[str, Any]  # JSON Schema for args
    func: ToolFunc
    # If True, the agent passes `_context={"view_state": ...}` as an extra
    # kwarg when invoking `func`. The LLM-visible `args_schema` is unaffected;
    # `_context` is a private channel so tools can validate against the
    # current UI snapshot without making the LLM aware of it.
    wants_context: bool = False


@dataclass
class ToolCatalog:
    _by_name: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self._by_name:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._by_name[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name)

    def all(self) -> list[Tool]:
        return list(self._by_name.values())

    def by_kind(self, kind: ToolKind) -> list[Tool]:
        return [t for t in self._by_name.values() if t.kind == kind]
