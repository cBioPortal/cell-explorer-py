"""In-memory test fake for the Langfuse client.

Captures the exact same calls the real SDK exposes (trace, generation,
span, update, flush) into Python data structures so tests can assert
on call ordering and payloads without network or SDK dependencies.

The TurnTrace context manager uses duck typing on these methods, so
the fake is sufficient for unit tests at every layer except the
end-to-end integration test in test_telemetry_integration.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeGeneration:
    name: str | None = None
    model: str | None = None
    input: Any = None
    output: Any = None
    usage_details: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None
    updates: list[dict[str, Any]] = field(default_factory=list)

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self, **kwargs: Any) -> None:
        if kwargs:
            self.updates.append({"_end": True, **kwargs})


@dataclass
class FakeSpan:
    name: str | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] | None = None
    level: str = "DEFAULT"
    status_message: str | None = None
    updates: list[dict[str, Any]] = field(default_factory=list)

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self, **kwargs: Any) -> None:
        if kwargs:
            self.updates.append({"_end": True, **kwargs})


@dataclass
class FakeTrace:
    name: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] | None = None
    input: Any = None
    output: Any = None
    generations: list[FakeGeneration] = field(default_factory=list)
    spans: list[FakeSpan] = field(default_factory=list)
    updates: list[dict[str, Any]] = field(default_factory=list)

    def generation(self, **kwargs: Any) -> FakeGeneration:
        gen = FakeGeneration(**kwargs)
        self.generations.append(gen)
        return gen

    def span(self, **kwargs: Any) -> FakeSpan:
        sp = FakeSpan(**kwargs)
        self.spans.append(sp)
        return sp

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeLangfuseClient:
    """In-memory Langfuse client double. Records every call."""

    def __init__(self) -> None:
        self.traces: list[FakeTrace] = []
        self.flushed_count: int = 0

    def trace(self, **kwargs: Any) -> FakeTrace:
        t = FakeTrace(**kwargs)
        self.traces.append(t)
        return t

    def flush(self) -> None:
        self.flushed_count += 1
