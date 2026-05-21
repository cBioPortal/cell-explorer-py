"""In-memory test fake for the Langfuse v3 client surface.

Captures the same calls the real v3 SDK exposes:
  - start_observation / start_as_current_observation
  - update_current_trace
  - observation.update(...) / observation.end()
  - flush()

The TurnTrace context manager and the agent's tests duck-type against
these methods, so the fake is sufficient for every unit test except the
end-to-end integration test in test_telemetry_integration.py (which
exercises the real SDK against a stub HTTP server).
"""

from __future__ import annotations

from typing import Any


class FakeObservation:
    """Mirrors the v3 SDK's observation object — span, generation, etc."""

    def __init__(
        self,
        *,
        client: "FakeLangfuseClient",
        name: str | None = None,
        as_type: str = "span",
        input: Any = None,
        metadata: dict[str, Any] | None = None,
        model: str | None = None,
        version: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        parent: "FakeObservation | None" = None,
    ) -> None:
        self._client = client
        self.name = name
        self.as_type = as_type
        self.input = input
        self.metadata = metadata
        self.model = model
        self.version = version
        self.model_parameters = model_parameters
        self.parent = parent
        # Set by .update():
        self.output: Any = None
        self.level: str = "DEFAULT"
        self.status_message: str | None = None
        self.usage_details: dict[str, int] | None = None
        self.cost_details: dict[str, float] | None = None
        # Lifecycle:
        self.ended: bool = False
        self.scores: list[dict[str, Any]] = []
        # Replay log for tests that want to verify call order.
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        # Record each call for tests that want call-order detail.
        self.updates.append(dict(kwargs))
        # Apply attributes.
        for k, v in kwargs.items():
            if k == "metadata":
                # Merge metadata rather than replace, mirroring v3 semantics.
                existing = self.metadata or {}
                self.metadata = {**existing, **(v or {})}
            else:
                setattr(self, k, v)

    def end(self, **kwargs: Any) -> None:
        if kwargs:
            self.update(**kwargs)
        self.ended = True

    def score(self, **kwargs: Any) -> None:
        self.scores.append(dict(kwargs))

    # Support `with start_as_current_observation(...) as obs:` ergonomics.
    def __enter__(self) -> "FakeObservation":
        self._client._push_current(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._client._pop_current(self)
        self.end()
        return None


class FakeLangfuseClient:
    """In-memory Langfuse v3 client double. Records every call."""

    def __init__(self) -> None:
        self.observations: list[FakeObservation] = []
        self.trace_updates: list[dict[str, Any]] = []
        self.flushed_count: int = 0
        self._current_stack: list[FakeObservation] = []
        # Tests can override; defaults to a synthetic value returned by
        # get_current_trace_id() while a span context is active.
        self.fake_trace_id: str = "fake-trace-id"
        # Records every create_score(**kwargs) call for assertion in tests.
        self.scores: list[dict[str, Any]] = []

    # ---- observation creation ----

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        """Create an observation. Auto-attaches as child of current, if any.
        Does NOT push onto the current stack — caller is responsible for
        calling .end() when done.
        """
        parent = self._current_stack[-1] if self._current_stack else None
        obs = FakeObservation(client=self, parent=parent, **kwargs)
        self.observations.append(obs)
        return obs

    def start_as_current_observation(self, **kwargs: Any) -> FakeObservation:
        """Create an observation AND return a context manager that pushes
        it as the current observation for the duration of the `with` block.
        On __exit__, pops and ends it.
        """
        parent = self._current_stack[-1] if self._current_stack else None
        obs = FakeObservation(client=self, parent=parent, **kwargs)
        self.observations.append(obs)
        return obs  # FakeObservation supports __enter__/__exit__

    # ---- trace-level updates ----

    def update_current_trace(self, **kwargs: Any) -> None:
        """Record a trace-level update. Tests inspect self.trace_updates."""
        self.trace_updates.append(dict(kwargs))

    def update_current_generation(self, **kwargs: Any) -> None:
        # Apply to the deepest generation on the current stack, if any.
        for obs in reversed(self._current_stack):
            if obs.as_type == "generation":
                obs.update(**kwargs)
                return

    def update_current_span(self, **kwargs: Any) -> None:
        for obs in reversed(self._current_stack):
            if obs.as_type == "span":
                obs.update(**kwargs)
                return

    # ---- trace id resolution ----

    def get_current_trace_id(self) -> str | None:
        """Mirrors v3 SDK: returns the active trace id while a span context
        is open; None otherwise."""
        return self.fake_trace_id if self._current_stack else None

    # ---- scores ----

    def create_score(self, **kwargs: Any) -> None:
        """Record a create_score call. Tests inspect self.scores."""
        self.scores.append(dict(kwargs))

    # ---- lifecycle ----

    def flush(self) -> None:
        self.flushed_count += 1

    # ---- helpers for FakeObservation.__enter__/__exit__ ----

    def _push_current(self, obs: FakeObservation) -> None:
        self._current_stack.append(obs)

    def _pop_current(self, obs: FakeObservation) -> None:
        # Pop only if it's the top; defensive against mismatched lifecycle.
        if self._current_stack and self._current_stack[-1] is obs:
            self._current_stack.pop()

    # ---- query helpers for tests ----

    @property
    def root_observation(self) -> FakeObservation | None:
        """The first observation created (the 'chat-turn' root in our usage)."""
        return self.observations[0] if self.observations else None

    @property
    def generations(self) -> list[FakeObservation]:
        return [o for o in self.observations if o.as_type == "generation"]

    @property
    def tool_spans(self) -> list[FakeObservation]:
        return [
            o for o in self.observations
            if o.as_type == "span" and (o.name or "").startswith("tool:")
        ]

    def trace_field(self, key: str) -> Any:
        """Return the most recent value set for a trace-level field."""
        for upd in reversed(self.trace_updates):
            if key in upd:
                return upd[key]
        return None
