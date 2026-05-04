"""Package-specific exceptions."""


class AgentError(Exception):
    """Base class for chat-agent errors."""


class ToolCallCapExceeded(AgentError):
    """Agent exceeded max_tool_calls_per_turn."""
