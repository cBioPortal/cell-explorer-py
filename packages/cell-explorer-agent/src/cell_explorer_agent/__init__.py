"""LLM chat agent and CLI for single-cell zarr datasets."""

from cell_explorer_agent.agent import ChatAgent
from cell_explorer_agent.config import AgentConfig
from cell_explorer_agent.events import (
    ChatEvent,
    Done,
    Error,
    TextDelta,
    TokenUsage,
    ToolProgress,
    UIAction,
    dump_event,
)
from cell_explorer_agent.llm.anthropic import AnthropicLLMClient
from cell_explorer_agent.llm.base import LLMClient
from cell_explorer_agent.llm.fake import FakeLLMClient, Script
from cell_explorer_agent.messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from cell_explorer_agent.prompt.dataset_context import (
    DatasetContext,
    build_dataset_context,
)
from cell_explorer_agent.tools.registry import Tool, ToolCatalog, ToolKind
from cell_explorer_agent.tools.zarr_protocol import (
    ObsColumn,
    ObsColumnSpec,
    ZarrAccess,
)

__version__ = "0.1.0"

__all__ = [
    # core
    "ChatAgent",
    "AgentConfig",
    # events
    "ChatEvent",
    "TextDelta",
    "ToolProgress",
    "UIAction",
    "Error",
    "Done",
    "TokenUsage",
    "dump_event",
    # messages
    "Message",
    "UserMessage",
    "AssistantMessage",
    "ToolCall",
    "ToolResult",
    "ToolResultMessage",
    # llm
    "LLMClient",
    "AnthropicLLMClient",
    "FakeLLMClient",
    "Script",
    # prompt
    "DatasetContext",
    "build_dataset_context",
    # tools
    "Tool",
    "ToolCatalog",
    "ToolKind",
    "ZarrAccess",
    "ObsColumn",
    "ObsColumnSpec",
]
