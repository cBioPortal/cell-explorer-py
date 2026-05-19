"""Observability hooks for the chat agent.

This package is the ONLY place in the agent that knows about Langfuse.
Everything else interacts via `TurnTrace` from `trace_context`.
"""
