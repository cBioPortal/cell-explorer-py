"""CLI-local exceptions (the in-process ChatSessionError hierarchy lives in services/)."""


class CliError(Exception):
    """Base class for CLI-local errors."""


class NotLoggedInError(CliError):
    """Raised when no auth.json exists and a command needs authentication."""


class SessionExpiredError(CliError):
    """Raised when both access and refresh tokens are unusable."""
