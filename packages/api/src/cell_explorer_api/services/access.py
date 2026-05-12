"""Shared access-check helper used by routes and the CLI."""

from dataclasses import dataclass
from typing import Protocol


class _DatasetLike(Protocol):
    is_public: bool
    required_roles: list[str]


class _UserLike(Protocol):
    roles: list[str]


def user_can_access(dataset: _DatasetLike, *, user: _UserLike | None) -> bool:
    """Return True iff the user can access the given dataset.

    - Public datasets are accessible to everyone (including anonymous).
    - Private datasets require at least one role overlap with required_roles.
    """
    if dataset.is_public:
        return True
    if user is None:
        return False
    if not dataset.required_roles:
        return False
    return bool(set(user.roles) & set(dataset.required_roles))


@dataclass(frozen=True)
class ChatPermission:
    """Result of the global chat-role gate (Layer 2 in the gating model).

    reason values:
      - None        — can_chat=True
      - "requires_auth" — anonymous user, must sign in
      - "missing_role:<name>" — authed user but lacks the required role
    """

    can_chat: bool
    reason: str | None


def compute_chat_permission(
    user: _UserLike | None, *, required_role: str | None
) -> ChatPermission:
    """Whether `user` may use chat under the global role gate.

    Anonymous users always fail — chat requires authentication regardless
    of whether `required_role` is configured.
    """
    if user is None:
        return ChatPermission(can_chat=False, reason="requires_auth")
    if required_role is None:
        return ChatPermission(can_chat=True, reason=None)
    if required_role in user.roles:
        return ChatPermission(can_chat=True, reason=None)
    return ChatPermission(can_chat=False, reason=f"missing_role:{required_role}")
