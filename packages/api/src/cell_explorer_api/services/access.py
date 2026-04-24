"""Shared access-check helper used by routes and the CLI."""

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
