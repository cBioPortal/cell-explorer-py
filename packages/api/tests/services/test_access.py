from dataclasses import dataclass

from cell_explorer_api.services.access import user_can_access


@dataclass
class FakeUser:
    roles: list[str]


@dataclass
class FakeDataset:
    is_public: bool
    required_roles: list[str]


def test_public_dataset_is_accessible_to_anonymous():
    assert user_can_access(FakeDataset(is_public=True, required_roles=[]), user=None) is True


def test_public_dataset_is_accessible_to_any_user():
    assert user_can_access(FakeDataset(is_public=True, required_roles=[]),
                           user=FakeUser(roles=[])) is True


def test_private_dataset_rejects_anonymous():
    assert user_can_access(FakeDataset(is_public=False, required_roles=["researcher"]),
                           user=None) is False


def test_private_dataset_requires_role_overlap():
    ds = FakeDataset(is_public=False, required_roles=["researcher", "admin"])
    assert user_can_access(ds, user=FakeUser(roles=["researcher"])) is True
    assert user_can_access(ds, user=FakeUser(roles=["other"])) is False


def test_private_dataset_without_required_roles_rejects_everyone():
    ds = FakeDataset(is_public=False, required_roles=[])
    assert user_can_access(ds, user=FakeUser(roles=["admin"])) is False


from cell_explorer_api.services.access import (
    ChatPermission,
    compute_chat_permission,
)


def test_chat_permission_anonymous_always_fails_when_no_role():
    perm = compute_chat_permission(None, required_role=None)
    assert perm == ChatPermission(can_chat=False, reason="requires_auth")


def test_chat_permission_anonymous_always_fails_when_role_required():
    perm = compute_chat_permission(None, required_role="cell-explorer-chat")
    assert perm == ChatPermission(can_chat=False, reason="requires_auth")


def test_chat_permission_authed_user_passes_when_no_role_required():
    perm = compute_chat_permission(FakeUser(roles=[]), required_role=None)
    assert perm == ChatPermission(can_chat=True, reason=None)


def test_chat_permission_authed_user_passes_when_role_held():
    perm = compute_chat_permission(
        FakeUser(roles=["cell-explorer-chat", "other"]),
        required_role="cell-explorer-chat",
    )
    assert perm == ChatPermission(can_chat=True, reason=None)


def test_chat_permission_authed_user_fails_when_role_missing():
    perm = compute_chat_permission(
        FakeUser(roles=["other"]),
        required_role="cell-explorer-chat",
    )
    assert perm == ChatPermission(
        can_chat=False, reason="missing_role:cell-explorer-chat"
    )
