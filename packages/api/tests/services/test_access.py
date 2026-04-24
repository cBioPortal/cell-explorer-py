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
