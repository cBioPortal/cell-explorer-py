"""Tests for /api/chat/{slug}/messages/{id}/feedback endpoints."""

import asyncio

from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from .conftest import _set_auth_cookie


def _create_thread_with_assistant_msg(seeded_app, *, user_sub: str) -> tuple[str, str]:
    """Seed one ChatThread + one assistant ChatMessageRow.

    Returns ``(thread_id, message_id)`` as strings.
    """
    from cell_explorer_api.db.models import ChatMessageRow, ChatThread, Dataset
    from sqlmodel import select

    async def _seed() -> tuple[str, str]:
        engine = seeded_app.state.db_engine
        async with AsyncSession(engine) as session:
            dataset = (await session.exec(select(Dataset))).first()
            t = ChatThread(
                user_sub=user_sub,
                dataset_id=dataset.id,
                title="test",
            )
            session.add(t)
            await session.flush()
            m = ChatMessageRow(thread_id=t.id, role="assistant", content="hello")
            session.add(m)
            await session.commit()
            await session.refresh(t)
            await session.refresh(m)
            return str(t.id), str(m.id)

    return asyncio.run(_seed())


def test_put_feedback_up_creates_row(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    _thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")

    res = client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "up"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["rating"] == "up"
    assert body["comment"] is None


def test_put_feedback_with_comment_down(seeded_app):
    """PUT with a comment + rating=down stores both."""
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    _thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")

    res = client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "down", "comment": "off-topic"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["rating"] == "down"
    assert body["comment"] == "off-topic"


def test_put_feedback_upserts_in_place(seeded_app):
    """Second PUT updates the existing row rather than creating a second."""
    from cell_explorer_api.db.models import ChatFeedback
    from sqlmodel import select

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    _thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")

    first = client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "up"},
    )
    assert first.status_code == 200, first.text

    second = client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "down", "comment": "changed my mind"},
    )
    assert second.status_code == 200, second.text
    assert second.json()["rating"] == "down"
    assert second.json()["comment"] == "changed my mind"

    async def _count() -> int:
        engine = seeded_app.state.db_engine
        async with AsyncSession(engine) as session:
            rows = (await session.exec(select(ChatFeedback))).all()
            return len(rows)

    assert asyncio.run(_count()) == 1


def test_put_feedback_rejects_other_users_message(seeded_app):
    """A message owned by another user 404s (existence-hiding)."""
    from cell_explorer_api.db.models import ChatMessageRow, ChatThread, Dataset
    from sqlmodel import select

    async def _seed_for_other_user() -> str:
        engine = seeded_app.state.db_engine
        async with AsyncSession(engine) as session:
            dataset = (await session.exec(select(Dataset))).first()
            t = ChatThread(
                user_sub="someone-else",
                dataset_id=dataset.id,
                title="not yours",
            )
            session.add(t)
            await session.flush()
            m = ChatMessageRow(thread_id=t.id, role="assistant", content="x")
            session.add(m)
            await session.commit()
            await session.refresh(m)
            return str(m.id)

    other_msg_id = asyncio.run(_seed_for_other_user())

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    res = client.put(
        f"/api/chat/public-atlas/messages/{other_msg_id}/feedback",
        json={"rating": "up"},
    )
    assert res.status_code == 404, res.text


def test_put_feedback_rejects_user_role_message(seeded_app):
    """Only assistant messages can be rated; user-role messages return 422."""
    from cell_explorer_api.db.models import ChatMessageRow, ChatThread, Dataset
    from sqlmodel import select

    async def _seed_user_msg() -> str:
        engine = seeded_app.state.db_engine
        async with AsyncSession(engine) as session:
            dataset = (await session.exec(select(Dataset))).first()
            t = ChatThread(
                user_sub="user-1",
                dataset_id=dataset.id,
                title="t",
            )
            session.add(t)
            await session.flush()
            m = ChatMessageRow(thread_id=t.id, role="user", content="hi")
            session.add(m)
            await session.commit()
            await session.refresh(m)
            return str(m.id)

    user_msg_id = asyncio.run(_seed_user_msg())

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    res = client.put(
        f"/api/chat/public-atlas/messages/{user_msg_id}/feedback",
        json={"rating": "up"},
    )
    assert res.status_code == 422, res.text


def test_put_feedback_rejects_invalid_rating(seeded_app):
    """Pydantic rejects ratings outside the Literal['up', 'down']."""
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    _thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")

    res = client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "meh"},
    )
    assert res.status_code == 422, res.text


def test_put_feedback_requires_auth(seeded_app):
    """Without the cce_access cookie, the route returns 401."""
    client = TestClient(seeded_app)
    # Note: no _set_auth_cookie call.
    _thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")

    res = client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "up"},
    )
    assert res.status_code == 401, res.text


def test_delete_feedback_removes_row(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    _thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")
    client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "up"},
    )
    res = client.delete(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
    )
    assert res.status_code == 204


def test_delete_feedback_404_when_none(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    _thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")
    res = client.delete(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
    )
    assert res.status_code == 404


# ---------- GET /threads/{id} embedded feedback tests ----------


def test_thread_detail_includes_callers_feedback(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")
    client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "up"},
    )
    res = client.get(f"/api/chat/public-atlas/threads/{thread_id}")
    assert res.status_code == 200
    msgs = res.json()["messages"]
    assert len(msgs) == 1
    assert msgs[0]["feedback"]["rating"] == "up"


def test_thread_detail_no_feedback_field_is_null(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    thread_id, _msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")
    res = client.get(f"/api/chat/public-atlas/threads/{thread_id}")
    assert res.status_code == 200
    assert res.json()["messages"][0]["feedback"] is None


def test_thread_detail_isolates_feedback_per_user(seeded_app):
    """Feedback from another user must not leak into the owner's thread response."""
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")
    # Owner rates "up".
    client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "up"},
    )
    # Insert another user's feedback row directly into the DB.
    import uuid as _u

    from cell_explorer_api.db.models import ChatFeedback

    async def _insert():
        async with AsyncSession(seeded_app.state.db_engine) as s:
            s.add(ChatFeedback(
                message_id=_u.UUID(msg_id),
                user_sub="other-user-sub",
                rating="down",
                comment="leaked?",
            ))
            await s.commit()

    asyncio.run(_insert())
    # Owner's view must still show only their own "up".
    res = client.get(f"/api/chat/public-atlas/threads/{thread_id}")
    assert res.status_code == 200
    fb = res.json()["messages"][0]["feedback"]
    assert fb["rating"] == "up"
    assert fb.get("comment") in (None, "")


def test_thread_detail_includes_message_id(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    thread_id, msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")
    res = client.get(f"/api/chat/public-atlas/threads/{thread_id}")
    assert res.status_code == 200
    assert res.json()["messages"][0]["id"] == msg_id
