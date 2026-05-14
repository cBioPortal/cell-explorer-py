"""Tests for /api/chat/{slug}/messages/{id}/feedback endpoints."""

import asyncio

from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from .conftest import _set_auth_cookie


def _create_thread_with_assistant_msg(seeded_app, *, user_sub: str) -> str:
    """Seed one ChatThread + one assistant ChatMessageRow; return the message id (str)."""
    from cell_explorer_api.db.models import ChatMessageRow, ChatThread, Dataset
    from sqlmodel import select

    async def _seed() -> str:
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
            await session.refresh(m)
            return str(m.id)

    return asyncio.run(_seed())


def test_put_feedback_up_creates_row(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    msg_id = _create_thread_with_assistant_msg(seeded_app, user_sub="user-1")

    res = client.put(
        f"/api/chat/public-atlas/messages/{msg_id}/feedback",
        json={"rating": "up"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["rating"] == "up"
    assert body["comment"] is None
