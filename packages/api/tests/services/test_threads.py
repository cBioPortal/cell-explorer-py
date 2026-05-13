"""Tests for services/threads.py."""

from dataclasses import dataclass

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.db.models import (
    ChatMessageRow,
    ChatThread,
    Dataset,
    Datasource,
    DatasourceType,
)
from cell_explorer_api.services.threads import (
    ThreadAccessDeniedError,
    ThreadNotFoundError,
    append_message,
    create_thread,
    delete_thread,
    derive_title,
    list_threads,
    load_thread,
)


@dataclass
class FakeUser:
    sub: str
    roles: list[str]


@pytest.fixture()
async def session():
    """Async SQLite session with foreign_keys=ON.

    Uses the project's create_engine helper so the SQLite-FK pragma listener
    from db/__init__.py is attached. Without that, ON DELETE CASCADE tests
    silently no-op.
    """
    from cell_explorer_api.db import create_engine
    engine = create_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture()
async def seeded_dataset(session):
    ds = Datasource(
        name="Test", type=DatasourceType.S3_CLOUDFRONT, base_url="https://example.com"
    )
    session.add(ds)
    await session.flush()
    dataset = Dataset(
        datasource_id=ds.id,
        name="Demo",
        slug="demo",
        path="demo.zarr",
        is_public=True,
        chat_enabled=True,
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return dataset


# ---------- derive_title -------------------------------------------------

def test_derive_title_short_message_passes_through():
    assert derive_title("Hi") == "Hi"


def test_derive_title_truncates_to_50_chars_with_ellipsis():
    long_msg = "A" * 100
    title = derive_title(long_msg)
    assert title == "A" * 50 + "…"
    assert len(title) == 51


def test_derive_title_collapses_internal_whitespace():
    assert derive_title("hello   world\n\n  foo") == "hello world foo"


def test_derive_title_collapses_then_truncates():
    msg = "  " + "a " * 60 + "  "
    title = derive_title(msg)
    assert title.endswith("…")
    assert len(title) == 51
    assert "  " not in title


# ---------- create_thread ------------------------------------------------

async def test_create_thread_derives_title_from_first_message(session, seeded_dataset):
    user = FakeUser(sub="user-1", roles=[])
    thread = await create_thread(
        session, user=user, dataset=seeded_dataset,
        first_user_message="What are these clusters?",
    )
    assert thread.title == "What are these clusters?"
    assert thread.user_sub == "user-1"
    assert thread.dataset_id == seeded_dataset.id


# ---------- list_threads --------------------------------------------------

async def test_list_threads_returns_only_users_threads_for_dataset(session, seeded_dataset):
    user_a = FakeUser(sub="user-A", roles=[])
    user_b = FakeUser(sub="user-B", roles=[])
    await create_thread(session, user=user_a, dataset=seeded_dataset,
                        first_user_message="A's thread 1")
    await create_thread(session, user=user_a, dataset=seeded_dataset,
                        first_user_message="A's thread 2")
    await create_thread(session, user=user_b, dataset=seeded_dataset,
                        first_user_message="B's thread")
    await session.commit()

    summaries_a = await list_threads(session, user=user_a, dataset=seeded_dataset)
    summaries_b = await list_threads(session, user=user_b, dataset=seeded_dataset)
    assert len(summaries_a) == 2
    assert len(summaries_b) == 1
    assert all(s.title.startswith("A's") for s in summaries_a)


async def test_list_threads_orders_by_updated_at_desc(session, seeded_dataset):
    user = FakeUser(sub="u", roles=[])
    t1 = await create_thread(session, user=user, dataset=seeded_dataset,
                             first_user_message="first")
    t2 = await create_thread(session, user=user, dataset=seeded_dataset,
                             first_user_message="second")
    await session.commit()
    await append_message(session, t1, role="assistant", content="reply")
    await session.commit()

    summaries = await list_threads(session, user=user, dataset=seeded_dataset)
    assert [s.title for s in summaries] == ["first", "second"]


async def test_list_threads_includes_message_count(session, seeded_dataset):
    user = FakeUser(sub="u", roles=[])
    t = await create_thread(session, user=user, dataset=seeded_dataset,
                            first_user_message="hi")
    await append_message(session, t, role="user", content="hi")
    await append_message(session, t, role="assistant", content="hello")
    await session.commit()
    summaries = await list_threads(session, user=user, dataset=seeded_dataset)
    assert summaries[0].message_count == 2


# ---------- load_thread ---------------------------------------------------

async def test_load_thread_returns_thread_for_owner(session, seeded_dataset):
    user = FakeUser(sub="u", roles=[])
    t = await create_thread(session, user=user, dataset=seeded_dataset,
                            first_user_message="hi")
    await session.commit()
    loaded = await load_thread(session, user=user, thread_id=t.id)
    assert loaded.id == t.id


async def test_load_thread_missing_id_raises(session, seeded_dataset):
    import uuid
    user = FakeUser(sub="u", roles=[])
    with pytest.raises(ThreadNotFoundError):
        await load_thread(session, user=user, thread_id=uuid.uuid4())


async def test_load_thread_cross_user_raises(session, seeded_dataset):
    user_a = FakeUser(sub="user-A", roles=[])
    user_b = FakeUser(sub="user-B", roles=[])
    t = await create_thread(session, user=user_a, dataset=seeded_dataset,
                            first_user_message="A's secret")
    await session.commit()
    with pytest.raises(ThreadAccessDeniedError):
        await load_thread(session, user=user_b, thread_id=t.id)


# ---------- append_message + updated_at ----------------------------------

async def test_append_message_bumps_updated_at(session, seeded_dataset):
    import asyncio
    user = FakeUser(sub="u", roles=[])
    t = await create_thread(session, user=user, dataset=seeded_dataset,
                            first_user_message="hi")
    await session.commit()
    before = t.updated_at
    await asyncio.sleep(0.01)
    await append_message(session, t, role="user", content="follow-up")
    await session.commit()
    await session.refresh(t)
    assert t.updated_at > before


# ---------- delete_thread -------------------------------------------------

async def test_delete_thread_removes_thread_and_cascades_messages(session, seeded_dataset):
    from sqlmodel import select
    user = FakeUser(sub="u", roles=[])
    t = await create_thread(session, user=user, dataset=seeded_dataset,
                            first_user_message="hi")
    await append_message(session, t, role="user", content="hi")
    await append_message(session, t, role="assistant", content="hello")
    await session.commit()

    await delete_thread(session, user=user, thread_id=t.id)
    await session.commit()

    result = await session.exec(select(ChatThread).where(ChatThread.id == t.id))
    assert result.first() is None
    msg_result = await session.exec(select(ChatMessageRow).where(ChatMessageRow.thread_id == t.id))
    assert msg_result.first() is None


async def test_delete_thread_cross_user_raises(session, seeded_dataset):
    user_a = FakeUser(sub="user-A", roles=[])
    user_b = FakeUser(sub="user-B", roles=[])
    t = await create_thread(session, user=user_a, dataset=seeded_dataset,
                            first_user_message="hi")
    await session.commit()
    with pytest.raises(ThreadAccessDeniedError):
        await delete_thread(session, user=user_b, thread_id=t.id)
