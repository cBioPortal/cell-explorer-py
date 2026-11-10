"""Chat without signing in, on datasets that allow it (anonymous_chat)."""

import asyncio
import re
import uuid
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.db.models import ChatThread, Dataset

from .conftest import _set_auth_cookie
from .test_chat import _FakeChatAgent

UUID_RE = re.compile(r"cce_anon=([0-9a-f-]{36});")


def _set_dataset(app, **fields):
    async def _go():
        async with AsyncSession(app.state.db_engine) as s:
            ds = (await s.exec(select(Dataset).where(Dataset.slug == "public-atlas"))).first()
            for k, v in fields.items():
                setattr(ds, k, v)
            s.add(ds)
            await s.commit()
    asyncio.run(_go())


def _threads(app):
    async def _go():
        async with AsyncSession(app.state.db_engine) as s:
            return list((await s.exec(select(ChatThread))).all())
    return asyncio.run(_go())


def _agent():
    from cell_explorer_agent.events import Done, TextDelta, TokenUsage
    return _FakeChatAgent(run_events=[TextDelta(text="hi"), Done(usage=TokenUsage())])


def _turn(client, text="hello", thread_id=None):
    async def _make_agent(**_):
        return _agent()
    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        body = {"messages": [{"role": "user", "content": text}]}
        if thread_id:
            body["thread_id"] = thread_id
        with client.stream("POST", "/api/chat/public-atlas/turns", json=body) as r:
            lines = [line for line in r.iter_lines() if line]
            return r, lines


def _context(client):
    async def _make_agent(**_):
        return _FakeChatAgent()
    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        return client.get("/api/chat/public-atlas/context")


# ---------- /context ----------

def test_context_offers_anonymous_chat_on_enabled_public_dataset(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    response = _context(TestClient(seeded_app))
    assert response.status_code == 200, response.text
    assert response.json()["permission"] == {"can_chat": True, "reason": None, "anonymous": True}


def test_context_still_requires_auth_when_switch_off(seeded_app):
    response = _context(TestClient(seeded_app))
    assert response.json()["permission"] == {"can_chat": False, "reason": "requires_auth", "anonymous": False}


def test_context_still_requires_auth_on_private_dataset(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True, is_public=False)
    response = _context(TestClient(seeded_app))
    assert response.json()["permission"]["can_chat"] is False


def test_context_still_requires_auth_when_chat_role_set(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    seeded_app.state.settings.chat_required_role = "chat-user"
    response = _context(TestClient(seeded_app))
    assert response.json()["permission"]["can_chat"] is False


# ---------- turns ----------

def test_anonymous_turn_creates_thread_and_sets_cookie(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    client = TestClient(seeded_app)
    r, lines = _turn(client)
    assert r.status_code == 200
    cookie = r.headers["set-cookie"]
    m = UUID_RE.search(cookie)
    assert m, cookie
    assert "Max-Age=604800" in cookie and "HttpOnly" in cookie and "Path=/api" in cookie
    [thread] = _threads(seeded_app)
    assert thread.user_sub == f"anon:{m.group(1)}"


def test_anonymous_turn_refused_when_switch_off(seeded_app):
    r, _ = _turn(TestClient(seeded_app))
    assert r.status_code == 401


def test_returning_visitor_keeps_the_same_id(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    client = TestClient(seeded_app)
    r, _ = _turn(client)
    anon = UUID_RE.search(r.headers["set-cookie"]).group(1)
    client.cookies.set("cce_anon", anon, path="/api")
    r2, _ = _turn(client, "again")
    assert "cce_anon=" not in r2.headers.get("set-cookie", "")
    assert {t.user_sub for t in _threads(seeded_app)} == {f"anon:{anon}"}


def test_forged_anon_cookie_is_replaced(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    client = TestClient(seeded_app)
    client.cookies.set("cce_anon", "x'; drop table chat_threads;--", path="/api")
    r, _ = _turn(client)
    assert r.status_code == 200
    new = UUID_RE.search(r.headers["set-cookie"]).group(1)
    assert str(uuid.UUID(new)) == new
    [thread] = _threads(seeded_app)
    assert thread.user_sub == f"anon:{new}"


def test_anonymous_visitors_are_isolated(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    a, b = TestClient(seeded_app), TestClient(seeded_app)
    a.cookies.set("cce_anon", str(uuid.uuid4()), path="/api")
    b.cookies.set("cce_anon", str(uuid.uuid4()), path="/api")
    _turn(a)
    [thread] = _threads(seeded_app)
    tid = str(thread.id)
    assert b.get("/api/chat/public-atlas/threads").json()["threads"] == []
    assert a.get("/api/chat/public-atlas/threads").json()["threads"][0]["id"] == tid
    assert b.get(f"/api/chat/public-atlas/threads/{tid}").status_code in (403, 404)
    assert b.delete(f"/api/chat/public-atlas/threads/{tid}").status_code in (403, 404)
    assert a.get(f"/api/chat/public-atlas/threads/{tid}").status_code == 200


def test_switch_off_closes_anonymous_history_too(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    client = TestClient(seeded_app)
    client.cookies.set("cce_anon", str(uuid.uuid4()), path="/api")
    _turn(client)
    _set_dataset(seeded_app, anonymous_chat=False)
    assert client.get("/api/chat/public-atlas/threads").status_code == 401


def test_expired_session_refreshes_instead_of_going_anonymous(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="real-user")
    valid = client.cookies.get("cce_access")
    client.cookies.set("cce_access", "not-a-jwt")
    client.cookies.set("cce_refresh", "refresh-token")
    seeded_app.state.oidc.refresh_token = AsyncMock(
        return_value={"access_token": valid, "refresh_token": "refresh-2"},
    )
    r, _ = _turn(client)
    assert r.status_code == 200
    [thread] = _threads(seeded_app)
    assert thread.user_sub == "real-user"


def test_signed_in_user_unchanged(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="real-user")
    r, _ = _turn(client)
    assert r.status_code == 200
    assert "cce_anon=" not in r.headers.get("set-cookie", "")
    [thread] = _threads(seeded_app)
    assert thread.user_sub == "real-user"


# ---------- limits ----------

def test_anonymous_turn_over_the_limit_is_refused_with_the_limit_message(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    seeded_app.state.settings.anon_chat_per_visitor_per_hour = 1
    client = TestClient(seeded_app)
    client.cookies.set("cce_anon", str(uuid.uuid4()), path="/api")
    r, _ = _turn(client)
    assert r.status_code == 200
    r2, _ = _turn(client, "again")
    assert r2.status_code == 429


def test_signed_in_users_are_not_limited(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    seeded_app.state.settings.anon_chat_per_visitor_per_hour = 1
    seeded_app.state.settings.anon_chat_per_day = 1
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="real-user")
    for text in ("one", "two", "three"):
        r, _ = _turn(client, text)
        assert r.status_code == 200


# ---------- size caps on signed-out turns ----------

def _post(client, body):
    async def _make_agent(**_):
        return _agent()
    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        with client.stream("POST", "/api/chat/public-atlas/turns", json=body) as r:
            r.read()
            return r


def _history(n_turns, chars=10):
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": "q" * chars})
        msgs.append({"role": "assistant", "content": "a" * chars})
    msgs.append({"role": "user", "content": "last"})
    return msgs


def _anon_client(app):
    _set_dataset(app, anonymous_chat=True)
    client = TestClient(app)
    client.cookies.set("cce_anon", str(uuid.uuid4()), path="/api")
    return client


def test_signed_out_turn_with_too_many_messages_is_refused(seeded_app):
    client = _anon_client(seeded_app)
    assert _post(client, {"messages": _history(14)}).status_code == 200  # 29 messages
    assert _post(client, {"messages": _history(15)}).status_code == 413  # 31 messages


def test_signed_out_turn_with_a_long_message_is_refused(seeded_app):
    client = _anon_client(seeded_app)
    assert _post(client, {"messages": [{"role": "user", "content": "x" * 4001}]}).status_code == 413


def test_signed_out_turn_with_too_much_history_is_refused(seeded_app):
    client = _anon_client(seeded_app)
    assert _post(client, {"messages": _history(12, chars=3500)}).status_code == 413  # 84k chars


def test_signed_out_turn_with_a_large_view_state_is_refused(seeded_app):
    client = _anon_client(seeded_app)
    big = {"x": "v" * 5000}
    body = {"messages": [{"role": "user", "content": "hi"}], "view_state": big}
    assert _post(client, body).status_code == 413


def test_a_refused_oversized_turn_is_not_counted(seeded_app):
    client = _anon_client(seeded_app)
    seeded_app.state.settings.anon_chat_per_visitor_per_hour = 1
    assert _post(client, {"messages": [{"role": "user", "content": "x" * 4001}]}).status_code == 413
    assert _post(client, {"messages": [{"role": "user", "content": "ok"}]}).status_code == 200


def test_signed_in_turns_are_not_size_capped(seeded_app):
    _set_dataset(seeded_app, anonymous_chat=True)
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="real-user")
    assert _post(client, {"messages": _history(15)}).status_code == 200
