"""Tests for /api/chat/{slug} routes."""

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import jwt
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.keycloak import KeycloakClient
from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import Dataset, Datasource, DatasourceType
from cell_explorer_api.main import create_app


async def _first_dataset_id_async(app):
    """Return the seeded dataset's id (async helper for test setup)."""
    from sqlmodel import select
    from sqlmodel.ext.asyncio.session import AsyncSession
    from cell_explorer_api.db.models import Dataset

    engine = app.state.db_engine
    async with AsyncSession(engine) as s:
        return (await s.exec(select(Dataset))).first().id


def _first_dataset_id(app):
    """Return the seeded dataset's id (sync helper for test setup)."""
    import asyncio
    return asyncio.run(_first_dataset_id_async(app))


def _generate_rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture()
def db_url():
    return "sqlite+aiosqlite://"


@pytest.fixture()
def settings(db_url):
    return Settings(database_url=db_url, anthropic_api_key="sk-ant-test")


@pytest.fixture()
def app(settings):
    return create_app(settings)


@pytest_asyncio.fixture()
async def seeded_app(app, db_url):
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        ds = Datasource(
            name="Test CDN",
            type=DatasourceType.S3_CLOUDFRONT,
            base_url="https://cdn.example.com",
        )
        session.add(ds)
        await session.flush()
        session.add(Dataset(
            datasource_id=ds.id,
            name="Public Atlas",
            slug="public-atlas",
            path="datasets/public.zarr",
            is_public=True,
            chat_enabled=True,  # NEW
        ))
        await session.commit()

    app.state.db_engine = engine
    return app


def _set_auth_cookie(client: TestClient, app, *, sub="user-1", roles=None):
    """Generate an RSA-signed test JWT and set it as the cce_access cookie."""
    private_key, public_key = _generate_rsa_keypair()
    settings = app.state.settings
    settings.keycloak_url = "https://auth.example.com"
    settings.keycloak_realm = "test-realm"
    settings.keycloak_client_id = "test-client"
    settings.keycloak_client_secret = "test-secret"
    keycloak = KeycloakClient(settings)
    keycloak._jwks = {
        "test-kid": public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    }
    app.state.keycloak = keycloak

    token = jwt.encode(
        {
            "sub": sub,
            "name": "Test User",
            "realm_access": {"roles": list(roles or [])},
            "iss": "https://auth.example.com/realms/test-realm",
            "aud": "test-client",
            "exp": int(time.time()) + 300,
            "iat": int(time.time()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-kid"},
    )
    client.cookies.set("cce_access", token)


# ---------- fake agent shared across tests ----------

@dataclass
class _FakeObsCol:
    name: str
    dtype: str = "categorical"
    cardinality: int | None = 5
    values: list[str] | None = None


@dataclass
class _FakeDatasetCtx:
    slug: str = "public-atlas"
    name: str = "Public Atlas"
    description: str = "A public test atlas"
    n_obs: int = 100
    n_var: int = 50
    obs_columns: list = None  # set in __post_init__
    embedding_keys: list = None

    def __post_init__(self):
        if self.obs_columns is None:
            self.obs_columns = [
                _FakeObsCol(name="cell_type", values=["T", "B", "M", "NK", "Mono"]),
            ]
        if self.embedding_keys is None:
            self.embedding_keys = ["X_umap"]


@dataclass
class _FakeTool:
    name: str


@dataclass
class _FakeCatalog:
    tools: list

    def all(self):
        return self.tools


class _FakeChatAgent:
    """Minimal stand-in for ChatAgent. Tests script .run() per case."""

    def __init__(self, *, ctx=None, catalog=None, run_events=None, run_raises=None):
        self.dataset_ctx = ctx or _FakeDatasetCtx()
        self.catalog = catalog or _FakeCatalog([_FakeTool("get_dataset_schema")])
        self._run_events = run_events or []
        self._run_raises = run_raises  # exception or callable[int -> exception | None]

    async def run(self, *, messages, view_state=None):
        self.last_view_state = view_state
        for i, ev in enumerate(self._run_events):
            yield ev
            if self._run_raises is not None:
                exc = self._run_raises(i) if callable(self._run_raises) else self._run_raises
                if exc:
                    raise exc


# ---------- GET /context tests ----------

def test_get_context_happy_path(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["slug"] == "public-atlas"
    assert data["name"] == "Public Atlas"
    assert data["description"] == "A public test atlas"
    assert data["n_obs"] == 100
    assert data["n_var"] == 50
    assert data["obs_columns"] == [
        {"name": "cell_type", "dtype": "categorical", "cardinality": 5,
         "values": ["T", "B", "M", "NK", "Mono"]},
    ]
    assert data["embedding_keys"] == ["X_umap"]
    assert data["available_tools"] == ["get_dataset_schema"]


def test_get_context_anonymous_does_not_401(seeded_app):
    """After switching to optional_auth, /context returns 200 for anonymous
    users on chat-enabled datasets (with permission.reason='requires_auth')."""
    client = TestClient(seeded_app)
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 200


def test_get_context_dataset_not_found_returns_404(seeded_app):
    from cell_explorer_api.services.chat_session import DatasetNotFoundError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise DatasetNotFoundError("dataset 'nope' not found")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/nope/context")

    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_get_context_access_denied_returns_403(seeded_app):
    from cell_explorer_api.services.chat_session import AccessDeniedError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise AccessDeniedError("access denied; required roles: ['admin']")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 403
    assert "admin" in response.json()["detail"]


def test_get_context_credential_mint_error_returns_502(seeded_app):
    from cell_explorer_api.services.chat_session import CredentialMintError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise CredentialMintError("signing key missing")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 502
    assert "credentials" in response.json()["detail"].lower()


def test_get_context_zarr_unreachable_returns_502(seeded_app):
    from cell_explorer_api.services.chat_session import ZarrUnreachableError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise ZarrUnreachableError("connection refused")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 502
    assert "zarr" in response.json()["detail"].lower()


# ---------- POST /turns tests ----------

def test_post_turn_validation_empty_messages(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.post("/api/chat/public-atlas/turns", json={"messages": []})

    assert response.status_code == 422


def test_post_turn_validation_last_must_be_user(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.post("/api/chat/public-atlas/turns", json={
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        })

    assert response.status_code == 422


def test_post_turn_validation_non_alternating(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.post("/api/chat/public-atlas/turns", json={
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "again"},
            ],
        })

    assert response.status_code == 422


def test_post_turn_validation_empty_content(seeded_app):
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.post("/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": ""}],
        })

    assert response.status_code == 422


def test_post_turn_streams_ndjson_events(seeded_app):
    """Fake agent yields a TextDelta + Done; response is parseable NDJSON."""
    import json as _json

    from cell_explorer_agent.events import Done, TextDelta, TokenUsage

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    fake = _FakeChatAgent(run_events=[
        TextDelta(text="hello "),
        TextDelta(text="world"),
        Done(usage=TokenUsage(input_tokens=10, output_tokens=2)),
    ])

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        with client.stream("POST", "/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "hi"}],
        }) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("application/x-ndjson")
            lines = [_json.loads(line) for line in r.iter_lines() if line]

    assert lines[0]["type"] == "thread_open"
    assert [l["type"] for l in lines[1:]] == ["text_delta", "text_delta", "done"]
    assert lines[1]["text"] == "hello "
    assert lines[2]["text"] == "world"
    assert lines[3]["usage"]["input_tokens"] == 10


def test_post_turn_dataset_not_found_returns_404(seeded_app):
    from cell_explorer_api.services.chat_session import DatasetNotFoundError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise DatasetNotFoundError("dataset 'nope' not found")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.post("/api/chat/nope/turns", json={
            "messages": [{"role": "user", "content": "hi"}],
        })

    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_post_turn_unauthenticated_returns_401(seeded_app):
    client = TestClient(seeded_app)
    response = client.post("/api/chat/public-atlas/turns", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 401


# ---------- chat_enabled gate tests ----------

def test_get_context_returns_404_when_chat_disabled(seeded_app):
    seeded_app.state.settings.anthropic_api_key = None
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)
    response = client.get("/api/chat/public-atlas/context")
    assert response.status_code == 404
    assert "not available" in response.json()["detail"].lower()


def test_post_turn_returns_404_when_chat_disabled(seeded_app):
    seeded_app.state.settings.anthropic_api_key = None
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)
    response = client.post("/api/chat/public-atlas/turns", json={
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert response.status_code == 404
    assert "not available" in response.json()["detail"].lower()


def test_post_turn_forwards_view_state_to_agent(seeded_app):
    """Wire-level: a `view_state` field on the request reaches agent.run()."""
    import json as _json

    from cell_explorer_agent.events import Done, TextDelta, TokenUsage

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    fake = _FakeChatAgent(run_events=[
        TextDelta(text="ok"),
        Done(usage=TokenUsage(input_tokens=1, output_tokens=1)),
    ])

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        with client.stream("POST", "/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "zoom in"}],
            "view_state": {"embedding": "X_umap", "gene": "CD8A"},
        }) as r:
            assert r.status_code == 200
            # consume stream so the agent.run() generator is actually driven
            for _ in r.iter_lines():
                pass

    assert fake.last_view_state == {"embedding": "X_umap", "gene": "CD8A"}


def test_post_turn_view_state_optional(seeded_app):
    """Omitting view_state keeps the request shape backwards-compatible —
    agent.run() receives None."""
    from cell_explorer_agent.events import Done, TextDelta, TokenUsage

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    fake = _FakeChatAgent(run_events=[
        TextDelta(text="ok"),
        Done(usage=TokenUsage()),
    ])

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        with client.stream("POST", "/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "hi"}],
        }) as r:
            assert r.status_code == 200
            for _ in r.iter_lines():
                pass

    assert fake.last_view_state is None


def test_post_turn_midstream_error_emits_error_event(seeded_app):
    import json as _json

    from cell_explorer_agent.events import TextDelta

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    fake = _FakeChatAgent(
        run_events=[TextDelta(text="part 1 ")],
        run_raises=lambda i: RuntimeError("boom") if i == 0 else None,
    )

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        with client.stream("POST", "/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "hi"}],
        }) as r:
            assert r.status_code == 200  # already 200 by the time the agent raised
            lines = [_json.loads(line) for line in r.iter_lines() if line]

    types = [l["type"] for l in lines]
    assert types == ["thread_open", "text_delta", "error"]
    assert lines[2]["message"] == "chat failed: boom"
    assert lines[2]["retryable"] is False


# ---------- permission field tests ----------

def test_get_context_includes_permission_when_authed_with_no_role_required(seeded_app):
    """Default config: no CHAT_REQUIRED_ROLE => any authed user passes."""
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 200, response.text
    assert response.json()["permission"] == {"can_chat": True, "reason": None}


def test_get_context_permission_missing_role(seeded_app):
    """Authed user without the required role gets can_chat=false."""
    seeded_app.state.settings.chat_required_role = "cell-explorer-chat"
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, roles=["something-else"])
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 200
    assert response.json()["permission"] == {
        "can_chat": False,
        "reason": "missing_role:cell-explorer-chat",
    }


def test_get_context_permission_user_has_role(seeded_app):
    seeded_app.state.settings.chat_required_role = "cell-explorer-chat"
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, roles=["cell-explorer-chat"])
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 200
    assert response.json()["permission"] == {"can_chat": True, "reason": None}


def test_get_context_anonymous_returns_requires_auth(seeded_app):
    """Anonymous on a public chat-enabled dataset: 200 + requires_auth."""
    client = TestClient(seeded_app)
    # No cookie — anonymous.
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 200, response.text
    assert response.json()["permission"] == {
        "can_chat": False,
        "reason": "requires_auth",
    }


def test_get_context_chat_disabled_dataset_returns_404(seeded_app):
    from cell_explorer_api.services.chat_session import ChatDisabledError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise ChatDisabledError("chat is not enabled for dataset 'public-atlas'")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.get("/api/chat/public-atlas/context")

    assert response.status_code == 404
    assert "chat is not available" in response.json()["detail"].lower()


def test_post_turn_missing_chat_role_returns_403(seeded_app):
    seeded_app.state.settings.chat_required_role = "cell-explorer-chat"
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, roles=["something-else"])
    fake = _FakeChatAgent()

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.post("/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "hi"}],
        })

    assert response.status_code == 403
    assert "cell-explorer-chat" in response.json()["detail"]


def test_post_turn_with_chat_role_streams_ok(seeded_app):
    """Sanity: when the role is present, the turn proceeds normally."""
    import json as _json
    from cell_explorer_agent.events import Done, TextDelta, TokenUsage

    seeded_app.state.settings.chat_required_role = "cell-explorer-chat"
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, roles=["cell-explorer-chat"])
    fake = _FakeChatAgent(run_events=[
        TextDelta(text="ok"),
        Done(usage=TokenUsage(input_tokens=1, output_tokens=1)),
    ])

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        with client.stream("POST", "/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "hi"}],
        }) as r:
            assert r.status_code == 200
            lines = [_json.loads(l) for l in r.iter_lines() if l]
    assert lines[0]["type"] == "thread_open"
    assert [l["type"] for l in lines[1:]] == ["text_delta", "done"]


def test_post_turn_chat_disabled_dataset_returns_404(seeded_app):
    from cell_explorer_api.services.chat_session import ChatDisabledError

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app)

    async def _make_agent(**_):
        raise ChatDisabledError("chat is not enabled for dataset 'public-atlas'")

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.post("/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "hi"}],
        })

    assert response.status_code == 404
    assert "chat is not available" in response.json()["detail"].lower()


# ---------- thread persistence tests ----------

def test_post_turn_creates_new_thread_when_thread_id_is_null(seeded_app):
    """First send with thread_id=None creates a new ChatThread."""
    import json as _json
    from cell_explorer_agent.events import Done, TextDelta, TokenUsage

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    fake = _FakeChatAgent(run_events=[
        TextDelta(text="hello"),
        Done(usage=TokenUsage()),
    ])

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        with client.stream("POST", "/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "Show me CD8A in T cells"}],
        }) as r:
            assert r.status_code == 200
            lines = [_json.loads(l) for l in r.iter_lines() if l]

    # First event must be thread_open with the derived title
    assert lines[0]["type"] == "thread_open"
    assert lines[0]["title"] == "Show me CD8A in T cells"
    thread_id = lines[0]["thread_id"]

    # Verify the thread + messages were persisted
    import asyncio
    from sqlmodel import select
    from cell_explorer_api.db.models import ChatThread, ChatMessageRow
    from sqlmodel.ext.asyncio.session import AsyncSession

    async def _verify():
        engine = seeded_app.state.db_engine
        async with AsyncSession(engine) as s:
            t = (await s.exec(select(ChatThread))).first()
            assert t is not None
            assert str(t.id) == thread_id
            assert t.user_sub == "user-1"
            msgs = (await s.exec(
                select(ChatMessageRow).where(ChatMessageRow.thread_id == t.id)
                .order_by(ChatMessageRow.created_at.asc())
            )).all()
            assert [(m.role, m.content) for m in msgs] == [
                ("user", "Show me CD8A in T cells"),
                ("assistant", "hello"),
            ]
    asyncio.run(_verify())


def test_post_turn_appends_to_existing_thread_when_thread_id_set(seeded_app):
    """Second send with the same thread_id appends to it."""
    import asyncio
    import json as _json
    from cell_explorer_agent.events import Done, TextDelta, TokenUsage
    from cell_explorer_api.db.models import ChatThread
    from sqlmodel.ext.asyncio.session import AsyncSession

    async def _seed():
        engine = seeded_app.state.db_engine
        async with AsyncSession(engine) as s:
            t = ChatThread(user_sub="user-1", dataset_id=await _first_dataset_id_async(seeded_app),
                           title="My existing thread")
            s.add(t)
            await s.commit()
            await s.refresh(t)
            return t.id
    existing_id = asyncio.run(_seed())

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    fake = _FakeChatAgent(run_events=[
        TextDelta(text="more"),
        Done(usage=TokenUsage()),
    ])

    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        with client.stream("POST", "/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "follow-up"}],
            "thread_id": str(existing_id),
        }) as r:
            assert r.status_code == 200
            lines = [_json.loads(l) for l in r.iter_lines() if l]

    assert lines[0]["thread_id"] == str(existing_id)


def test_post_turn_thread_id_cross_user_returns_403(seeded_app):
    import asyncio
    from cell_explorer_api.db.models import ChatThread
    from sqlmodel.ext.asyncio.session import AsyncSession

    async def _seed():
        engine = seeded_app.state.db_engine
        async with AsyncSession(engine) as s:
            t = ChatThread(user_sub="other-user", dataset_id=await _first_dataset_id_async(seeded_app),
                           title="not yours")
            s.add(t)
            await s.commit()
            await s.refresh(t)
            return t.id
    other_id = asyncio.run(_seed())

    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    fake = _FakeChatAgent()
    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.post("/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "hi"}],
            "thread_id": str(other_id),
        })
    assert response.status_code == 403


def test_post_turn_thread_id_missing_returns_404(seeded_app):
    import uuid
    client = TestClient(seeded_app)
    _set_auth_cookie(client, seeded_app, sub="user-1")
    fake = _FakeChatAgent()
    async def _make_agent(**_):
        return fake

    with patch("cell_explorer_api.routes.chat.make_chat_agent", _make_agent):
        response = client.post("/api/chat/public-atlas/turns", json={
            "messages": [{"role": "user", "content": "hi"}],
            "thread_id": str(uuid.uuid4()),
        })
    assert response.status_code == 404
