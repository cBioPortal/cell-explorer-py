"""OAuth settings and tables for the MCP signed-in tier."""

import uuid
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.config import Settings
from cell_explorer_api.db.models import utcnow

AUTH = dict(
    keycloak_url="https://kc.example.org",
    keycloak_realm="r",
    keycloak_client_id="c",
    keycloak_client_secret="s",
)


def test_oauth_disabled_without_public_url():
    s = Settings(**AUTH)
    assert s.public_url is None
    assert s.oauth_issuer_url is None
    assert s.oauth_enabled is False


def test_oauth_disabled_without_auth():
    s = Settings(public_url="https://ce.example.org")
    assert s.oauth_enabled is False


def test_oauth_issuer_url_from_public_url(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL", "https://ce.example.org/")
    s = Settings(**AUTH)
    assert s.oauth_issuer_url == "https://ce.example.org/api/oauth"
    assert s.oauth_enabled is True


@pytest.mark.asyncio
async def test_oauth_tables_roundtrip():
    from cell_explorer_api.db.oauth_models import (
        McpResource,
        OAuthClient,
        OAuthCode,
        OAuthGrant,
        OAuthPendingAuthorization,
        OAuthToken,
    )

    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    now = utcnow()
    grant = OAuthGrant(
        user_sub="u1", user_email="a@b.c", user_name="A", roles=["r1", "r2"],
        client_id="cid", resource="https://x/mcp", expires_at=now + timedelta(days=1),
    )
    grant_id = grant.id
    async with AsyncSession(engine) as s:
        s.add(OAuthClient(client_id="cid", info={"redirect_uris": ["http://l/cb"]}))
        s.add(McpResource(url="https://x/mcp", name="x", key_hash="h" * 64))
        s.add(OAuthPendingAuthorization(
            id="p1", client_id="cid", params={"state": "z"}, csrf="c", expires_at=now,
        ))
        s.add(grant)
        await s.commit()
        s.add(OAuthCode(
            code_hash="c" * 64, grant_id=grant_id, client_id="cid", code_challenge="ch",
            redirect_uri="http://l/cb", redirect_uri_provided_explicitly=True,
            resource="https://x/mcp", expires_at=now,
        ))
        s.add(OAuthToken(
            token_hash="t" * 64, kind="refresh", grant_id=grant_id, client_id="cid",
            audience="https://x/mcp", expires_at=now,
        ))
        await s.commit()

    async with AsyncSession(engine) as s:
        assert (await s.get(OAuthClient, "cid")).info == {"redirect_uris": ["http://l/cb"]}
        g = await s.get(OAuthGrant, grant_id)
        assert g.roles == ["r1", "r2"] and g.revoked_at is None
        t = await s.get(OAuthToken, "t" * 64)
        assert t.kind == "refresh" and t.used_at is None
        assert (await s.get(OAuthCode, "c" * 64)).redirect_uri_provided_explicitly is True
        assert (await s.get(OAuthPendingAuthorization, "p1")).params == {"state": "z"}
    await engine.dispose()
