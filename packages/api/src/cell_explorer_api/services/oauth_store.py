"""Hashed storage for OAuth grants, tokens and MCP resource keys.

Secrets are random (``secrets.token_urlsafe(32)``) and only their SHA-256 hex
digest is stored; plaintext is returned once, at creation.
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import and_, exists, or_
from sqlmodel import col, delete, select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.db.models import utcnow
from cell_explorer_api.db.oauth_models import (
    McpResource,
    OAuthCode,
    OAuthGrant,
    OAuthPendingAuthorization,
    OAuthToken,
)

CODE_TTL = 60
ACCESS_TTL = 3600
API_TTL = 600
GRANT_TTL = timedelta(days=30)
PENDING_TTL = 600
# An approved grant whose code was never exchanged is dropped after this.
UNEXCHANGED_GRANT_TTL = timedelta(minutes=10)

_TTL = {"access": ACCESS_TTL, "api": API_TTL}


def hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def create_grant(
    db: AsyncSession, *, user: User, client_id: str, resource: str, scopes: list[str] | None = None
) -> OAuthGrant:
    """Record consent; roles are copied and never change for the grant's lifetime."""
    grant = OAuthGrant(
        user_sub=user.sub,
        user_email=user.email or "",
        user_name=user.name or "",
        roles=list(user.roles),
        scopes=list(scopes or []),
        client_id=client_id,
        resource=resource,
        expires_at=utcnow() + GRANT_TTL,
    )
    db.add(grant)
    await db.commit()
    await db.refresh(grant)
    return grant


async def issue_token(
    db: AsyncSession,
    *,
    grant: OAuthGrant,
    kind: Literal["access", "refresh", "api"],
    audience: str,
    not_after: datetime | None = None,
) -> tuple[str, OAuthToken]:
    """Create a token; returns (plaintext, row). Never outlives its grant, nor ``not_after``."""
    await db.refresh(grant)  # earlier commits may have expired it
    plaintext = secrets.token_urlsafe(32)
    expires_at = grant.expires_at
    if kind != "refresh":
        expires_at = min(utcnow() + timedelta(seconds=_TTL[kind]), expires_at)
    if not_after is not None:
        expires_at = min(expires_at, not_after)
    token = OAuthToken(
        token_hash=hash_secret(plaintext),
        kind=kind,
        grant_id=grant.id,
        client_id=grant.client_id,
        audience=audience,
        expires_at=expires_at,
    )
    db.add(token)
    await db.commit()
    await db.refresh(token)
    await db.refresh(grant)  # commit expires the caller's grant
    return plaintext, token


async def find_token(
    db: AsyncSession, plaintext: str, *, kind: str, allow_used: bool = False
) -> tuple[OAuthToken, OAuthGrant] | None:
    """Return the token and its grant if still valid, else None.

    ``allow_used`` still returns a rotated refresh token so callers can detect reuse.
    """
    token = await db.get(OAuthToken, hash_secret(plaintext))
    if token is None or token.kind != kind:
        return None
    now = utcnow()
    if token.expires_at <= now or (token.used_at is not None and not allow_used):
        return None
    grant = await db.get(OAuthGrant, token.grant_id)
    if grant is None or grant.revoked_at is not None or grant.expires_at <= now:
        return None
    return token, grant


async def revoke_grant(db: AsyncSession, grant_id: uuid.UUID) -> None:
    await db.exec(
        update(OAuthGrant)
        .where(col(OAuthGrant.id) == grant_id, col(OAuthGrant.revoked_at).is_(None))
        .values(revoked_at=utcnow())
    )
    await db.commit()


async def revoke_user_grants(db: AsyncSession, user_sub: str) -> int:
    """Revoke all of a user's live grants; returns how many were revoked."""
    result = await db.exec(
        update(OAuthGrant)
        .where(col(OAuthGrant.user_sub) == user_sub, col(OAuthGrant.revoked_at).is_(None))
        .values(revoked_at=utcnow())
    )
    await db.commit()
    return result.rowcount


async def purge_expired(db: AsyncSession) -> None:
    """Delete expired pending requests, codes and tokens, and dead grants with their rows.

    A grant is dead once expired or revoked, or when it is older than
    ``UNEXCHANGED_GRANT_TTL`` and has no tokens (approved, code never exchanged).
    """
    now = utcnow()
    has_token = exists().where(col(OAuthToken.grant_id) == col(OAuthGrant.id))
    dead = await db.exec(
        select(OAuthGrant.id).where(
            or_(
                col(OAuthGrant.expires_at) <= now,
                col(OAuthGrant.revoked_at).is_not(None),
                and_(col(OAuthGrant.created_at) <= now - UNEXCHANGED_GRANT_TTL, ~has_token),
            )
        )
    )
    dead_ids = list(dead.all())
    # Children first: SQLite enforces the cascade only with foreign keys on.
    for start in range(0, len(dead_ids), 500):
        chunk = dead_ids[start : start + 500]
        await db.exec(delete(OAuthCode).where(col(OAuthCode.grant_id).in_(chunk)))
        await db.exec(delete(OAuthToken).where(col(OAuthToken.grant_id).in_(chunk)))
        await db.exec(delete(OAuthGrant).where(col(OAuthGrant.id).in_(chunk)))
    await db.exec(delete(OAuthToken).where(col(OAuthToken.expires_at) <= now))
    await db.exec(delete(OAuthCode).where(col(OAuthCode.expires_at) <= now))
    await db.exec(delete(OAuthPendingAuthorization).where(col(OAuthPendingAuthorization.expires_at) <= now))
    await db.commit()


async def create_mcp_resource(db: AsyncSession, *, url: str, name: str) -> tuple[str, McpResource]:
    """Register an MCP server; the plaintext key is returned once."""
    key = secrets.token_urlsafe(32)
    resource = McpResource(url=url, name=name, key_hash=hash_secret(key))
    db.add(resource)
    await db.commit()
    await db.refresh(resource)
    return key, resource


async def rotate_mcp_resource_key(
    db: AsyncSession, resource: McpResource, *, name: str
) -> tuple[str, McpResource]:
    """Re-register a revoked resource: same row, new key; the plaintext is returned once."""
    key = secrets.token_urlsafe(32)
    resource.key_hash = hash_secret(key)
    resource.name = name
    resource.revoked_at = None
    db.add(resource)
    await db.commit()
    await db.refresh(resource)
    return key, resource


async def find_mcp_resource_by_key(db: AsyncSession, key: str) -> McpResource | None:
    result = await db.exec(
        select(McpResource).where(
            col(McpResource.key_hash) == hash_secret(key), col(McpResource.revoked_at).is_(None)
        )
    )
    return result.first()


async def find_mcp_resource_by_url(db: AsyncSession, url: str) -> McpResource | None:
    result = await db.exec(select(McpResource).where(col(McpResource.url) == url))
    return result.first()
