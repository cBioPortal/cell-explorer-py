"""OAuth 2.1 authorization server for MCP clients, backed by the catalog DB.

The MCP SDK's auth routes call this provider. Sign-in itself happens on the
consent page (``approve``); refresh never calls the identity provider, and the
roles copied onto the grant at approval are the only roles tokens carry.
"""

import math
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from sqlmodel import col, delete, update
from sqlmodel.ext.asyncio.session import AsyncSession

from cell_explorer_api.auth.models import User
from cell_explorer_api.db.models import utcnow
from cell_explorer_api.db.oauth_models import (
    OAuthClient,
    OAuthCode,
    OAuthGrant,
    OAuthPendingAuthorization,
)
from cell_explorer_api.db.oauth_models import OAuthToken as OAuthTokenRow
from cell_explorer_api.services.oauth_store import (
    CODE_TTL,
    PENDING_TTL,
    create_grant,
    find_mcp_resource_by_url,
    find_token,
    hash_secret,
    issue_token,
    revoke_grant,
)


def _epoch(dt: datetime) -> int:
    return int(dt.replace(tzinfo=UTC).timestamp())


def _live(grant: OAuthGrant | None) -> bool:
    return grant is not None and grant.revoked_at is None and grant.expires_at > utcnow()


class CellExplorerOAuthProvider(OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]):
    def __init__(self, session_factory: Callable[[], AsyncSession], issuer_url: str):
        self._session = session_factory
        self._issuer_url = issuer_url.rstrip("/")

    # Clients

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        async with self._session() as db:
            row = await db.get(OAuthClient, client_id)
            return OAuthClientInformationFull.model_validate(row.info) if row else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        async with self._session() as db:
            db.add(OAuthClient(client_id=client_info.client_id, info=client_info.model_dump(mode="json")))
            await db.commit()

    # Authorization

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        async with self._session() as db:
            await self._require_resource(db, params.resource)
            pending_id = secrets.token_urlsafe(32)
            pending = OAuthPendingAuthorization(
                id=pending_id,
                client_id=client.client_id,
                params=params.model_dump(mode="json"),
                csrf=secrets.token_urlsafe(32),
                expires_at=utcnow() + timedelta(seconds=PENDING_TTL),
            )
            db.add(pending)
            await db.commit()
            return f"{self._issuer_url}/consent?request={pending_id}"

    async def approve(self, pending_id: str, csrf: str, user: User) -> str:
        """Consent given: grant + single-use code; returns the client redirect URL."""
        async with self._session() as db:
            pending = await db.get(OAuthPendingAuthorization, pending_id)
            if pending is None or not secrets.compare_digest(pending.csrf.encode(), csrf.encode()):
                raise AuthorizeError("invalid_request", "unknown authorization request")
            client_id, raw, expires_at = pending.client_id, pending.params, pending.expires_at
            # Claim the request atomically so it can be approved only once.
            claimed = await db.exec(
                delete(OAuthPendingAuthorization).where(col(OAuthPendingAuthorization.id) == pending_id)
            )
            await db.commit()
            if claimed.rowcount != 1 or expires_at <= utcnow():
                raise AuthorizeError("invalid_request", "authorization request expired")
            params = AuthorizationParams.model_validate(raw)
            await self._require_resource(db, params.resource)

            grant = await create_grant(
                db, user=user, client_id=client_id, resource=params.resource, scopes=params.scopes
            )
            code = secrets.token_urlsafe(32)
            db.add(
                OAuthCode(
                    code_hash=hash_secret(code),
                    grant_id=grant.id,
                    client_id=client_id,
                    code_challenge=params.code_challenge,
                    redirect_uri=str(params.redirect_uri),
                    redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
                    resource=params.resource,
                    expires_at=utcnow() + timedelta(seconds=CODE_TTL),
                )
            )
            await db.commit()
            return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        async with self._session() as db:
            row = await db.get(OAuthCode, hash_secret(authorization_code))
            if row is None or row.expires_at <= utcnow():
                return None
            grant = await db.get(OAuthGrant, row.grant_id)
            if not _live(grant):
                return None
            return AuthorizationCode(
                code=authorization_code,
                scopes=grant.scopes or [],
                expires_at=_epoch(row.expires_at),
                client_id=row.client_id,
                code_challenge=row.code_challenge,
                redirect_uri=row.redirect_uri,
                redirect_uri_provided_explicitly=row.redirect_uri_provided_explicitly,
                resource=row.resource,
                subject=grant.user_sub,
            )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        code_hash = hash_secret(authorization_code.code)
        async with self._session() as db:
            row = await db.get(OAuthCode, code_hash)
            grant_id = row.grant_id if row else None
            # Single use: deleting the row is the claim.
            claimed = await db.exec(delete(OAuthCode).where(col(OAuthCode.code_hash) == code_hash))
            await db.commit()
            if claimed.rowcount != 1:
                raise TokenError("invalid_grant", "authorization code already used")
            grant = await db.get(OAuthGrant, grant_id)
            if not _live(grant):
                raise TokenError("invalid_grant", "grant expired or revoked")
            return await self._issue_pair(db, grant)

    # Refresh

    async def load_refresh_token(self, client: OAuthClientInformationFull, refresh_token: str) -> RefreshToken | None:
        async with self._session() as db:
            found = await find_token(db, refresh_token, kind="refresh", allow_used=True)
            if found is None:
                return None
            token, grant = found
            if token.used_at is not None:
                await revoke_grant(db, grant.id)  # rotated token replayed
                return None
            return RefreshToken(
                token=refresh_token,
                client_id=token.client_id,
                scopes=grant.scopes or [],
                expires_at=_epoch(token.expires_at),
                resource=grant.resource,
                subject=grant.user_sub,
            )

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        token_hash = hash_secret(refresh_token.token)
        async with self._session() as db:
            row = await db.get(OAuthTokenRow, token_hash)
            if row is None or row.kind != "refresh":
                raise TokenError("invalid_grant", "refresh token does not exist")
            grant_id = row.grant_id
            claimed = await db.exec(
                update(OAuthTokenRow)
                .where(col(OAuthTokenRow.token_hash) == token_hash, col(OAuthTokenRow.used_at).is_(None))
                .values(used_at=utcnow())
            )
            await db.commit()
            if claimed.rowcount != 1:
                await revoke_grant(db, grant_id)
                raise TokenError("invalid_grant", "refresh token reused; grant revoked")
            found = await find_token(db, refresh_token.token, kind="refresh", allow_used=True)
            if found is None:
                raise TokenError("invalid_grant", "refresh token expired or revoked")
            return await self._issue_pair(db, found[1])

    # Access + revocation

    async def load_access_token(self, token: str) -> AccessToken | None:
        async with self._session() as db:
            found = await find_token(db, token, kind="access")
            if found is None:
                return None
            row, grant = found
            return AccessToken(
                token=token,
                client_id=row.client_id,
                scopes=grant.scopes or [],
                expires_at=_epoch(row.expires_at),
                resource=grant.resource,
                subject=grant.user_sub,
            )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        async with self._session() as db:
            row = await db.get(OAuthTokenRow, hash_secret(token.token))
            if row is not None:
                await revoke_grant(db, row.grant_id)

    # Helpers

    @staticmethod
    async def _require_resource(db: AsyncSession, resource: str | None) -> None:
        found = await find_mcp_resource_by_url(db, resource) if resource else None
        if found is None or found.revoked_at is not None:
            raise AuthorizeError("invalid_request", "resource must be a registered MCP server URL")

    @staticmethod
    async def _issue_pair(db: AsyncSession, grant: OAuthGrant) -> OAuthToken:
        access, row = await issue_token(db, grant=grant, kind="access", audience=grant.resource)
        access_expires_at = row.expires_at  # read before the next commit expires the row
        refresh, _ = await issue_token(db, grant=grant, kind="refresh", audience=grant.resource)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            # From the issued row (capped by the grant); never 0 or negative.
            expires_in=max(1, math.ceil((access_expires_at - utcnow()).total_seconds())),
            scope=" ".join(grant.scopes) or None,
            refresh_token=refresh,
        )
