from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from cell_explorer_api.cli.config import AuthConfig
from cell_explorer_api.cli.errors import SessionExpiredError
from cell_explorer_api.cli.tokens import ensure_fresh_access_token


def _cfg(access_offset_s: float, refresh_offset_s: float = 30 * 86400) -> AuthConfig:
    now = datetime.now(timezone.utc)
    return AuthConfig(
        api_url="http://localhost:8000",
        access_token="ACC",
        refresh_token="REF",
        access_expires_at=now + timedelta(seconds=access_offset_s),
        refresh_expires_at=now + timedelta(seconds=refresh_offset_s),
        username="u",
    )


@pytest.mark.asyncio
async def test_token_still_valid_is_returned_unchanged():
    cfg = _cfg(access_offset_s=300)  # 5 min ahead
    keycloak = MagicMock()
    keycloak.refresh_token = AsyncMock()

    result = await ensure_fresh_access_token(cfg, keycloak)

    assert result is cfg
    keycloak.refresh_token.assert_not_called()


@pytest.mark.asyncio
async def test_token_near_expiry_triggers_refresh():
    cfg = _cfg(access_offset_s=30)  # 30 s ahead — inside 60 s threshold
    keycloak = MagicMock()
    new_tokens = {
        "access_token": "NEW_ACC",
        "refresh_token": "NEW_REF",
        "expires_in": 300,
    }
    keycloak.refresh_token = AsyncMock(return_value=new_tokens)

    result = await ensure_fresh_access_token(cfg, keycloak)

    keycloak.refresh_token.assert_awaited_once_with("REF")
    assert result.access_token == "NEW_ACC"
    assert result.refresh_token == "NEW_REF"
    # The AuthConfig's api_url and username are preserved
    assert result.api_url == cfg.api_url
    assert result.username == cfg.username


@pytest.mark.asyncio
async def test_refresh_failure_raises_session_expired():
    cfg = _cfg(access_offset_s=-60)  # already expired
    keycloak = MagicMock()
    keycloak.refresh_token = AsyncMock(side_effect=Exception("refresh denied"))

    with pytest.raises(SessionExpiredError, match="refresh denied"):
        await ensure_fresh_access_token(cfg, keycloak)
