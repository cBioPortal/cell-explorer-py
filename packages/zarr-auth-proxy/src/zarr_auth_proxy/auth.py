"""JWT validation and path authorization."""

import logging

import jwt

logger = logging.getLogger(__name__)


def validate_token(token: str, public_key: bytes) -> dict | None:
    """Decode and validate an RS256 JWT.

    Returns the claims dict if valid, None if invalid/expired.
    """
    try:
        return jwt.decode(token, public_key, algorithms=["RS256"])
    except Exception as e:
        logger.debug("Token validation failed: %s", e)
        return None


def is_path_authorized(requested_path: str, token_path: str) -> bool:
    """Check if the requested path falls within the token's allowed path.

    Uses directory-boundary matching to prevent 'brca.zarr.evil'
    from matching 'brca.zarr'.
    """
    # Normalize: strip leading/trailing slashes
    requested = requested_path.strip("/")
    allowed = token_path.strip("/")

    if requested == allowed:
        return True

    # Must match at a directory boundary
    return requested.startswith(allowed + "/")
