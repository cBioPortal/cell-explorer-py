"""Application info endpoint."""

from importlib.metadata import version

from fastapi import APIRouter, Request
from pydantic import BaseModel

from cell_explorer_api.branding import DEFAULT_BRAND, Brand

router = APIRouter(tags=["status"])


class BrandInfo(BaseModel):
    name: str
    short_name: str
    tagline: str
    logo_href: str | None
    # Resolved URLs, not filenames: the frontend never builds /brand/ paths.
    logo_on_light: str | None
    logo_on_dark: str | None
    colors: dict[str, str]


class InfoResponse(BaseModel):
    version: str
    environment: str
    git_sha: str | None
    auth_enabled: bool
    chat_enabled: bool
    # Null when unset, which the frontend reads as "load no analytics".
    google_analytics_id: str | None
    # Null when no BRAND_DIR is configured, which the frontend reads as
    # "use the built-in cBioPortal branding".
    brand: BrandInfo | None


def _asset_url(filename: str | None) -> str | None:
    return f"/brand/{filename}" if filename else None


def _brand_info(brand: Brand) -> BrandInfo | None:
    if brand == DEFAULT_BRAND:
        return None
    colors = {"ink": brand.colors.ink, "themeColor": brand.colors.theme_color}
    if brand.colors.ink_deep is not None:
        colors["inkDeep"] = brand.colors.ink_deep
    return BrandInfo(
        name=brand.name,
        short_name=brand.short_name,
        tagline=brand.tagline,
        logo_href=brand.logo_href,
        logo_on_light=_asset_url(brand.logo.on_light),
        logo_on_dark=_asset_url(brand.logo.on_dark),
        colors=colors,
    )


@router.get("/info")
async def info(request: Request) -> InfoResponse:
    settings = request.app.state.settings
    return InfoResponse(
        version=version("cell-explorer-api"),
        environment=settings.environment,
        git_sha=settings.git_sha,
        auth_enabled=settings.auth_enabled,
        chat_enabled=settings.chat_enabled,
        google_analytics_id=settings.google_analytics_id,
        brand=_brand_info(request.app.state.brand),
    )
