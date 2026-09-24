"""Deployment branding: model, validation, and loading.

A brand is operator-supplied deployment data, not repo content. Every field is
optional and falls back independently, and every validation failure degrades
that one field rather than the application — a bad bundle may look wrong, but
it must never stop the app from serving.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

ALLOWED_ASSET_SUFFIXES = frozenset({".svg", ".png", ".ico", ".jpg", ".jpeg", ".webp"})

# `ink` is the background of a light-on-dark identity band. Anything paler makes
# the header text unreadable, so it is rejected rather than rendered illegibly.
MAX_INK_LUMINANCE = 0.35

_MAX_LEN = {
    "name": 120,
    "shortName": 40,
    "title": 160,
    "tagline": 200,
    "alt": 120,
}


def relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of an #RRGGBB string, 0.0 (black) to 1.0 (white)."""
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def _linear(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _linear(r) + 0.7152 * _linear(g) + 0.0722 * _linear(b)


@dataclass(frozen=True)
class BrandColors:
    ink: str
    ink_deep: str | None
    theme_color: str


@dataclass(frozen=True)
class BrandLogo:
    on_light: str | None
    on_dark: str | None
    alt: str


@dataclass(frozen=True)
class BrandFavicon:
    ico: str | None
    svg: str | None
    png192: str | None
    png512: str | None
    apple_touch: str | None


@dataclass(frozen=True)
class Brand:
    name: str
    short_name: str
    title: str
    tagline: str
    logo_href: str | None
    logo: BrandLogo
    colors: BrandColors
    favicon: BrandFavicon


DEFAULT_BRAND = Brand(
    name="cBioPortal",
    short_name="cBioPortal",
    title="cBioPortal Cell Explorer",
    tagline="Explore millions of cells in your browser.",
    logo_href=None,
    logo=BrandLogo(on_light=None, on_dark=None, alt="cBioPortal"),
    colors=BrandColors(ink="#0d2c48", ink_deep=None, theme_color="#123a5e"),
    favicon=BrandFavicon(ico=None, svg=None, png192=None, png512=None, apple_touch=None),
)


def _section(raw: dict, key: str, warn: Callable[[str], None]) -> dict:
    """Return raw[key] when it is a mapping, else warn and return empty."""
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        warn(f"brand.json: '{key}' must be an object, got {type(value).__name__}; ignoring")
        return {}
    return value


def _text(raw: dict, key: str, warn: Callable[[str], None]) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        warn(f"brand.json: '{key}' must be a string; ignoring")
        return None
    limit = _MAX_LEN[key]
    if len(value) > limit:
        warn(f"brand.json: '{key}' exceeds {limit} characters; ignoring")
        return None
    return value


def _color(raw: dict, key: str, warn: Callable[[str], None]) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not HEX_RE.match(value):
        warn(f"brand.json: colors.{key} must be #RRGGBB; ignoring")
        return None
    return value


def _asset(raw: dict, section: str, key: str, warn: Callable[[str], None]) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        warn(f"brand.json: {section}.{key} must be a string; ignoring")
        return None
    # A bare basename only. Anything with a separator or a parent reference is a
    # traversal attempt or a mistake; both are rejected the same way.
    if "/" in value or ".." in value or value in {"", "."}:
        warn(f"brand.json: {section}.{key} must be a bare filename; ignoring")
        return None
    if PurePosixPath(value).suffix.lower() not in ALLOWED_ASSET_SUFFIXES:
        warn(
            f"brand.json: {section}.{key} has a disallowed extension; "
            f"allowed: {', '.join(sorted(ALLOWED_ASSET_SUFFIXES))}; ignoring"
        )
        return None
    return value


def _href(raw: dict, key: str, warn: Callable[[str], None]) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        warn(f"brand.json: '{key}' must be a string; ignoring")
        return None
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        warn(f"brand.json: '{key}' must be an absolute http(s) URL; ignoring")
        return None
    return value


def parse_brand(raw: dict, *, warn: Callable[[str], None]) -> Brand:
    """Validate a raw brand.json mapping into a Brand, field by field.

    Invalid fields are dropped with a warning and take the cBioPortal default.
    """
    d = DEFAULT_BRAND

    name = _text(raw, "name", warn) or d.name
    short_name = _text(raw, "shortName", warn) or name
    title = _text(raw, "title", warn) or f"{name} Cell Explorer"
    tagline = _text(raw, "tagline", warn) or d.tagline
    logo_href = _href(raw, "logoHref", warn)

    logo_raw = _section(raw, "logo", warn)
    logo = BrandLogo(
        on_light=_asset(logo_raw, "logo", "onLight", warn),
        on_dark=_asset(logo_raw, "logo", "onDark", warn),
        alt=_text(logo_raw, "alt", warn) or name,
    )

    colors_raw = _section(raw, "colors", warn)
    ink = _color(colors_raw, "ink", warn)
    if ink is not None and relative_luminance(ink) > MAX_INK_LUMINANCE:
        warn(
            f"brand.json: colors.ink {ink} has relative luminance above "
            f"{MAX_INK_LUMINANCE}; it is a dark band behind light text; ignoring"
        )
        ink = None
    theme_color = _color(colors_raw, "themeColor", warn)
    colors = BrandColors(
        ink=ink or d.colors.ink,
        ink_deep=_color(colors_raw, "inkDeep", warn),
        theme_color=theme_color or ink or d.colors.theme_color,
    )

    favicon_raw = _section(raw, "favicon", warn)
    favicon = BrandFavicon(
        ico=_asset(favicon_raw, "favicon", "ico", warn),
        svg=_asset(favicon_raw, "favicon", "svg", warn),
        png192=_asset(favicon_raw, "favicon", "png192", warn),
        png512=_asset(favicon_raw, "favicon", "png512", warn),
        apple_touch=_asset(favicon_raw, "favicon", "appleTouch", warn),
    )

    return replace(
        d,
        name=name,
        short_name=short_name,
        title=title,
        tagline=tagline,
        logo_href=logo_href,
        logo=logo,
        colors=colors,
        favicon=favicon,
    )


logger = logging.getLogger(__name__)


def _existing_asset(brand_dir: Path, filename: str | None) -> str | None:
    """Keep a validated filename only when it is a real file in the bundle."""
    if filename is None:
        return None
    if not (brand_dir / filename).is_file():
        logger.warning(
            "Brand asset %r referenced by brand.json is not a file in %s; ignoring",
            filename,
            brand_dir,
        )
        return None
    return filename


def load_brand(brand_dir: Path | None) -> Brand:
    """Load and validate a brand bundle. Never raises; defaults on any failure."""
    if brand_dir is None:
        return DEFAULT_BRAND

    try:
        is_dir = brand_dir.is_dir()
    except OSError as exc:
        logger.warning("Could not stat BRAND_DIR %s (%s); using default branding", brand_dir, exc)
        return DEFAULT_BRAND
    if not is_dir:
        logger.warning("BRAND_DIR %s is not a directory; using default branding", brand_dir)
        return DEFAULT_BRAND

    config_path = brand_dir / "brand.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("No brand.json in %s; using default branding", brand_dir)
        return DEFAULT_BRAND
    except (OSError, ValueError) as exc:
        logger.warning("Could not read brand.json in %s (%s); using default branding", brand_dir, exc)
        return DEFAULT_BRAND

    if not isinstance(raw, dict):
        logger.warning("brand.json in %s is not a JSON object; using default branding", brand_dir)
        return DEFAULT_BRAND

    brand = parse_brand(raw, warn=logger.warning)

    # Filenames passed validation; now confirm they exist on disk.
    logo = BrandLogo(
        on_light=_existing_asset(brand_dir, brand.logo.on_light),
        on_dark=_existing_asset(brand_dir, brand.logo.on_dark),
        alt=brand.logo.alt,
    )
    favicon = BrandFavicon(
        ico=_existing_asset(brand_dir, brand.favicon.ico),
        svg=_existing_asset(brand_dir, brand.favicon.svg),
        png192=_existing_asset(brand_dir, brand.favicon.png192),
        png512=_existing_asset(brand_dir, brand.favicon.png512),
        apple_touch=_existing_asset(brand_dir, brand.favicon.apple_touch),
    )
    return replace(brand, logo=logo, favicon=favicon)
