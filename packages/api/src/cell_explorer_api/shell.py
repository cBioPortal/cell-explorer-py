"""Brand substitution into the static HTML shell.

`index.html` and `site.webmanifest` are served verbatim to the browser before
any JavaScript runs, so runtime config cannot reach the tab title or favicon.
These are substituted server-side instead.

Elements are located by anchored patterns rather than a full HTML parse:
index.html carries an inline SPA-redirect script that reserializing would risk
mangling, and the file is one we author and whose shape these tests assert.
Safety comes from escaping the injected values, which happens either way.
"""

from __future__ import annotations

import json
import logging
import re
from html import escape
from urllib.parse import quote

from cell_explorer_api.branding import DEFAULT_BRAND, Brand

logger = logging.getLogger(__name__)

_TITLE_RE = re.compile(r"<title>.*?</title>", re.DOTALL)
_THEME_RE = re.compile(r'(<meta\s+name="theme-color"\s+content=")[^"]*(")')


def _asset_url(filename: str | None) -> str | None:
    # Operator filenames are validated as bare names, not as URL-safe ones: a
    # `#`, `?`, `%`, space or backslash in one would otherwise build a URL the
    # browser reparses into something that 404s silently.
    return f"/brand/{quote(filename, safe='')}" if filename else None


def _sub_once(pattern: re.Pattern[str], repl, html: str, *, anchor: str) -> str:
    """Substitute once, warning when the anchor found nothing.

    The anchors hardcode attribute order in a file built in another repo, so a
    build-tool change could silently stop branding one element. A miss leaves
    the shell's built-in value in place — degraded, never broken — but the
    operator gets told which anchor stopped matching.
    """
    out, count = pattern.subn(repl, html, count=1)
    if count == 0:
        logger.warning(
            "Brand substitution found no match for %r in index.html; that element "
            "keeps its built-in value. The shell's markup may have changed shape.",
            anchor,
        )
    return out


def _replace_link_href(html: str, anchor: str, url: str | None) -> str:
    """Point one <link> at a brand asset, leaving it alone when unset.

    `anchor` is the literal head of the tag, far enough in to tell it apart
    from the other <link>s. It may stop at the distinguishing attribute, in
    which case href is matched further along the same tag, or run right up to
    `href=` itself when that is what distinguishes the tag.
    """
    if url is None:
        return html
    head = re.escape(anchor)
    if not anchor.endswith("href="):
        head += r'[^>]*?href='
    pattern = re.compile(rf'({head}")[^"]*(")')
    # A lambda, not a template string: a replacement template would read
    # backslashes in an operator-supplied filename as regex escapes.
    value = escape(url, quote=True)
    return _sub_once(
        pattern, lambda m: f"{m.group(1)}{value}{m.group(2)}", html, anchor=anchor
    )


def render_index_html(html: str, brand: Brand) -> str:
    if brand == DEFAULT_BRAND:
        return html

    title = escape(brand.title)
    html = _sub_once(
        _TITLE_RE, lambda m: f"<title>{title}</title>", html, anchor="<title>"
    )
    html = _sub_once(
        _THEME_RE,
        lambda m: f"{m.group(1)}{escape(brand.colors.theme_color)}{m.group(2)}",
        html,
        anchor='<meta name="theme-color">',
    )
    html = _replace_link_href(html, '<link rel="icon" href=', _asset_url(brand.favicon.ico))
    html = _replace_link_href(
        html, '<link rel="icon" type="image/svg+xml"', _asset_url(brand.favicon.svg)
    )
    html = _replace_link_href(
        html, '<link rel="apple-touch-icon"', _asset_url(brand.favicon.apple_touch)
    )
    return html


def render_webmanifest(manifest_json: str, brand: Brand) -> str:
    """Substitute brand values into site.webmanifest.

    JSON, unlike the HTML shell, round-trips losslessly — so this parses.

    Nothing here may raise: this runs at startup, and a hand-edited manifest
    under STATIC_DIR must degrade to the file as written rather than stop the
    app from booting. Parsing is only the first shape check — valid JSON can
    still be a list, a string, or carry an `icons` that is not a list of
    objects.
    """
    if brand == DEFAULT_BRAND:
        return manifest_json

    try:
        manifest = json.loads(manifest_json)
    except ValueError:
        logger.warning("site.webmanifest is not valid JSON; serving it unbranded")
        return manifest_json

    if not isinstance(manifest, dict):
        logger.warning(
            "site.webmanifest is a JSON %s, not an object; serving it unbranded",
            type(manifest).__name__,
        )
        return manifest_json

    manifest["name"] = brand.title
    manifest["short_name"] = brand.short_name
    manifest["theme_color"] = brand.colors.theme_color
    manifest["background_color"] = brand.colors.ink

    by_size = {"192x192": brand.favicon.png192, "512x512": brand.favicon.png512}
    icons = manifest.get("icons")
    if icons is not None and not isinstance(icons, list):
        logger.warning("site.webmanifest 'icons' is not a list; leaving icons unbranded")
    for icon in icons if isinstance(icons, list) else []:
        if not isinstance(icon, dict):
            continue
        sizes = icon.get("sizes")
        replacement = _asset_url(by_size.get(sizes)) if isinstance(sizes, str) else None
        if replacement:
            icon["src"] = replacement

    return json.dumps(manifest, indent=2) + "\n"
