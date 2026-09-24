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
import re
from html import escape

from cell_explorer_api.branding import DEFAULT_BRAND, Brand

_TITLE_RE = re.compile(r"<title>.*?</title>", re.DOTALL)
_THEME_RE = re.compile(r'(<meta\s+name="theme-color"\s+content=")[^"]*(")')


def _asset_url(filename: str | None) -> str | None:
    return f"/brand/{filename}" if filename else None


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
    return pattern.sub(lambda m: f"{m.group(1)}{value}{m.group(2)}", html, count=1)


def render_index_html(html: str, brand: Brand) -> str:
    if brand == DEFAULT_BRAND:
        return html

    title = escape(brand.title)
    html = _TITLE_RE.sub(lambda m: f"<title>{title}</title>", html, count=1)
    html = _THEME_RE.sub(
        lambda m: f"{m.group(1)}{escape(brand.colors.theme_color)}{m.group(2)}",
        html,
        count=1,
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
    """
    if brand == DEFAULT_BRAND:
        return manifest_json

    try:
        manifest = json.loads(manifest_json)
    except json.JSONDecodeError:
        return manifest_json

    manifest["name"] = brand.title
    manifest["short_name"] = brand.short_name
    manifest["theme_color"] = brand.colors.theme_color
    manifest["background_color"] = brand.colors.ink

    by_size = {"192x192": brand.favicon.png192, "512x512": brand.favicon.png512}
    for icon in manifest.get("icons", []):
        replacement = _asset_url(by_size.get(icon.get("sizes")))
        if replacement:
            icon["src"] = replacement

    return json.dumps(manifest, indent=2) + "\n"
