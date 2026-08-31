"""Every setting must appear in the README's Configuration section.

The README is hand-written rather than generated, because the useful part of a
setting is rarely its type or default — it is that auth needs three values to
resolve together, or that REFRESH_COOKIE_MAX_AGE must not exceed the realm's
session lifespan. Prose carries that; a generated table cannot.

What generation *would* have given for free is the guarantee that nothing is
missing. This test supplies that guarantee on its own: add a field to Settings
without documenting it and this fails, naming the field.
"""

from pathlib import Path

from cell_explorer_api.config import Settings

README = Path(__file__).resolve().parents[3] / "README.md"


def _documented_names() -> str:
    text = README.read_text()
    marker = "## Configuration"
    assert marker in text, f"{README} has no '{marker}' section"
    return text[text.index(marker) :]


def test_every_setting_is_documented():
    section = _documented_names()
    missing = sorted(
        name.upper()
        for name in Settings.model_fields
        if f"`{name.upper()}`" not in section
    )
    assert not missing, (
        "Settings fields absent from the README's Configuration section: "
        + ", ".join(missing)
        + ". Add each as `NAME` in the appropriate table."
    )


def test_readme_documents_no_setting_that_no_longer_exists():
    # The other direction: a removed setting left behind in the docs is a
    # reader following instructions that do nothing.
    import re

    section = _documented_names()
    known = {name.upper() for name in Settings.model_fields}
    # Only table rows, so prose mentions of env vars from other systems
    # (KEYCLOAK realm settings, for instance) are not swept up.
    documented = {
        m.group(1)
        for m in re.finditer(r"^\| `([A-Z][A-Z0-9_]{2,})` \|", section, re.MULTILINE)
    }
    stale = sorted(documented - known)
    assert not stale, (
        "README documents settings that no longer exist on Settings: "
        + ", ".join(stale)
    )
