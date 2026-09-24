import pytest

from cell_explorer_api.branding import (
    DEFAULT_BRAND,
    parse_brand,
    relative_luminance,
)


def _parse(raw):
    """Parse, collecting warnings so tests can assert on them."""
    warnings: list[str] = []
    brand = parse_brand(raw, warn=warnings.append)
    return brand, warnings


def test_empty_config_yields_defaults():
    brand, warnings = _parse({})
    assert brand == DEFAULT_BRAND
    assert warnings == []


def test_single_field_config_is_valid():
    # The spec's incremental-adoption guarantee: one field is enough.
    brand, warnings = _parse({"name": "Break Through Cancer"})
    assert brand.name == "Break Through Cancer"
    assert brand.tagline == DEFAULT_BRAND.tagline
    assert brand.colors.ink == DEFAULT_BRAND.colors.ink
    assert warnings == []


def test_title_derives_from_name_when_absent():
    brand, _ = _parse({"name": "Break Through Cancer"})
    assert brand.title == "Break Through Cancer Cell Explorer"


def test_explicit_title_wins():
    brand, _ = _parse({"name": "BTC", "title": "BTC — Cell Explorer"})
    assert brand.title == "BTC — Cell Explorer"


def test_short_name_falls_back_to_name():
    brand, _ = _parse({"name": "Break Through Cancer"})
    assert brand.short_name == "Break Through Cancer"


def test_logo_alt_falls_back_to_name():
    brand, _ = _parse({"name": "Break Through Cancer"})
    assert brand.logo.alt == "Break Through Cancer"


def test_theme_color_falls_back_to_ink():
    brand, _ = _parse({"colors": {"ink": "#240d00"}})
    assert brand.colors.theme_color == "#240d00"


def test_valid_colors_accepted():
    brand, warnings = _parse({"colors": {"ink": "#240D00", "inkDeep": "#120600"}})
    assert brand.colors.ink == "#240D00"
    assert brand.colors.ink_deep == "#120600"
    assert warnings == []


def test_malformed_hex_rejected_and_defaults():
    brand, warnings = _parse({"colors": {"ink": "240D00"}})
    assert brand.colors.ink == DEFAULT_BRAND.colors.ink
    assert any("ink" in w for w in warnings)


def test_ink_too_light_rejected():
    brand, warnings = _parse({"colors": {"ink": "#DDDDDD"}})
    assert brand.colors.ink == DEFAULT_BRAND.colors.ink
    assert any("luminance" in w for w in warnings)


def test_dark_ink_passes_luminance_gate():
    brand, warnings = _parse({"colors": {"ink": "#240D00"}})
    assert brand.colors.ink == "#240D00"
    assert warnings == []


def test_relative_luminance_known_values():
    # approx, not ==: the coefficients sum to 1.0 only within float tolerance.
    assert relative_luminance("#000000") == pytest.approx(0.0)
    assert relative_luminance("#FFFFFF") == pytest.approx(1.0)
    assert relative_luminance("#0d2c48") < 0.35


def test_asset_filename_with_traversal_rejected():
    brand, warnings = _parse({"logo": {"onDark": "../../etc/passwd"}})
    assert brand.logo.on_dark is None
    assert any("onDark" in w for w in warnings)


def test_asset_filename_with_subdirectory_rejected():
    brand, warnings = _parse({"logo": {"onDark": "nested/logo.svg"}})
    assert brand.logo.on_dark is None
    assert any("onDark" in w for w in warnings)


def test_disallowed_asset_extension_rejected():
    brand, warnings = _parse({"logo": {"onDark": "logo.exe"}})
    assert brand.logo.on_dark is None
    assert any("onDark" in w for w in warnings)


def test_allowed_asset_extension_accepted():
    brand, warnings = _parse({"logo": {"onDark": "logo-white.svg"}})
    assert brand.logo.on_dark == "logo-white.svg"
    assert warnings == []


def test_javascript_logo_href_rejected():
    brand, warnings = _parse({"logoHref": "javascript:alert(1)"})
    assert brand.logo_href is None
    assert any("logoHref" in w for w in warnings)


def test_relative_logo_href_rejected():
    brand, warnings = _parse({"logoHref": "/somewhere"})
    assert brand.logo_href is None
    assert any("logoHref" in w for w in warnings)


def test_https_logo_href_accepted():
    brand, warnings = _parse({"logoHref": "https://breakthroughcancer.org"})
    assert brand.logo_href == "https://breakthroughcancer.org"
    assert warnings == []


def test_overlong_text_rejected():
    brand, warnings = _parse({"name": "x" * 121})
    assert brand.name == DEFAULT_BRAND.name
    assert any("name" in w for w in warnings)


def test_unknown_keys_ignored():
    brand, warnings = _parse({"name": "BTC", "futureField": {"a": 1}})
    assert brand.name == "BTC"
    assert warnings == []


def test_one_bad_field_does_not_sink_the_others():
    brand, warnings = _parse(
        {"name": "Break Through Cancer", "colors": {"ink": "not-a-color"}}
    )
    assert brand.name == "Break Through Cancer"
    assert brand.colors.ink == DEFAULT_BRAND.colors.ink
    assert len(warnings) == 1


def test_non_dict_nested_value_does_not_raise():
    brand, warnings = _parse({"colors": "navy"})
    assert brand.colors.ink == DEFAULT_BRAND.colors.ink
    assert any("colors" in w for w in warnings)


def test_asset_filename_with_backslash_traversal_rejected():
    brand, warnings = _parse({"logo": {"onDark": "..\\..\\evil.svg"}})
    assert brand.logo.on_dark is None
    assert any("onDark" in w for w in warnings)


def test_valid_favicon_bundle_parses():
    brand, warnings = _parse(
        {
            "favicon": {
                "ico": "favicon.ico",
                "svg": "favicon.svg",
                "png192": "favicon-192.png",
                "png512": "favicon-512.png",
                "appleTouch": "apple-touch-icon.png",
            }
        }
    )
    assert brand.favicon.ico == "favicon.ico"
    assert brand.favicon.svg == "favicon.svg"
    assert brand.favicon.png192 == "favicon-192.png"
    assert brand.favicon.png512 == "favicon-512.png"
    assert brand.favicon.apple_touch == "apple-touch-icon.png"
    assert warnings == []


def test_invalid_favicon_filename_rejected():
    brand, warnings = _parse({"favicon": {"ico": "evil/../../../etc/passwd.ico"}})
    assert brand.favicon.ico is None
    assert any("ico" in w for w in warnings)


# --- Filename length: a byte limit, not a character limit ------------------


def test_overlong_asset_filename_rejected():
    """Unbounded filenames reach a stat the OS refuses to answer (ENAMETOOLONG)."""
    brand, warnings = _parse({"logo": {"onDark": "a" * 252 + ".svg"}})  # 256 bytes
    assert brand.logo.on_dark is None
    assert any("onDark" in w and "255 bytes" in w for w in warnings)


def test_asset_filename_at_the_limit_accepted():
    name = "a" * 251 + ".svg"  # 255 bytes exactly
    assert len(name.encode("utf-8")) == 255
    brand, warnings = _parse({"logo": {"onDark": name}})
    assert brand.logo.on_dark == name
    assert warnings == []


def test_asset_filename_cap_counts_bytes_not_characters():
    """The sneaky case: comfortably under 256 characters, over 255 bytes.

    Each 'é' costs two bytes in UTF-8, so a 200-character name is 396 bytes and
    a character-based cap would wave it through into the raising stat.
    """
    name = "é" * 196 + ".svg"  # 200 characters, 396 bytes
    assert len(name) < 256 and len(name.encode("utf-8")) > 255

    brand, warnings = _parse({"logo": {"onDark": name}})
    assert brand.logo.on_dark is None
    assert any("onDark" in w and "255 bytes" in w for w in warnings)
