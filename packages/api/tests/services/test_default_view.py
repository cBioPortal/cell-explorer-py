"""Tests for default_view validation."""

import pytest

from cell_explorer_api.services.default_view import (
    DefaultViewError,
    validate_default_view,
)


def test_valid_category_view_passes():
    value = {"colorBy": "category", "category": "cell_type", "showCategoryLabels": True}
    assert validate_default_view(value) == value


def test_valid_gene_view_with_scale_passes():
    value = {"colorBy": "gene", "gene": "CD8A", "colorScaleName": "magma"}
    assert validate_default_view(value) == value


def test_rendering_only_view_passes():
    value = {"pointSize": 2.0, "opacity": 0.5}
    assert validate_default_view(value) == value


def test_empty_view_is_valid():
    assert validate_default_view({}) == {}


def test_show_category_labels_with_gene_mode_is_allowed():
    value = {"colorBy": "gene", "gene": "CD8A", "showCategoryLabels": True}
    assert validate_default_view(value) == value


def test_unknown_key_rejected():
    with pytest.raises(DefaultViewError):
        validate_default_view({"bogusField": 1})


def test_deferred_key_rejected():
    # `embedding` is a real AppConfig field but out of v1 scope.
    with pytest.raises(DefaultViewError):
        validate_default_view({"embedding": "X_umap"})


def test_bad_color_scale_rejected():
    with pytest.raises(DefaultViewError):
        validate_default_view({"colorBy": "gene", "gene": "CD8A", "colorScaleName": "rainbow"})


def test_opacity_out_of_range_rejected():
    with pytest.raises(DefaultViewError):
        validate_default_view({"opacity": 1.5})


def test_nonpositive_point_size_rejected():
    with pytest.raises(DefaultViewError):
        validate_default_view({"pointSize": 0})


def test_colorby_gene_without_gene_rejected():
    with pytest.raises(DefaultViewError):
        validate_default_view({"colorBy": "gene"})


def test_colorby_category_without_category_rejected():
    with pytest.raises(DefaultViewError):
        validate_default_view({"colorBy": "category"})


def test_highlighted_categories_requires_category_mode_rejected():
    with pytest.raises(DefaultViewError):
        validate_default_view({"colorBy": "gene", "gene": "CD8A", "highlightedCategories": ["T"]})


def test_highlighted_categories_in_category_mode_without_category_rejected():
    # colorBy='category' but no 'category' — rejected by the category-companion
    # rule, keeping validation symmetric with the frontend applyConfig checks.
    with pytest.raises(DefaultViewError):
        validate_default_view({"colorBy": "category", "highlightedCategories": ["T"]})
