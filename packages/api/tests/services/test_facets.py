"""Canonical facet definitions and column-name resolution."""

from cell_explorer_api.services.facets import (
    FACET_DEFINITIONS,
    FacetDefinition,
    resolve_facet,
    resolve_facet_with,
)


def test_resolves_the_canonical_name():
    assert resolve_facet("tissue") == "tissue"


def test_resolves_an_alias_to_its_canonical_key():
    # The whole point: a dataset calling it `organ` joins the same Tissue facet.
    assert resolve_facet("organ") == "tissue"


def test_unmapped_column_resolves_to_none():
    assert resolve_facet("seurat_clusters") is None


def test_resolution_is_case_sensitive_and_exact():
    # Loose matching would silently map `tissue_ontology_term_id` onto Tissue.
    assert resolve_facet("tissue_ontology_term_id") is None


def test_definitions_have_unique_keys_and_no_shared_columns():
    keys = [d.key for d in FACET_DEFINITIONS]
    assert len(keys) == len(set(keys)), "duplicate facet key"
    seen: dict[str, str] = {}
    for d in FACET_DEFINITIONS:
        for col in d.columns:
            assert col not in seen, f"{col!r} claimed by both {seen.get(col)!r} and {d.key!r}"
            seen[col] = d.key


def test_precedence_when_a_dataset_carries_two_candidates():
    # Order within `columns` decides; both resolve, and the caller picks by order.
    defs = (FacetDefinition("tissue", "Tissue", 10, ("tissue", "organ")),)
    assert resolve_facet_with(defs, "tissue") == "tissue"
    assert resolve_facet_with(defs, "organ") == "tissue"
