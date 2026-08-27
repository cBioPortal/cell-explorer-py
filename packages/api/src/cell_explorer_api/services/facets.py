"""Canonical facets, and the mapping from physical column names onto them.

Datasets do not share a schema. Today every dataset in the catalogue comes from
CELLxGENE/HTAN and so agrees on names, but that is an accident of the current
catalogue, not a property of the system. A dataset calling it `organ` must join
the same Tissue facet rather than splitting the catalogue into two.

These definitions are applied when a response is built, never when data is
stored, so correcting a mapping takes effect on the next request instead of
requiring every dataset to be re-harvested.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FacetDefinition:
    key: str                  # canonical identity, e.g. "tissue"
    label: str                # display, e.g. "Tissue"
    order: int                # sidebar ordering
    columns: tuple[str, ...]  # candidate column names, in precedence order


# The canonical keys and each tuple's first entry are the column names measured
# present in 100% of the current catalogue. The remaining candidates are
# plausible alternatives, not observed ones — no dataset uses them today. They
# are starting guesses; GET /api/admin/facets/unmapped is how real alternatives
# get discovered rather than imagined.
FACET_DEFINITIONS: tuple[FacetDefinition, ...] = (
    FacetDefinition("tissue", "Tissue", 10, ("tissue", "organ", "tissue_site")),
    FacetDefinition("cell_type", "Cell type", 20, ("cell_type", "celltype", "author_cell_type")),
    FacetDefinition("disease", "Disease", 30, ("disease", "condition")),
    FacetDefinition("assay", "Assay", 40, ("assay",)),
    FacetDefinition("sex", "Sex", 50, ("sex",)),
    FacetDefinition("development_stage", "Development stage", 60, ("development_stage",)),
    FacetDefinition("donor", "Donor", 70, ("donor_id", "patient", "subject_id")),
    FacetDefinition("suspension_type", "Suspension", 80, ("suspension_type",)),
)


def resolve_facet_with(
    definitions: tuple[FacetDefinition, ...], column_name: str
) -> str | None:
    """Canonical facet key for a column name, or None when unmapped.

    Matching is exact and case-sensitive on purpose: loose matching would map
    `tissue_ontology_term_id` onto Tissue and serve UBERON ids as tissue labels.
    """
    for definition in definitions:
        if column_name in definition.columns:
            return definition.key
    return None


def resolve_facet(column_name: str) -> str | None:
    """Resolve against the module's definitions."""
    return resolve_facet_with(FACET_DEFINITIONS, column_name)
