import numpy as np
import pytest

from cell_explorer_mcp import queries
from cell_explorer_mcp.queries import QueryError


def _dense(adata):
    return np.asarray(adata.X, dtype="float64")


async def test_search_genes_ranks_exact_then_prefix(pbmc):
    result = await queries.search_genes(pbmc, "cd79", limit=10)
    assert {m["symbol"] for m in result["matches"]} == {"CD79A", "CD79B"}

    ranked = await queries.search_genes(pbmc, "cd7")
    symbols = [m["symbol"] for m in ranked["matches"]]
    assert symbols[0] == "CD72"
    assert set(symbols[1:3]) == {"CD79A", "CD79B"}

    exact = await queries.search_genes(pbmc, "nkg7")
    assert exact["matches"][0]["symbol"] == "NKG7"


async def test_search_genes_rejects_empty_query(pbmc):
    with pytest.raises(QueryError):
        await queries.search_genes(pbmc, "  ")


async def test_gene_expression_strata_matches_x(strata_tiny, strata_tiny_adata):
    X = _dense(strata_tiny_adata)
    cell_type = strata_tiny_adata.obs["cell_type"].to_numpy()

    result = await queries.gene_expression(strata_tiny, "gene3", group_by="cell_type")

    assert result["source"] == "strata"
    assert result["overall"]["n_cells"] == 50
    assert result["overall"]["mean"] == pytest.approx(X[:, 2].mean(), abs=1e-4)
    by_group = {g["group"]: g for g in result["groups"]}
    for group in ("A", "B", "C"):
        col = X[cell_type == group, 2]
        assert by_group[group]["n_cells"] == col.size
        assert by_group[group]["mean"] == pytest.approx(col.mean(), abs=1e-4)
        assert by_group[group]["fraction_expressing"] == pytest.approx((col > 0).mean(), abs=1e-4)
    means = [g["mean"] for g in result["groups"]]
    assert means == sorted(means, reverse=True)


async def test_gene_expression_cells_path_for_column_without_strata(strata_tiny, strata_tiny_adata):
    X = _dense(strata_tiny_adata)
    donor = strata_tiny_adata.obs["donor"].to_numpy()

    result = await queries.gene_expression(strata_tiny, "GENE4", group_by="donor")

    assert result["gene"] == "gene4"
    assert result["source"] == "cells"
    by_group = {g["group"]: g for g in result["groups"]}
    for group in ("d1", "d2"):
        col = X[donor == group, 3]
        assert by_group[group]["mean"] == pytest.approx(col.mean(), abs=1e-4)
        assert by_group[group]["fraction_expressing"] == pytest.approx((col > 0).mean(), abs=1e-4)


async def test_gene_expression_unknown_gene_suggests(pbmc):
    with pytest.raises(QueryError, match=r"Similar: CD79[AB], CD79[AB]"):
        await queries.gene_expression(pbmc, "CD79X")


async def test_gene_expression_rejects_non_categorical_group_by(pbmc):
    with pytest.raises(QueryError, match="not categorical"):
        await queries.gene_expression(pbmc, "CD79A", group_by="n_counts")


async def test_gene_column_read_is_capped(pbmc, monkeypatch):
    monkeypatch.setattr(queries, "MAX_X_COLUMN_CELLS", 100)
    with pytest.raises(QueryError, match="per-cell reads are limited"):
        await queries.gene_expression(pbmc, "CD79A", group_by="louvain")


async def test_top_genes_strata_matches_x(strata_tiny, strata_tiny_adata):
    X = _dense(strata_tiny_adata)
    mask = strata_tiny_adata.obs["cell_type"].to_numpy() == "B"
    expected = X[mask].mean(axis=0)

    result = await queries.top_genes(strata_tiny, "cell_type", "B", n=3)

    assert result["source"] == "strata"
    assert result["n_cells"] == int(mask.sum())
    got = [g["mean"] for g in result["genes"]]
    assert got == pytest.approx(sorted(expected, reverse=True)[:3], abs=1e-4)


async def test_top_genes_x_scan_without_strata(pbmc, pbmc_adata):
    X = _dense(pbmc_adata)
    mask = pbmc_adata.obs["louvain"].to_numpy() == "B cells"
    means = X[mask].mean(axis=0)
    expected = [str(pbmc_adata.var_names[i]) for i in np.argsort(-means, kind="stable")[:5]]

    result = await queries.top_genes(pbmc, "louvain", "B cells", n=5)

    assert result["source"] == "cells"
    assert [g["symbol"] for g in result["genes"]] == expected


async def test_top_genes_x_scan_is_capped(pbmc, monkeypatch):
    monkeypatch.setattr(queries, "MAX_X_SCAN_ELEMENTS", 1000)
    with pytest.raises(QueryError, match="exceeds this server's limit"):
        await queries.top_genes(pbmc, "louvain", "B cells")


async def test_top_genes_unknown_group(strata_tiny):
    with pytest.raises(QueryError, match="not in 'cell_type'"):
        await queries.top_genes(strata_tiny, "cell_type", "Z")


async def test_obs_column_values_categorical(pbmc, pbmc_adata):
    result = await queries.obs_column_values(pbmc, "louvain", limit=3)
    counts = pbmc_adata.obs["louvain"].value_counts()
    assert result["dtype"] == "categorical"
    assert result["n_values"] == len(counts)
    assert [v["count"] for v in result["values"]] == counts.iloc[:3].tolist()
    assert result["other_count"] == int(counts.iloc[3:].sum())


async def test_obs_column_values_numeric(pbmc, pbmc_adata):
    result = await queries.obs_column_values(pbmc, "n_genes")
    assert result["dtype"] == "numeric"
    assert result["stats"]["max"] == pbmc_adata.obs["n_genes"].max()


async def test_obs_column_values_unknown_column(pbmc):
    with pytest.raises(QueryError, match="Available: "):
        await queries.obs_column_values(pbmc, "nope")


async def test_dataset_info(strata_tiny):
    entry = {
        "name": "Strata tiny",
        "metadata": {"obs_columns": [{"name": "cell_type", "dtype": "categorical", "cardinality": 3, "values": ["A", "B", "C"]}]},
    }
    info = await queries.dataset_info(strata_tiny, entry)
    assert (info["n_cells"], info["n_genes"]) == (50, 10)
    assert info["fast_group_by_columns"] == ["cell_type"]
    cell_type = next(c for c in info["obs_columns"] if c["name"] == "cell_type")
    assert cell_type["categories"] == ["A", "B", "C"]
