"""plot_bar — grouped bar charts comparing a few groups of an obs column."""

import numpy as np
import pytest

from cell_explorer_agent.tools.data.plot import plot_bar_tool
from cell_explorer_agent.tools.zarr_protocol import ObsColumn

from tests.fakes.fake_zarr import FakeZarrAccess


def _cat(name: str, labels: list[str]) -> ObsColumn:
    cats = sorted(set(labels))
    return ObsColumn(
        name=name,
        dtype="categorical",
        values=np.array([cats.index(v) for v in labels]),
        categories=cats,
    )


@pytest.fixture
def sites_zarr() -> FakeZarrAccess:
    # 8 cells: 4 in Adnexa, 4 in Omentum.
    site = ["Adnexa"] * 4 + ["Omentum"] * 4
    cell_type = ["T", "T", "B", "F", "T", "B", "B", "F"]
    sort = ["+", "+", "+", "-", "+", "+", "-", "-"]
    cd8a = np.array([4, 2, 0, 0, 6, 0, 0, 0], dtype="float32")
    return FakeZarrAccess(
        n_obs=8,
        n_var=2,
        obs={
            "site": _cat("site", site),
            "cell_type": _cat("cell_type", cell_type),
            "sort": _cat("sort", sort),
            "n_counts": ObsColumn(
                name="n_counts", dtype="numeric", values=np.arange(8.0),
            ),
        },
        var=["CD8A", "CD4"],
        expression={"CD8A": cd8a, "CD4": np.ones(8, dtype="float32")},
    )


def _tool(z):
    return plot_bar_tool(z, limit_bytes=32_768, concurrency=4)


def _values(data) -> dict[str, list[float]]:
    return dict(zip(data["categories"], data["values"]))


async def test_composition_fractions_per_group(sites_zarr):
    result = await _tool(sites_zarr).func(
        measure="composition", obs_column="site",
        groups=["Adnexa", "Omentum"], of="cell_type",
    )
    data = result["data"]
    assert data["series"] == ["Adnexa", "Omentum"]
    assert data["value_kind"] == "fraction"
    assert data["n_cells"] == [4, 4]
    assert _values(data) == {
        "T": [0.5, 0.25],
        "B": [0.25, 0.5],
        "F": [0.25, 0.25],
    }
    assert result["chart"] == {"type": "grouped_bar", "data": data}


async def test_composition_where_restricts_cells(sites_zarr):
    result = await _tool(sites_zarr).func(
        measure="composition", obs_column="site",
        groups=["Adnexa", "Omentum"], of="cell_type", where={"sort": "+"},
    )
    data = result["data"]
    assert data["where"] == {"sort": "+"}
    assert data["n_cells"] == [3, 2]
    # F has no CD45+ cells in either group, so it is dropped.
    values = _values(data)
    assert set(values) == {"T", "B"}
    assert values["T"] == pytest.approx([2 / 3, 0.5])
    assert values["B"] == pytest.approx([1 / 3, 0.5])


async def test_composition_merges_tail_into_other():
    n = 15
    labels = [f"c{i:02d}" for i in range(n)]
    z = FakeZarrAccess(
        n_obs=n,
        obs={"g": _cat("g", ["x"] * n), "kind": _cat("kind", labels)},
    )
    result = await _tool(z).func(
        measure="composition", obs_column="g", groups=["x"], of="kind",
    )
    data = result["data"]
    assert len(data["categories"]) == 11
    assert data["categories"][-1] == "All other"
    assert data["values"][-1] == [pytest.approx(5 / n)]


async def test_cell_counts_one_bar_per_group(sites_zarr):
    result = await _tool(sites_zarr).func(
        measure="cell_counts", obs_column="site",
        groups=["Omentum", "Adnexa"], where={"sort": "+"},
    )
    data = result["data"]
    assert data["value_kind"] == "count"
    assert data["categories"] == ["Omentum", "Adnexa"]
    assert data["series"] == ["cells"]
    assert data["values"] == [[2], [3]]


async def test_gene_mean_per_gene_per_group(sites_zarr):
    result = await _tool(sites_zarr).func(
        measure="gene_mean", obs_column="site",
        groups=["Adnexa", "Omentum"], genes=["CD8A"],
    )
    data = result["data"]
    assert data["value_kind"] == "mean_expression"
    assert data["categories"] == ["CD8A"]
    assert data["values"] == [[1.5, 1.5]]
    assert data["fraction_expressing"] == [[0.5, 0.25]]


async def test_gene_mean_with_where(sites_zarr):
    result = await _tool(sites_zarr).func(
        measure="gene_mean", obs_column="site",
        groups=["Adnexa", "Omentum"], genes=["CD8A"], where={"cell_type": "T"},
    )
    assert result["data"]["values"] == [[3.0, 6.0]]


async def test_duplicate_groups_are_collapsed(sites_zarr):
    result = await _tool(sites_zarr).func(
        measure="cell_counts", obs_column="site", groups=["Adnexa", "Adnexa"],
    )
    assert result["data"]["categories"] == ["Adnexa"]


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"measure": "violin", "obs_column": "site", "groups": ["Adnexa"]}, "measure"),
        ({"measure": "cell_counts", "obs_column": "nope", "groups": ["Adnexa"]}, "nope"),
        ({"measure": "cell_counts", "obs_column": "n_counts", "groups": ["1"]}, "categorical"),
        ({"measure": "cell_counts", "obs_column": "site", "groups": []}, "groups"),
        ({"measure": "cell_counts", "obs_column": "site", "groups": ["Ovary"]}, "Ovary"),
        ({"measure": "composition", "obs_column": "site", "groups": ["Adnexa"]}, "of"),
        (
            {"measure": "composition", "obs_column": "site", "groups": ["Adnexa"], "of": "site"},
            "of",
        ),
        ({"measure": "gene_mean", "obs_column": "site", "groups": ["Adnexa"]}, "genes"),
        (
            {"measure": "gene_mean", "obs_column": "site", "groups": ["Adnexa"], "genes": ["XYZ"]},
            "XYZ",
        ),
        (
            {"measure": "cell_counts", "obs_column": "site", "groups": ["Adnexa"],
             "where": {"sort": "maybe"}},
            "maybe",
        ),
        (
            {"measure": "cell_counts", "obs_column": "site", "groups": ["Adnexa"],
             "where": {"n_counts": "1"}},
            "categorical",
        ),
    ],
)
async def test_invalid_args_return_error(sites_zarr, kwargs, fragment):
    result = await _tool(sites_zarr).func(**kwargs)
    assert "error" in result
    assert fragment in result["error"]


async def test_too_many_groups_errors(sites_zarr):
    result = await _tool(sites_zarr).func(
        measure="cell_counts", obs_column="site", groups=[f"g{i}" for i in range(9)],
    )
    assert "error" in result


def test_plot_bar_is_data_tool(sites_zarr):
    tool = _tool(sites_zarr)
    assert tool.name == "plot_bar"
    assert tool.kind == "data"
