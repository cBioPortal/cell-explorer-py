"""Protocol-shape tests for ZarrAccess (currently exercised via FakeZarrAccess)."""


async def test_zarr_access_exposes_var_columns(fake_zarr):
    cols = await fake_zarr.var_columns()
    assert isinstance(cols, list)
    assert all(isinstance(c, str) for c in cols)
    # FakeZarrAccess.default() seeds [feature_id, gene_symbol]
    assert "gene_symbol" in cols
