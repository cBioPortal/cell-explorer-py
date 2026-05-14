"""Smoke test: the strata submodule is importable."""


class TestStrataSubmodule:
    def test_strata_package_imports(self):
        from cell2zarr import strata  # noqa: F401

    def test_strata_package_re_exports_nothing_yet(self):
        from cell2zarr import strata
        # No public symbols (like build_strata) are exposed yet — those come
        # in later tasks. This guards against accidental premature exports.
        assert not hasattr(strata, "build_strata")
