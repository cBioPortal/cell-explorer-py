"""Smoke test: the strata submodule is importable."""


class TestStrataSubmodule:
    def test_strata_package_imports(self):
        from cell2zarr import strata  # noqa: F401

    def test_strata_package_re_exports_build_strata(self):
        from cell2zarr import strata
        # build_strata is the top-level orchestrator entry point.
        assert hasattr(strata, "build_strata")
        assert callable(strata.build_strata)
