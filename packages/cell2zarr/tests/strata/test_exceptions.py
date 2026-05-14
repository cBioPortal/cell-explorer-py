"""Tests for the strata exception hierarchy."""
import pytest

from cell2zarr.strata import (
    StrataError,
    StrataConfigError,
    StrataExistsError,
    StrataPreflightError,
    StrataBuildError,
    StrataWriteError,
)


class TestExceptionHierarchy:
    @pytest.mark.parametrize("cls", [
        StrataConfigError,
        StrataExistsError,
        StrataPreflightError,
        StrataBuildError,
        StrataWriteError,
    ])
    def test_subclass_of_base(self, cls):
        assert issubclass(cls, StrataError)

    def test_base_is_exception(self):
        assert issubclass(StrataError, Exception)

    def test_can_raise_and_catch_as_base(self):
        with pytest.raises(StrataError):
            raise StrataPreflightError("preflight failed")
