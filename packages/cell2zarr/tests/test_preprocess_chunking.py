"""Missing-value handling in the preprocess_chunking script."""
import numpy as np
import pandas as pd

from cell2zarr.preprocess_chunking import _as_string_column


def test_missing_values_become_unknown():
    col = pd.Series(["a", None, np.nan, "nan", "None", "b"])
    assert _as_string_column(col).tolist() == ["a", "unknown", "unknown", "unknown", "unknown", "b"]


def test_categorical_missing_values_become_unknown():
    col = pd.Series(pd.Categorical(["a", None, "b"]))
    assert _as_string_column(col).tolist() == ["a", "unknown", "b"]


def test_numeric_column_is_stringified():
    assert _as_string_column(pd.Series([1.5, np.nan])).tolist() == ["1.5", "unknown"]
