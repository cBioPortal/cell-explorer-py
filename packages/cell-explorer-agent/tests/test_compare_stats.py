"""Unit tests for the shared compute_stats helper."""

import numpy as np

from cell_explorer_agent.tools.data.compare_stats import compute_stats


def test_compute_stats_known_values():
    """Hand-computed scenario: 5 cells per group, 3 genes.

    gene1: A=[1,2,3,4,5], B=[3,4,5,6,7]  -> means 3 vs 5, vars 2.5 vs 2.5
    gene2: A=[10]*5,      B=[10]*5       -> means equal, vars both 0 (NaN d/t)
    gene3: A=[-1,0,1,2,3], B=[1,2,3,4,5] -> means 1 vs 3, vars 2.5 vs 2.5
    """
    n_a = 5
    n_b = 5
    # gene order: [gene1, gene2, gene3]
    sum_x_a = np.array([15.0, 50.0, 5.0])
    sum_xx_a = np.array([55.0, 500.0, 15.0])
    sum_x_b = np.array([25.0, 50.0, 15.0])
    sum_xx_b = np.array([135.0, 500.0, 55.0])

    out = compute_stats(sum_x_a, sum_xx_a, n_a, sum_x_b, sum_xx_b, n_b)

    np.testing.assert_allclose(out["mean_a"], [3.0, 10.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(out["mean_b"], [5.0, 10.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(out["var_a"], [2.5, 0.0, 2.5], atol=1e-9)
    np.testing.assert_allclose(out["var_b"], [2.5, 0.0, 2.5], atol=1e-9)

    # gene1 and gene3 share the same |d| and |t| by construction.
    expected_d = -2.0 / np.sqrt(2.5)  # ~ -1.2649
    expected_t = -2.0  # (mean diff) / sqrt(0.5+0.5)
    np.testing.assert_allclose(out["cohens_d"][0], expected_d, atol=1e-9)
    np.testing.assert_allclose(out["cohens_d"][2], expected_d, atol=1e-9)
    np.testing.assert_allclose(out["t_statistic"][0], expected_t, atol=1e-9)
    np.testing.assert_allclose(out["t_statistic"][2], expected_t, atol=1e-9)

    # gene2: zero variance, equal means -> NaN for both d and t
    assert np.isnan(out["cohens_d"][1])
    assert np.isnan(out["t_statistic"][1])


def test_compute_stats_inf_when_zero_variance_unequal_means():
    """Zero variance in both groups + unequal means -> ±Inf, not NaN."""
    sum_x_a = np.array([0.0])
    sum_xx_a = np.array([0.0])
    sum_x_b = np.array([10.0])  # 5 cells each constant at 2.0
    sum_xx_b = np.array([20.0])

    out = compute_stats(sum_x_a, sum_xx_a, 5, sum_x_b, sum_xx_b, 5)

    assert np.isinf(out["cohens_d"][0])
    assert np.isinf(out["t_statistic"][0])


def test_compute_stats_clips_negative_variance_from_float_noise():
    """sum_xx - sum_x**2/n can produce tiny negative values from float
    cancellation when stored sums are large and means are large vs variance.
    Must clip to 0 so sqrt(var) doesn't produce NaN downstream.
    """
    # n=100 cells each at mean=100_000: sums are large enough that storing
    # them in float32 causes catastrophic cancellation in sum_xx - sum_x**2/n.
    n = 100
    mean = 100_000.0
    sum_x = np.array([n * mean], dtype="float32")    # 10_000_000
    sum_xx = np.array([n * mean**2], dtype="float32")  # 1e12, truncated in float32

    # Verify this test setup actually triggers the cancellation: compute the
    # raw (unclipped) sample variance in float64 from the float32 sums and
    # confirm it is negative before the np.maximum clamp fires.
    raw = (sum_xx.astype("float64") - sum_x.astype("float64") ** 2 / n) / (n - 1)
    assert raw[0] < 0, f"test setup didn't trigger cancellation, raw={raw[0]}"

    out = compute_stats(sum_x, sum_xx, n, sum_x, sum_xx, n)

    assert out["var_a"][0] >= 0.0
    assert out["var_b"][0] >= 0.0
    assert not np.isnan(out["var_a"][0])
    assert not np.isnan(out["var_b"][0])


def test_compute_stats_accepts_float32_sums_and_returns_float64():
    """Strata sums are float32; the helper must cast for compute precision."""
    sum_x_a = np.array([15.0], dtype="float32")
    sum_xx_a = np.array([55.0], dtype="float32")
    sum_x_b = np.array([25.0], dtype="float32")
    sum_xx_b = np.array([135.0], dtype="float32")

    out = compute_stats(sum_x_a, sum_xx_a, 5, sum_x_b, sum_xx_b, 5)

    for key in ("mean_a", "mean_b", "var_a", "var_b", "cohens_d", "t_statistic"):
        assert out[key].dtype == np.float64, key
