"""Unit tests for econometric functions and pipeline helpers."""
import numpy as np
import pandas as pd
import pytest
from frequency_tails.data import log_change, assign_gepu_scenario
from frequency_tails.models import (stationary_indices, block_indices, hac_ols,
                                    nested_increments, normalized_generalized_fevd, adjust_pvalues)


def test_log_returns_do_not_fill_missing_prices():
    x = pd.Series([100., np.nan, 110., 121.])
    r = log_change(x)
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[1]) and np.isnan(r.iloc[2])
    assert np.isclose(r.iloc[3], 100*np.log(121/110))


def test_gepu_assignments_do_not_look_into_reference_month():
    months = pd.period_range('2020-01', periods=3, freq='M')
    changes = pd.Series([1., 2., 3.], index=months)
    dates = pd.bdate_range('2020-01-01', '2020-06-15')
    result = assign_gepu_scenario(changes, dates, delay_months=2)
    assert result.loc['2020-03-02'] == 1.
    assert result.loc['2020-04-01'] == 2.
    assert result.loc['2020-05-01'] == 3.
    assert result.loc['2020-01'].sum() == 0


def test_blocks_have_valid_indices():
    rng = np.random.default_rng(10)
    for fn in (stationary_indices, block_indices):
        v = fn(107, 20, rng)
        assert len(v) == 107 and np.all((v >= 0) & (v < 107))


def test_nested_models_have_common_sample_and_monotone_r2():
    rng = np.random.default_rng(13)
    n = 250
    data = pd.DataFrame(rng.normal(size=(n,4)), columns=['y','market','vix','gepu'])
    data.loc[9, 'gepu'] = np.nan
    result = nested_increments(data, 'y', 'market')
    assert set(result.nobs) == {249}
    assert result.loc[result.model=='market_both','r2'].iloc[0] >= result.loc[result.model=='market_only','r2'].iloc[0]


def test_fevd_rows_sum_to_100():
    rng = np.random.default_rng(42)
    n = 500
    x = pd.DataFrame(rng.normal(size=(n,3)), columns=['a','b','c'])
    matrix, directions, tci, model = normalized_generalized_fevd(x, list(x.columns), p=1, horizon=5)
    assert np.allclose(matrix.sum(axis=1), 100, atol=1e-8)
    assert 0 <= tci <= 100


def test_holm_corrections():
    table = pd.DataFrame({'family':['one','one','two'], 'p':[.01,.03,.04]})
    result = adjust_pvalues(table)
    assert np.isclose(result.loc[0, 'holm_p'], .02)
    assert np.isclose(result.loc[2, 'holm_p'], .04)
