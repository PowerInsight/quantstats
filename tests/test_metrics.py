import pandas as pd
import quantstats.stats as stats
import numpy as np


def test_sharpe(meta: pd.DataFrame):
    # show sharpe ratio
    assert np.allclose(stats.sharpe(meta), 0.780269)


def test_win_rate(meta: pd.DataFrame):
    # show sharpe ratio
    assert np.allclose(stats.win_rate(meta), 0.5269761)


def test_cagr(meta: pd.DataFrame):
    # show sharpe ratio
    assert np.allclose(stats.cagr(meta), 0.172418375)


def test_sortino(meta: pd.DataFrame):
    # show sharpe ratio
    assert np.allclose(stats.sortino(meta), 1.16078681)
