import yfinance as _yf
import pandas as _pd
import numpy as _np


def download_returns(ticker, period="max", proxy=None):
    params = {
        "tickers": ticker,
        "proxy": proxy,
        "auto_adjust": True,
        "multi_level_index": False,
        "progress": False,
    }
    if isinstance(period, _pd.DatetimeIndex):
        params["start"] = period[0]
    else:
        params["period"] = period
    df = _yf.download(**params)["Close"].pct_change()
    df = df.tz_localize(None)
    return df


def make_index(
    ticker_weights, rebalance="1M", period="max", returns=None, match_dates=False
):
    """
    Makes an index out of the given tickers and weights.
    Optionally you can pass a dataframe with the returns.
    If returns is not given it try to download them with yfinance

    Args:
        * ticker_weights (Dict): A python dict with tickers as keys
            and weights as values
        * rebalance: Pandas resample interval or None for never
        * period: time period of the returns to be downloaded
        * returns (Series, DataFrame): Optional. Returns If provided,
            it will fist check if returns for the given ticker are in
            this dataframe, if not it will try to download them with
            yfinance
    Returns:
        * index_returns (Series, DataFrame): Returns for the index
    """
    # Declare a returns variable
    index = None
    portfolio = {}

    # Iterate over weights
    for ticker in ticker_weights.keys():
        if (returns is None) or (ticker not in returns.columns):
            # Download the returns for this ticker, e.g. GOOG
            ticker_returns = download_returns(ticker, period)
        else:
            ticker_returns = returns[ticker]

        portfolio[ticker] = ticker_returns

    # index members time-series
    index = _pd.DataFrame(portfolio).dropna()

    if match_dates:
        index = index[max(index.ne(0).idxmax()) :]

    # no rebalance?
    if rebalance is None:
        for ticker, weight in ticker_weights.items():
            index[ticker] = weight * index[ticker]
        return index.sum(axis=1)

    last_day = index.index[-1]

    # rebalance marker
    rbdf = index.resample(rebalance).first()
    rbdf["break"] = rbdf.index.strftime("%s")

    # index returns with rebalance markers
    index = _pd.concat([index, rbdf["break"]], axis=1)

    # mark first day day
    index["first_day"] = _pd.isna(index["break"]) & ~_pd.isna(index["break"].shift(1))
    index.loc[index.index[0], "first_day"] = True

    # multiply first day of each rebalance period by the weight
    for ticker, weight in ticker_weights.items():
        index[ticker] = _np.where(
            index["first_day"], weight * index[ticker], index[ticker]
        )

    # drop first marker
    index.drop(columns=["first_day"], inplace=True)

    # drop when all are NaN
    index.dropna(how="all", inplace=True)
    return index[index.index <= last_day].sum(axis=1)
