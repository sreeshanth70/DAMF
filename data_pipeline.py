"""
Consolidated data loading, feature construction, and regime-label generation.

This factors out the ~80-line block that was previously duplicated near-
identically across patchtst_macro.py, patchtst_macro_leakfix.py,
patchtst_walkforward.py, and sota_baselines.py. All four scripts can import
`load_dataset()` from here instead of repeating the fetch/feature/regime code
inline. Regime thresholds (drawdown, momentum) match Section 3.2 of the paper
exactly; see regime_sensitivity_check() for the threshold-robustness analysis
reported in Section 7.5.

Regime encoding: 0=Normal, 1=Boom, 2=Recovery, 3=Crash
  Boom     = not distressed, bullish trend
  Normal   = not distressed, not bullish trend
  Recovery = distressed,     bullish trend
  Crash    = distressed,     not bullish trend
"""
import numpy as np
import pandas as pd
import yfinance as yf

REG_NAMES = {0: "Normal", 1: "Boom", 2: "Recovery", 3: "Crash"}

DEFAULT_DD_THRESH = -0.10     # distress if 252d or 63d drawdown <= this
DEFAULT_TREND_THRESH = 0.02   # bullish if 63d cumulative log return >= this


def load_dataset(start="2000-01-01", end="2023-12-31",
                  dd_thresh=DEFAULT_DD_THRESH, trend_thresh=DEFAULT_TREND_THRESH):
    """Fetch S&P 500 + VIX + macro features, construct the 12-feature set, and
    assign regime labels. Returns a DataFrame with one row per trading day
    (after warm-up), matching Section 4 (Feature Engineering) of the paper."""

    sp = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze()
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze()
    raw = pd.DataFrame({"date": pd.to_datetime(sp.index), "close": sp.values,
                        "vix": vix.reindex(sp.index).values}).dropna()

    import pandas_datareader.data as web
    try:
        yc_raw = web.DataReader("T10Y2Y", "fred", "1999-01-01", end)
        yc_raw.index = pd.to_datetime(yc_raw.index)
    except Exception:
        tnx = yf.download("^TNX", start="1999-01-01", end=end, auto_adjust=True, progress=False)["Close"].squeeze()
        irx = yf.download("^IRX", start="1999-01-01", end=end, auto_adjust=True, progress=False)["Close"].squeeze()
        yc_raw = pd.DataFrame({"T10Y2Y": (tnx - irx).values}, index=pd.to_datetime(tnx.index))

    hyg = yf.download("HYG", start="1999-01-01", end=end, auto_adjust=True, progress=False)["Close"].squeeze()
    ief = yf.download("IEF", start="1999-01-01", end=end, auto_adjust=True, progress=False)["Close"].squeeze()
    hyg.index = pd.to_datetime(hyg.index)
    ief.index = pd.to_datetime(ief.index)
    hyg_ret63 = np.log(hyg / hyg.shift(63)) * 100
    ief_ret63 = np.log(ief / ief.shift(63)) * 100
    credit_hyg = (ief_ret63 - hyg_ret63)

    _vix_s = pd.Series(
        yf.download("^VIX", start="1999-01-01", end=end, auto_adjust=True, progress=False)["Close"].squeeze().values,
        index=pd.to_datetime(yf.download("^VIX", start="1999-01-01", end=end, progress=False).index))
    _vix_z = (_vix_s - _vix_s.rolling(252).mean()) / (_vix_s.rolling(252).std() + 1e-8)
    _overlap = credit_hyg.dropna().index
    _hyg_mean, _hyg_std = credit_hyg.loc[_overlap].mean(), credit_hyg.loc[_overlap].std()
    _vix_proxy = _vix_z * _hyg_std + _hyg_mean
    credit_combined = credit_hyg.copy()
    _pre_hyg = credit_combined[credit_combined.isna()].index
    credit_combined.loc[_pre_hyg] = _vix_proxy.reindex(_pre_hyg)
    hy_raw = credit_combined.to_frame("BAMLH0A0HYM2")

    df = raw.copy().set_index("date")
    yc_daily = yc_raw["T10Y2Y"].reindex(df.index).ffill().bfill()
    hy_daily = hy_raw["BAMLH0A0HYM2"].reindex(df.index).ffill().bfill()
    df["yield_slope"] = yc_daily.values
    df["yield_slope_chg"] = df["yield_slope"].diff(1)
    df["credit_spread"] = hy_daily.values
    df["credit_spread_chg"] = df["credit_spread"].diff(1)
    df = df.reset_index()

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["mom5"] = df["log_ret"].rolling(5).sum()
    df["mom21"] = df["log_ret"].rolling(21).sum()
    df["vol21"] = df["log_ret"].rolling(21).std()
    df["vix"] = df["vix"] / 100.0
    df["vol5"] = df["log_ret"].rolling(5).std()
    df["vol_ratio"] = df["vol5"] / (df["vol21"] + 1e-8)
    df["drawdown"] = df["close"] / df["close"].rolling(63).max() - 1
    df["vix_change"] = df["vix"].pct_change(1)
    df = df.dropna().reset_index(drop=True)

    df = assign_regimes(df, dd_thresh, trend_thresh)
    return df


def assign_regimes(df, dd_thresh=DEFAULT_DD_THRESH, trend_thresh=DEFAULT_TREND_THRESH):
    """Assign the 4-way regime label (Section 3.2, Eq. 4) given a threshold pair.
    Exposed separately from load_dataset() so the Section 7.5 threshold-
    sensitivity check can re-label an already-fetched DataFrame without
    re-downloading or re-fetching macro data."""
    df = df.copy()
    df["_dd252"] = df["close"] / df["close"].rolling(252).max() - 1
    df["_trend63"] = df["log_ret"].rolling(63).sum()
    dist = (df["_dd252"] <= dd_thresh) | (df["drawdown"] <= dd_thresh)
    bull = df["_trend63"] >= trend_thresh
    df["regime"] = np.where(~dist & bull, 1,
                    np.where(~dist & ~bull, 0,
                    np.where(dist & bull, 2, 3))).astype(int)
    return df.dropna().reset_index(drop=True)


FEATURE_COLUMNS = [
    "log_ret", "mom5", "mom21", "vol21", "vol_ratio", "drawdown",
    "vix", "vix_change", "yield_slope", "yield_slope_chg",
    "credit_spread", "credit_spread_chg",
]

if __name__ == "__main__":
    df = load_dataset()
    print(f"Total days: {len(df)}")
    for r, nm in REG_NAMES.items():
        n = (df["regime"] == r).sum()
        print(f"  {nm:<10} {n:5d} ({100*n/len(df):.1f}%)")
