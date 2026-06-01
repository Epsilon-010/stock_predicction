"""Cross-sectional features: a ticker's rank within the universe per day.

These features are what separates a serious quant pipeline from a per-ticker
toy. Predicting *"is Toyota likely to go up"* in isolation is much harder than
predicting *"will Toyota outperform the average TSE stock this week"* — the
second framing strips out market-wide moves and lets the model learn relative
strength.

All ranks are normalised to [0, 1] (0 = worst in universe today, 1 = best).
"""

from __future__ import annotations

import polars as pl

# Columns we rank cross-sectionally, plus the direction (descending=True means
# "high value is good", e.g. high momentum is a positive signal).
_RANK_COLUMNS: dict[str, bool] = {
    "returns_5d": True,
    "returns_20d": True,
    "returns_60d": True,
    "volume_zscore_20d": True,
    "volatility_20d": False,  # low vol = top-ranked (defensive ranking)
    "rsi_14": True,
}


def add_cross_sectional_ranks(df: pl.DataFrame) -> pl.DataFrame:
    """Rank each ticker against its peers on the same `date`.

    The DataFrame must contain a `date` column and at least one of the columns
    in `_RANK_COLUMNS`. Output columns are named `<col>_rank` and live in [0, 1].
    """
    exprs: list[pl.Expr] = []
    for col, descending in _RANK_COLUMNS.items():
        if col not in df.columns:
            continue
        # Polars rank is 1-indexed. With `descending=True` rank 1 means
        # "biggest value" — which is what we want at the top of our 0..1
        # ranking (best = 1.0, worst = 0.0). Hence `(n - rank) / (n - 1)`.
        rank = pl.col(col).rank(method="average", descending=descending).over("date")
        n = pl.col(col).count().over("date").cast(pl.Float64)
        exprs.append(((n - rank) / (n - 1)).alias(f"{col}_rank"))
    return df.with_columns(exprs) if exprs else df


def add_market_relative_returns(
    df: pl.DataFrame,
    market_symbol: str = "^TPX",
) -> pl.DataFrame:
    """Excess return vs the market index (e.g. TOPIX).

    Assumes `df` is a *combined* long DataFrame containing both individual
    tickers and the market index. The market's returns are subtracted from
    each ticker's returns of the same date.

    If the market symbol is not present, the function is a no-op.
    """
    if "symbol" not in df.columns or "returns_5d" not in df.columns:
        return df

    market_df = df.filter(pl.col("symbol") == market_symbol).select(
        ["date", pl.col("returns_5d").alias("_mkt_returns_5d")]
    )
    if market_df.is_empty():
        return df

    return (
        df.join(market_df, on="date", how="left")
        .with_columns(
            (pl.col("returns_5d") - pl.col("_mkt_returns_5d")).alias("excess_returns_5d"),
        )
        .drop("_mkt_returns_5d")
    )
