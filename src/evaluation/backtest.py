"""Honest long-only backtest with transaction costs.

The goal is not to claim alpha — it's to **show whether the model's edge
survives realistic frictions**. A model with AUC 0.55 in cross-validation
can still lose money after costs if the average winning trade is smaller
than the round-trip transaction cost.

Strategy implemented:
  * On each rebalance day, take the top `top_k` predictions (by probability).
  * Hold them equally weighted until the next rebalance.
  * Charge `transaction_cost_bps` per leg on every position change.
  * Return the daily P&L series and standard summary metrics
    (cumulative return, annualised Sharpe, max drawdown).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BacktestResult:
    cumulative_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "cumulative_return": self.cumulative_return,
            "annualised_return": self.annualised_return,
            "annualised_volatility": self.annualised_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
        }


def _annualisation_factor(rebalance_days: int) -> float:
    """~252 trading days / N days between rebalances."""
    return 252.0 / max(rebalance_days, 1)


def backtest_top_k(
    predictions: pl.DataFrame,
    realised_returns_col: str,
    top_k: int = 20,
    rebalance_days: int = 5,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """Simulate a long-only top-K portfolio rebalanced every `rebalance_days`.

    Args:
        predictions: long DataFrame with columns
            `date, symbol, predicted_probability, <realised_returns_col>`.
            `realised_returns_col` is the forward return the model was trying
            to predict (e.g. `label_return_5d`).
        top_k:                 Number of positions held simultaneously.
        rebalance_days:        Trading days between rebalances.
        transaction_cost_bps:  Round-trip cost per position change (basis points).
    """
    if predictions.is_empty():
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    # On each date, rank by probability and keep top_k.
    ranked = predictions.with_columns(
        pl.col("predicted_probability")
        .rank(method="ordinal", descending=True)
        .over("date")
        .alias("_rk")
    )
    picks = ranked.filter(pl.col("_rk") <= top_k)

    # Equal-weight portfolio return per rebalance date.
    per_date = (
        picks.group_by("date")
        .agg(pl.col(realised_returns_col).mean().alias("gross_return"))
        .sort("date")
    )
    cost_per_rebalance = 2.0 * transaction_cost_bps / 10_000.0
    per_date = per_date.with_columns(
        (pl.col("gross_return") - cost_per_rebalance).alias("net_return")
    )

    returns = per_date["net_return"].to_numpy()
    if returns.size == 0:
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    cum = float(np.prod(1.0 + returns) - 1.0)
    factor = _annualisation_factor(rebalance_days)
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    ann_ret = mean_r * factor
    ann_vol = std_r * np.sqrt(factor)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0

    # Max drawdown on the equity curve.
    equity = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity)
    drawdown = (equity - running_peak) / running_peak
    max_dd = float(drawdown.min())

    win_rate = float((returns > 0).mean())
    n_trades = int(len(returns) * top_k)

    return BacktestResult(
        cumulative_return=cum,
        annualised_return=ann_ret,
        annualised_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        n_trades=n_trades,
    )
