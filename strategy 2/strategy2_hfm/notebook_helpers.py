from __future__ import annotations

from typing import Optional

import pandas as pd

from .backtest import Strategy2BacktestResult
from .data import Strategy2PreparedData
from .portfolio import Strategy2TargetWeights
from .signal import Strategy2SignalResult


def summarize_backtest(metrics: dict[str, float]) -> str:
    lines = [
        f"Initial equity:        {metrics['initial_equity']:.2f}",
        f"Final equity:          {metrics['final_equity']:.2f}",
        f"Total return:          {metrics['total_return']:.2%}",
        f"Annualized return:     {metrics['annualized_return']:.2%}",
        f"Annualized volatility: {metrics['annualized_volatility']:.2%}",
        f"Sharpe:                {metrics['sharpe']:.3f}",
        f"Sortino:               {metrics['sortino']:.3f}",
        f"Max drawdown:          {metrics['max_drawdown']:.2%}",
        f"Trade count:           {metrics['trade_count']:.0f}",
        f"Turnover multiple:     {metrics['turnover_multiple']:.2f}x avg equity",
        f"Average cash weight:   {metrics['average_cash_weight']:.2%}",
        f"Average gross exposure:{metrics['average_gross_exposure']:.2%}",
        f"Average holding count: {metrics['average_holding_count']:.2f}",
        f"Time fully in cash:    {metrics['pct_time_fully_in_cash']:.2%}",
    ]
    return "\n".join(lines)


def latest_signal_snapshot(
    signal_result: Strategy2SignalResult,
    *,
    timestamp: Optional[pd.Timestamp] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    valid = signal_result.signal.dropna(how="all")
    if valid.empty:
        return pd.DataFrame()
    ts = valid.index.max() if timestamp is None else timestamp

    frame = pd.DataFrame({"signal": signal_result.signal.loc[ts]})
    for component_name, x_frame in signal_result.x_components.items():
        frame[f"x_{component_name}"] = x_frame.loc[ts]
    for component_name, z_frame in signal_result.z_components.items():
        frame[f"z_{component_name}"] = z_frame.loc[ts]
    for component_name, u_frame in signal_result.u_components.items():
        frame[f"u_{component_name}"] = u_frame.loc[ts]

    frame = frame.sort_values("signal", ascending=False)
    if top_n is not None:
        frame = frame.head(top_n)
    return frame


def latest_target_snapshot(
    target_weights: Strategy2TargetWeights,
    *,
    timestamp: Optional[pd.Timestamp] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    valid = target_weights.weights.dropna(how="all")
    if valid.empty:
        return pd.DataFrame()
    ts = valid.index.max() if timestamp is None else timestamp

    frame = pd.DataFrame(
        {
            "target_weight": target_weights.weights.loc[ts],
        }
    ).sort_values("target_weight", ascending=False)
    if top_n is not None:
        frame = frame.head(top_n)
    return frame


def latest_market_cap_snapshot(
    prepared_data: Strategy2PreparedData,
    *,
    timestamp: Optional[pd.Timestamp] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    snapshot = prepared_data.get_market_cap_snapshot(timestamp=timestamp)
    frame = snapshot.rename("market_cap_usd").to_frame().sort_values("market_cap_usd", ascending=False)
    if top_n is not None:
        frame = frame.head(top_n)
    return frame


def build_daily_equity_curve(result: Strategy2BacktestResult) -> pd.DataFrame:
    equity = result.equity_curve.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
    equity["date"] = equity["timestamp"].dt.floor("D")
    daily = equity.groupby("date", as_index=False).last()
    daily["daily_return"] = daily["equity"].pct_change().fillna(0.0)
    return daily


__all__ = [
    "build_daily_equity_curve",
    "latest_market_cap_snapshot",
    "latest_signal_snapshot",
    "latest_target_snapshot",
    "summarize_backtest",
]
