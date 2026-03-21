from __future__ import annotations

import math
from typing import Dict

import pandas as pd


ANNUALIZATION_FACTOR_HOURLY = 24.0 * 365.0


def cumulative_return(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0:
        return float("nan")
    return float(end / start - 1.0)


def annualized_return(equity: pd.Series, periods_per_year: float = ANNUALIZATION_FACTOR_HOURLY) -> float:
    if equity.empty:
        return float("nan")
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0:
        return float("nan")
    n_periods = len(equity)
    if n_periods <= 1:
        return 0.0
    return float((end / start) ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(hourly_returns: pd.Series, periods_per_year: float = ANNUALIZATION_FACTOR_HOURLY) -> float:
    if hourly_returns.empty:
        return float("nan")
    return float(hourly_returns.std(ddof=0) * math.sqrt(periods_per_year))


def sharpe_ratio(hourly_returns: pd.Series) -> float:
    ann_vol = annualized_volatility(hourly_returns)
    if not math.isfinite(ann_vol) or ann_vol <= 0:
        return float("nan")
    ann_ret = annualized_return((1.0 + hourly_returns.fillna(0.0)).cumprod())
    if not math.isfinite(ann_ret):
        return float("nan")
    return float(ann_ret / ann_vol)


def sortino_ratio(hourly_returns: pd.Series) -> float:
    downside = hourly_returns[hourly_returns < 0.0]
    if downside.empty:
        return float("nan")
    downside_vol = float(downside.std(ddof=0) * math.sqrt(ANNUALIZATION_FACTOR_HOURLY))
    if downside_vol <= 0:
        return float("nan")
    ann_ret = annualized_return((1.0 + hourly_returns.fillna(0.0)).cumprod())
    if not math.isfinite(ann_ret):
        return float("nan")
    return float(ann_ret / downside_vol)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def turnover_multiple(turnover_notional: float, average_equity: float) -> float:
    if average_equity <= 0:
        return float("nan")
    return float(turnover_notional / average_equity)


def build_backtest_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    turnover_notional: float,
) -> Dict[str, float]:
    if equity_curve.empty:
        raise RuntimeError("Cannot build metrics from an empty equity curve.")

    curve = equity_curve.copy().reset_index(drop=True)
    curve["hourly_return"] = curve["equity"].pct_change().fillna(0.0)
    equity = curve["equity"]
    hourly_returns = curve["hourly_return"]

    metrics = {
        "initial_equity": float(equity.iloc[0]),
        "final_equity": float(equity.iloc[-1]),
        "total_return": cumulative_return(equity),
        "annualized_return": annualized_return(equity),
        "annualized_volatility": annualized_volatility(hourly_returns),
        "sharpe": sharpe_ratio(hourly_returns),
        "sortino": sortino_ratio(hourly_returns),
        "max_drawdown": max_drawdown(equity),
        "trade_count": float(len(trades)),
        "turnover_notional": float(turnover_notional),
        "turnover_multiple": turnover_multiple(float(turnover_notional), float(equity.mean())),
        "average_cash_weight": float(curve["cash_weight"].mean()) if "cash_weight" in curve else float("nan"),
        "average_gross_exposure": float(curve["gross_exposure"].mean()) if "gross_exposure" in curve else float("nan"),
        "average_holding_count": float(curve["holding_count"].mean()) if "holding_count" in curve else float("nan"),
        "pct_time_fully_in_cash": float((curve["gross_exposure"] <= 1e-12).mean()) if "gross_exposure" in curve else float("nan"),
    }
    return metrics


__all__ = [
    "ANNUALIZATION_FACTOR_HOURLY",
    "annualized_return",
    "annualized_volatility",
    "build_backtest_metrics",
    "cumulative_return",
    "max_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "turnover_multiple",
]
