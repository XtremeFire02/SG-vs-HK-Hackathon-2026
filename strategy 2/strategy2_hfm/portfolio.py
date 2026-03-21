from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


@dataclass
class Strategy2TargetWeights:
    """Long/cash target weights derived from the bounded HFM signal."""

    weights: pd.DataFrame
    cash_weight: pd.Series
    gross_exposure: pd.Series
    active_count: pd.Series
    long_signal_threshold: float = 0.0



def build_long_only_ts_target_weights(
    signal: pd.DataFrame,
    *,
    long_signal_threshold: float = 0.0,
    position_rule: str = "signal_above_threshold_div_universe",
    bet_size: float | None = None,
) -> Strategy2TargetWeights:
    """Build long/cash target weights from the signal and sizing rule.

    Supported modes:
        - signal_above_threshold_div_universe:
            w_i,t = signal_i,t / N when signal_i,t > threshold, else 0.
        - fixed_pct_per_active_asset:
            w_i,t = bet_size when signal_i,t > threshold, else 0.

    fixed_usd_per_active_asset is intentionally excluded here because it depends on
    portfolio NAV at each rebalance and is handled directly inside the backtest loop.
    """

    if signal.empty:
        raise ValueError("signal frame is empty.")

    if not math.isfinite(long_signal_threshold):
        raise ValueError("long_signal_threshold must be finite.")

    n_assets = int(signal.shape[1])
    if n_assets <= 0:
        raise ValueError("signal frame must contain at least one asset column.")

    active_mask = signal.gt(float(long_signal_threshold)).fillna(False)

    if position_rule == "signal_above_threshold_div_universe":
        thresholded_signal = signal.where(active_mask, 0.0)
        positive_signal = thresholded_signal.clip(lower=0.0).fillna(0.0)
        weights = positive_signal / float(n_assets)
    elif position_rule == "fixed_pct_per_active_asset":
        if bet_size is None:
            raise ValueError("bet_size is required for fixed_pct_per_active_asset.")
        weights = active_mask.astype("float64") * float(bet_size)
    elif position_rule == "fixed_usd_per_active_asset":
        raise ValueError(
            "fixed_usd_per_active_asset is a direct-notional sizing rule and must be "
            "constructed inside run_long_only_hfm_backtest."
        )
    else:
        raise ValueError(f"Unknown position_rule: {position_rule}")

    gross_exposure = weights.sum(axis=1)
    cash_weight = (1.0 - gross_exposure).clip(lower=0.0)
    active_count = (weights > 0.0).sum(axis=1)

    return Strategy2TargetWeights(
        weights=weights.astype("float64"),
        cash_weight=cash_weight.astype("float64"),
        gross_exposure=gross_exposure.astype("float64"),
        active_count=active_count.astype("int64"),
        long_signal_threshold=float(long_signal_threshold),
    )


__all__ = [
    "Strategy2TargetWeights",
    "build_long_only_ts_target_weights",
]
