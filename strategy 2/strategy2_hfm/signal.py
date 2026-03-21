from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import Strategy2SignalConfig
from .data import build_indexed_prices, compute_log_returns


_RESPONSE_DENOMINATOR = math.sqrt(2.0) * math.exp(-0.5)


@dataclass
class Strategy2SignalResult:
    indexed_price: pd.DataFrame
    log_return: pd.DataFrame
    ema_short: Dict[str, pd.DataFrame]
    ema_long: Dict[str, pd.DataFrame]
    x_components: Dict[str, pd.DataFrame]
    short_volatility: pd.DataFrame
    y_components: Dict[str, pd.DataFrame]
    long_volatility: Dict[str, pd.DataFrame]
    z_components: Dict[str, pd.DataFrame]
    u_components: Dict[str, pd.DataFrame]
    signal: pd.DataFrame
    warmup_hours_lost: int
    first_valid_timestamp: pd.Timestamp | None


def compute_ema(series: pd.Series | pd.DataFrame, n: int) -> pd.Series | pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive.")
    alpha = 1.0 / float(n)
    return series.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def response_function(z: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return (z * np.exp(-(z ** 2) / 4.0)) / _RESPONSE_DENOMINATOR


def _rolling_std(frame: pd.DataFrame, window: int, ddof: int) -> pd.DataFrame:
    return frame.rolling(window=window, min_periods=window).std(ddof=ddof)


def compute_hfm_signal(
    close: pd.DataFrame,
    signal_config: Strategy2SignalConfig,
) -> Strategy2SignalResult:
    signal_config.validate()

    indexed_price = build_indexed_prices(close)
    log_return = compute_log_returns(indexed_price)

    ema_short: Dict[str, pd.DataFrame] = {}
    ema_long: Dict[str, pd.DataFrame] = {}
    x_components: Dict[str, pd.DataFrame] = {}
    y_components: Dict[str, pd.DataFrame] = {}
    long_volatility: Dict[str, pd.DataFrame] = {}
    z_components: Dict[str, pd.DataFrame] = {}
    u_components: Dict[str, pd.DataFrame] = {}

    short_volatility = _rolling_std(
        indexed_price,
        window=signal_config.short_norm_hours,
        ddof=signal_config.rolling_std_ddof,
    )

    component_names: List[str] = []
    for component_idx, (short_n, long_n) in enumerate(
        zip(signal_config.ema_short_n, signal_config.ema_long_n),
        start=1,
    ):
        component_name = f"k{component_idx}"
        component_names.append(component_name)

        ema_short_frame = compute_ema(indexed_price, short_n)
        ema_long_frame = compute_ema(indexed_price, long_n)
        x_frame = ema_short_frame - ema_long_frame

        zero_x = x_frame.abs() <= 1e-15
        short_defined = short_volatility.notna()
        zero_short = short_defined & (short_volatility.abs() <= 1e-15)

        y_frame = x_frame / short_volatility
        y_frame = y_frame.mask(zero_x & short_defined, 0.0)
        y_frame = y_frame.mask(zero_short, 0.0)

        long_vol_frame = _rolling_std(
            y_frame,
            window=signal_config.long_norm_hours,
            ddof=signal_config.rolling_std_ddof,
        )
        long_defined = long_vol_frame.notna()
        zero_long = long_defined & (long_vol_frame.abs() <= 1e-15)

        z_frame = y_frame / long_vol_frame
        z_frame = z_frame.mask(zero_x & short_defined & long_defined, 0.0)
        z_frame = z_frame.mask(zero_short, 0.0)
        z_frame = z_frame.mask(zero_long, 0.0)

        u_frame = response_function(z_frame)

        ema_short[component_name] = ema_short_frame
        ema_long[component_name] = ema_long_frame
        x_components[component_name] = x_frame
        y_components[component_name] = y_frame
        long_volatility[component_name] = long_vol_frame
        z_components[component_name] = z_frame
        u_components[component_name] = u_frame

    signal = u_components[component_names[0]].copy()
    for component_name in component_names[1:]:
        signal = signal + u_components[component_name]
    signal = signal / float(len(component_names))
    warmup_hours_lost = (signal_config.short_norm_hours - 1) + (signal_config.long_norm_hours - 1)

    valid_signal = signal.dropna(how="all")
    first_valid_timestamp = None if valid_signal.empty else valid_signal.index.min()

    return Strategy2SignalResult(
        indexed_price=indexed_price,
        log_return=log_return,
        ema_short=ema_short,
        ema_long=ema_long,
        x_components=x_components,
        short_volatility=short_volatility,
        y_components=y_components,
        long_volatility=long_volatility,
        z_components=z_components,
        u_components=u_components,
        signal=signal,
        warmup_hours_lost=warmup_hours_lost,
        first_valid_timestamp=first_valid_timestamp,
    )


__all__ = [
    "Strategy2SignalResult",
    "compute_ema",
    "compute_hfm_signal",
    "response_function",
]
