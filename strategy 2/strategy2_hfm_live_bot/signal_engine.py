from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np
import pandas as pd

from .config import SignalSettings


_RESPONSE_DENOMINATOR = math.sqrt(2.0) * math.exp(-0.5)


@dataclass(frozen=True)
class LiveSignalResult:
    close: pd.DataFrame
    indexed_price: pd.DataFrame
    signal: pd.DataFrame
    x_components: Dict[str, pd.DataFrame]
    z_components: Dict[str, pd.DataFrame]
    u_components: Dict[str, pd.DataFrame]
    warmup_hours_lost: int
    first_valid_timestamp: pd.Timestamp | None

    @property
    def latest_signal_timestamp(self) -> pd.Timestamp:
        valid = self.signal.dropna(how="all")
        if valid.empty:
            raise RuntimeError("No valid signal rows were produced.")
        return valid.index[-1]

    def latest_signal_snapshot(self) -> pd.DataFrame:
        ts = self.latest_signal_timestamp
        frame = pd.DataFrame({"signal": self.signal.loc[ts]})
        for name, x_frame in self.x_components.items():
            frame[f"x_{name}"] = x_frame.loc[ts]
        for name, z_frame in self.z_components.items():
            frame[f"z_{name}"] = z_frame.loc[ts]
        for name, u_frame in self.u_components.items():
            frame[f"u_{name}"] = u_frame.loc[ts]
        return frame.sort_index()


def build_indexed_prices(close: pd.DataFrame) -> pd.DataFrame:
    if close.empty:
        raise ValueError("close panel is empty.")
    first_row = close.iloc[0]
    if first_row.isna().any():
        raise ValueError("close panel cannot contain NaNs in the first row.")
    return close.divide(first_row, axis="columns").astype("float64")


def compute_ema(frame: pd.Series | pd.DataFrame, n: int) -> pd.Series | pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be positive.")
    alpha = 1.0 / float(n)
    return frame.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def response_function(z: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return (z * np.exp(-(z ** 2) / 4.0)) / _RESPONSE_DENOMINATOR


def _rolling_std(frame: pd.DataFrame, window: int, ddof: int) -> pd.DataFrame:
    return frame.rolling(window=window, min_periods=window).std(ddof=ddof)


def compute_live_signal(close: pd.DataFrame, signal_settings: SignalSettings) -> LiveSignalResult:
    signal_settings.validate()

    if len(close) < signal_settings.required_history_bars:
        raise ValueError(
            f"Not enough close history to compute the signal. "
            f"Need at least {signal_settings.required_history_bars} bars, got {len(close)}."
        )

    indexed_price = build_indexed_prices(close)
    short_volatility = _rolling_std(
        indexed_price,
        window=signal_settings.short_norm_hours,
        ddof=signal_settings.rolling_std_ddof,
    )

    x_components: Dict[str, pd.DataFrame] = {}
    z_components: Dict[str, pd.DataFrame] = {}
    u_components: Dict[str, pd.DataFrame] = {}

    signal_sum = pd.DataFrame(0.0, index=indexed_price.index, columns=indexed_price.columns)
    component_count = 0

    for component_idx, (short_n, long_n) in enumerate(
        zip(signal_settings.ema_short_n, signal_settings.ema_long_n),
        start=1,
    ):
        component_name = f"k{component_idx}"
        ema_short = compute_ema(indexed_price, short_n)
        ema_long = compute_ema(indexed_price, long_n)

        x_frame = ema_short - ema_long
        zero_x = x_frame.abs() <= 1e-15

        y_frame = x_frame / short_volatility
        y_frame = y_frame.mask(zero_x, 0.0)
        y_frame = y_frame.mask(short_volatility.abs() <= 1e-15, 0.0)

        long_volatility = _rolling_std(
            y_frame,
            window=signal_settings.long_norm_hours,
            ddof=signal_settings.rolling_std_ddof,
        )
        z_frame = y_frame / long_volatility
        z_frame = z_frame.mask(zero_x, 0.0)
        z_frame = z_frame.mask(short_volatility.abs() <= 1e-15, 0.0)
        z_frame = z_frame.mask(long_volatility.abs() <= 1e-15, 0.0)

        u_frame = response_function(z_frame)

        x_components[component_name] = x_frame
        z_components[component_name] = z_frame
        u_components[component_name] = u_frame

        signal_sum = signal_sum.add(u_frame, fill_value=np.nan)
        component_count += 1

    signal = signal_sum / float(component_count)
    warmup_hours_lost = signal_settings.warmup_hours_lost
    valid_signal = signal.dropna(how="all")
    first_valid_timestamp = None if valid_signal.empty else valid_signal.index[0]

    return LiveSignalResult(
        close=close.copy(),
        indexed_price=indexed_price,
        signal=signal,
        x_components=x_components,
        z_components=z_components,
        u_components=u_components,
        warmup_hours_lost=warmup_hours_lost,
        first_valid_timestamp=first_valid_timestamp,
    )
