from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Strategy2SignalConfig:
    """Signal-engine settings for the EMA-based HFM momentum signal."""

    ema_short_n: Tuple[int, int, int] = (8, 16, 32)
    ema_long_n: Tuple[int, int, int] = (24, 48, 96)
    short_norm_hours: int = 12
    long_norm_hours: int = 168
    rolling_std_ddof: int = 1

    def validate(self) -> None:
        if len(self.ema_short_n) != len(self.ema_long_n):
            raise ValueError("ema_short_n and ema_long_n must have the same length.")
        if len(self.ema_short_n) == 0:
            raise ValueError("At least one EMA pair is required.")
        if self.short_norm_hours <= 1:
            raise ValueError("short_norm_hours must be greater than 1.")
        if self.long_norm_hours <= 1:
            raise ValueError("long_norm_hours must be greater than 1.")
        if self.rolling_std_ddof not in {0, 1}:
            raise ValueError("rolling_std_ddof must be 0 or 1.")
