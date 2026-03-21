from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


DEFAULT_BINANCE_QUOTE_PRIORITY: Tuple[str, ...] = (
    "USDT",
    "FDUSD",
    "USDC",
    "TUSD",
    "BUSD",
    "USDP",
)


@dataclass(frozen=True)
class Strategy2DataConfig:
    """Data-loading settings for the Strategy 2 research backtest."""

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: str = "1h"
    min_history_days: int = 90
    max_missing_ratio: float = 0.02
    drop_incomplete_timestamps: bool = True
    cache_dir: str = "strategy2_hfm_cache"
    roostoo_quote_asset: str = "USD"
    binance_quote_priority: Tuple[str, ...] = DEFAULT_BINANCE_QUOTE_PRIORITY

    # Optional market-cap enrichment after universe preparation.
    include_market_cap: bool = True
    market_cap_csv_path: str = ""
    refresh_market_cap_cache: bool = False
    allow_missing_market_cap: bool = True

    # Convenience defaults for later train/OOS slicing.
    insample_fraction: float = 0.70
    oos_start_date: Optional[str] = None

    refresh_roostoo_exchange_info: bool = False
    refresh_binance_exchange_info: bool = False
    refresh_history: bool = False
    use_archive_downloads: bool = True
    include_rest_topup: bool = True
    request_timeout_seconds: int = 20
    request_pause_seconds: float = 0.0

    def validate(self) -> None:
        if self.interval != "1h":
            raise ValueError("This Strategy 2 implementation currently supports interval='1h' only.")
        if self.min_history_days < 0:
            raise ValueError("min_history_days cannot be negative.")
        if not 0.0 <= self.max_missing_ratio < 1.0:
            raise ValueError("max_missing_ratio must be in [0, 1).")
        if not 0.0 < self.insample_fraction < 1.0:
            raise ValueError("insample_fraction must be in (0, 1).")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        if self.request_pause_seconds < 0:
            raise ValueError("request_pause_seconds cannot be negative.")
