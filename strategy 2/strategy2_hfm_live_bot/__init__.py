"""Live-trading implementation of Strategy 2 HFM momentum (fixed USD per active asset only)."""

from .config import (
    BinanceSettings,
    LiveBotSettings,
    LoggingSettings,
    RoostooApiSettings,
    SignalSettings,
    TradingSettings,
    load_env_file,
)
from .bot import HfmFixedUsdLiveBot
from .signal_engine import LiveSignalResult, compute_live_signal
from .binance_client import BinanceSpotMarketDataClient
from .roostoo_client import RoostooClient

__all__ = [
    "BinanceSettings",
    "HfmFixedUsdLiveBot",
    "LiveBotSettings",
    "LiveSignalResult",
    "LoggingSettings",
    "RoostooApiSettings",
    "RoostooClient",
    "SignalSettings",
    "TradingSettings",
    "BinanceSpotMarketDataClient",
    "compute_live_signal",
    "load_env_file",
]
