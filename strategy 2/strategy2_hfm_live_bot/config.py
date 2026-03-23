from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple


DEFAULT_ACTIVE_PAIRS: Tuple[str, ...] = ("FLOKI/USD", "NEAR/USD", "PEPE/USD", "SHIB/USD")
DEFAULT_BINANCE_QUOTE_PRIORITY: Tuple[str, ...] = (
    "USDT",
    "FDUSD",
    "USDC",
    "TUSD",
    "BUSD",
    "USDP",
)


def load_env_file(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> None:
    """Lightweight .env loader so the bot does not depend on python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value from: {value!r}")


def _parse_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    return int(value.strip())


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None:
        return default
    return float(value.strip())


def _parse_csv_tuple(value: str | None, *, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return default
    items = [item.strip() for item in value.split(",")]
    return tuple(item for item in items if item)


def _parse_json_mapping(value: str | None) -> dict[str, str]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object for the symbol override mapping.")
    return {str(k): str(v) for k, v in parsed.items()}


@dataclass(frozen=True)
class RoostooApiSettings:
    mode: str = "TEST"
    test_base_url: str = "https://mock-api.roostoo.com"
    live_base_url: str = ""
    test_api_key: str = ""
    test_secret_key: str = ""
    live_api_key: str = ""
    live_secret_key: str = ""
    timeout_seconds: int = 10
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5

    def validate(self) -> None:
        mode = self.mode.upper()
        if mode not in {"TEST", "LIVE"}:
            raise ValueError("roostoo mode must be TEST or LIVE.")
        if not self.resolved_base_url:
            raise ValueError("Resolved Roostoo base URL is empty.")
        if not self.resolved_api_key or not self.resolved_secret_key:
            raise ValueError(f"Missing API credentials for mode={mode}.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be positive.")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative.")

    @property
    def resolved_base_url(self) -> str:
        if self.mode.upper() == "LIVE":
            return self.live_base_url or self.test_base_url
        return self.test_base_url

    @property
    def resolved_api_key(self) -> str:
        return self.live_api_key if self.mode.upper() == "LIVE" else self.test_api_key

    @property
    def resolved_secret_key(self) -> str:
        return self.live_secret_key if self.mode.upper() == "LIVE" else self.test_secret_key


@dataclass(frozen=True)
class BinanceSettings:
    base_url: str = "https://data-api.binance.vision"
    interval: str = "1h"
    quote_priority: Tuple[str, ...] = DEFAULT_BINANCE_QUOTE_PRIORITY
    symbol_overrides: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: int = 20
    pause_seconds: float = 0.0

    def validate(self) -> None:
        if self.interval != "1h":
            raise ValueError("This live bot currently supports Binance interval='1h' only.")
        if not self.base_url:
            raise ValueError("Binance base URL cannot be empty.")
        if self.timeout_seconds <= 0:
            raise ValueError("Binance timeout_seconds must be positive.")
        if self.pause_seconds < 0:
            raise ValueError("Binance pause_seconds cannot be negative.")
        if not self.quote_priority:
            raise ValueError("At least one Binance quote asset priority is required.")


@dataclass(frozen=True)
class SignalSettings:
    ema_short_n: Tuple[int, int, int] = (8, 16, 32)
    ema_long_n: Tuple[int, int, int] = (24, 48, 96)
    short_norm_hours: int = 12
    long_norm_hours: int = 168
    rolling_std_ddof: int = 1
    signal_threshold: float = 0.5
    extra_warmup_hours: int = 0

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
        if self.extra_warmup_hours < 0:
            raise ValueError("extra_warmup_hours cannot be negative.")

    @property
    def warmup_hours_lost(self) -> int:
        return (self.short_norm_hours - 1) + (self.long_norm_hours - 1)

    @property
    def required_history_bars(self) -> int:
        return self.warmup_hours_lost + 1 + self.extra_warmup_hours


@dataclass(frozen=True)
class TradingSettings:
    active_pairs: Tuple[str, ...] = DEFAULT_ACTIVE_PAIRS
    bet_size_usd: float = 1_000.0
    cash_reserve_usd: float = 0.0
    execute_trades: bool = False
    run_forever: bool = False
    bar_close_delay_seconds: int = 10
    loop_poll_seconds: float = 5.0
    estimated_fee_rate: float = 0.001
    balance_epsilon: float = 1e-12
    usd_asset: str = "USD"

    def validate(self) -> None:
        if not self.active_pairs:
            raise ValueError("active_pairs cannot be empty.")
        if self.bet_size_usd <= 0:
            raise ValueError("bet_size_usd must be positive.")
        if self.cash_reserve_usd < 0:
            raise ValueError("cash_reserve_usd cannot be negative.")
        if self.bar_close_delay_seconds < 0:
            raise ValueError("bar_close_delay_seconds cannot be negative.")
        if self.loop_poll_seconds <= 0:
            raise ValueError("loop_poll_seconds must be positive.")
        if self.estimated_fee_rate < 0:
            raise ValueError("estimated_fee_rate cannot be negative.")
        if self.balance_epsilon < 0:
            raise ValueError("balance_epsilon cannot be negative.")
        if not self.usd_asset:
            raise ValueError("usd_asset cannot be empty.")
        seen = set()
        for pair in self.active_pairs:
            if pair in seen:
                raise ValueError(f"Duplicate active pair: {pair}")
            seen.add(pair)


@dataclass(frozen=True)
class LoggingSettings:
    jsonl_path: str = "strategy2_hfm_live_log.jsonl"

    def validate(self) -> None:
        if not self.jsonl_path:
            raise ValueError("jsonl_path cannot be empty.")


@dataclass(frozen=True)
class LiveBotSettings:
    roostoo: RoostooApiSettings = field(default_factory=RoostooApiSettings)
    binance: BinanceSettings = field(default_factory=BinanceSettings)
    signal: SignalSettings = field(default_factory=SignalSettings)
    trading: TradingSettings = field(default_factory=TradingSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)

    def validate(self) -> None:
        self.roostoo.validate()
        self.binance.validate()
        self.signal.validate()
        self.trading.validate()
        self.logging.validate()

    @classmethod
    def from_env(cls, env_path: str | None = None) -> "LiveBotSettings":
        if env_path:
            load_env_file(env_path)

        roostoo = RoostooApiSettings(
            mode=_get_env("ROOSTOO_MODE", "TEST") or "TEST",
            test_base_url=_get_env("ROOSTOO_TEST_BASE_URL", _get_env("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")) or "https://mock-api.roostoo.com",
            live_base_url=_get_env("ROOSTOO_LIVE_BASE_URL", _get_env("ROOSTOO_COMPETITION_BASE_URL", "")) or "",
            test_api_key=_get_env("ROOSTOO_TEST_API_KEY", "") or "",
            test_secret_key=_get_env("ROOSTOO_TEST_SECRET_KEY", "") or "",
            live_api_key=_get_env("ROOSTOO_LIVE_API_KEY", "") or "",
            live_secret_key=_get_env("ROOSTOO_LIVE_SECRET_KEY", "") or "",
            timeout_seconds=_parse_int(_get_env("ROOSTOO_TIMEOUT_SECONDS"), default=10),
            max_retries=_parse_int(_get_env("ROOSTOO_MAX_RETRIES"), default=3),
            retry_backoff_seconds=_parse_float(_get_env("ROOSTOO_RETRY_BACKOFF_SECONDS"), default=0.5),
        )

        binance = BinanceSettings(
            base_url=_get_env("BINANCE_BASE_URL", "https://data-api.binance.vision") or "https://data-api.binance.vision",
            interval=_get_env("BINANCE_INTERVAL", "1h") or "1h",
            quote_priority=_parse_csv_tuple(_get_env("BINANCE_QUOTE_PRIORITY"), default=DEFAULT_BINANCE_QUOTE_PRIORITY),
            symbol_overrides=_parse_json_mapping(_get_env("BINANCE_SYMBOL_OVERRIDES_JSON")),
            timeout_seconds=_parse_int(_get_env("BINANCE_TIMEOUT_SECONDS"), default=20),
            pause_seconds=_parse_float(_get_env("BINANCE_PAUSE_SECONDS"), default=0.0),
        )

        signal = SignalSettings(
            ema_short_n=tuple(int(x.strip()) for x in (_get_env("EMA_SHORT_N", "8,16,32") or "8,16,32").split(",")),
            ema_long_n=tuple(int(x.strip()) for x in (_get_env("EMA_LONG_N", "24,48,96") or "24,48,96").split(",")),
            short_norm_hours=_parse_int(_get_env("SHORT_NORM_HOURS"), default=12),
            long_norm_hours=_parse_int(_get_env("LONG_NORM_HOURS"), default=168),
            rolling_std_ddof=_parse_int(_get_env("ROLLING_STD_DDOF"), default=1),
            signal_threshold=_parse_float(_get_env("SIGNAL_THRESHOLD"), default=0.5),
            extra_warmup_hours=_parse_int(_get_env("EXTRA_WARMUP_HOURS"), default=0),
        )

        trading = TradingSettings(
            active_pairs=_parse_csv_tuple(_get_env("ACTIVE_COINS"), default=DEFAULT_ACTIVE_PAIRS),
            bet_size_usd=_parse_float(_get_env("BET_SIZE_USD"), default=1_000.0),
            cash_reserve_usd=_parse_float(_get_env("CASH_RESERVE_USD"), default=0.0),
            execute_trades=_parse_bool(_get_env("EXECUTE_TRADES"), default=False),
            run_forever=_parse_bool(_get_env("RUN_FOREVER"), default=False),
            bar_close_delay_seconds=_parse_int(_get_env("BAR_CLOSE_DELAY_SECONDS"), default=10),
            loop_poll_seconds=_parse_float(_get_env("LOOP_POLL_SECONDS"), default=5.0),
            estimated_fee_rate=_parse_float(_get_env("ESTIMATED_FEE_RATE"), default=0.001),
            balance_epsilon=_parse_float(_get_env("BALANCE_EPSILON"), default=1e-12),
            usd_asset=_get_env("USD_ASSET", "USD") or "USD",
        )

        logging = LoggingSettings(
            jsonl_path=_get_env("LIVE_LOG_JSONL_PATH", "strategy2_hfm_live_log.jsonl") or "strategy2_hfm_live_log.jsonl",
        )

        settings = cls(
            roostoo=roostoo,
            binance=binance,
            signal=signal,
            trading=trading,
            logging=logging,
        )
        settings.validate()
        return settings
