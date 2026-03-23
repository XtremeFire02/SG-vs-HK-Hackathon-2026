from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import requests


ONE_HOUR_MS = 60 * 60 * 1000
_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "num_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def _to_utc_timestamp(value: str | pd.Timestamp | None) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _normalize_timestamp_ms(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().all():
        raise ValueError("Timestamp column could not be parsed as numeric.")
    median = float(numeric.dropna().median())
    if median > 1e14:
        numeric = numeric / 1000.0
    elif median < 1e11:
        numeric = numeric * 1000.0
    return numeric.round().astype("int64")


def _standardize_kline_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "num_trades",
                "timestamp",
            ]
        )

    frame = df.copy()
    rename_map = {i: col for i, col in enumerate(_KLINE_COLUMNS[: frame.shape[1]])}
    frame = frame.rename(columns=rename_map)

    for missing_col in _KLINE_COLUMNS:
        if missing_col not in frame.columns:
            frame[missing_col] = np.nan

    frame["open_time"] = _normalize_timestamp_ms(frame["open_time"])
    frame["close_time"] = _normalize_timestamp_ms(frame["close_time"])
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume", "num_trades"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["open_time", "open", "high", "low", "close"])
    frame = frame.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    return frame[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "timestamp",
        ]
    ].reset_index(drop=True)


@dataclass(frozen=True)
class BinanceSymbolMapping:
    pair: str
    symbol: str
    quote_asset: str
    method: str


class BinanceSpotMarketDataClient:
    """Small Binance spot public-data client for hourly klines and symbol mapping."""

    def __init__(
        self,
        *,
        base_url: str = "https://data-api.binance.vision",
        timeout: int = 20,
        pause_seconds: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.pause_seconds = float(pause_seconds)
        self.session = requests.Session()

    def _request_json(self, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def get_exchange_info(self) -> Dict[str, Any]:
        payload = self._request_json("/api/v3/exchangeInfo")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Binance exchangeInfo payload: {payload!r}")
        return payload

    def build_spot_symbol_table(self, exchange_info: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
        info = dict(exchange_info or self.get_exchange_info())
        rows = []
        for item in info.get("symbols", []):
            rows.append(
                {
                    "symbol": item.get("symbol"),
                    "base_asset": item.get("baseAsset"),
                    "quote_asset": item.get("quoteAsset"),
                    "status": item.get("status"),
                    "spot_allowed": bool(item.get("isSpotTradingAllowed", False)),
                }
            )
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        table = table[table["spot_allowed"]].copy()
        table = table[table["status"].eq("TRADING")].copy()
        table = table.reset_index(drop=True)
        return table

    def map_roostoo_pairs_to_spot_symbols(
        self,
        pairs: Sequence[str],
        *,
        quote_priority: Sequence[str],
        symbol_overrides: Optional[Mapping[str, str]] = None,
        exchange_info: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, BinanceSymbolMapping]:
        overrides = dict(symbol_overrides or {})
        table = self.build_spot_symbol_table(exchange_info=exchange_info)
        if table.empty:
            raise RuntimeError("Binance exchangeInfo did not contain any tradable spot symbols.")

        by_symbol = set(table["symbol"].dropna().tolist())
        by_base_quote = {
            (str(row["base_asset"]), str(row["quote_asset"])): str(row["symbol"])
            for row in table.to_dict(orient="records")
            if isinstance(row.get("base_asset"), str) and isinstance(row.get("quote_asset"), str)
        }

        output: dict[str, BinanceSymbolMapping] = {}
        for pair in pairs:
            base_asset, _, _ = pair.partition("/")
            if pair in overrides:
                candidate = overrides[pair]
                if candidate not in by_symbol:
                    raise KeyError(f"Manual Binance symbol override {candidate!r} for {pair} was not found.")
                matched_quote = table.loc[table["symbol"].eq(candidate), "quote_asset"].iloc[0]
                output[pair] = BinanceSymbolMapping(
                    pair=pair,
                    symbol=candidate,
                    quote_asset=str(matched_quote),
                    method="manual_override",
                )
                continue

            matched_symbol: Optional[str] = None
            matched_quote: Optional[str] = None
            method = ""
            for quote_asset in quote_priority:
                candidate = by_base_quote.get((base_asset, quote_asset))
                if candidate is not None:
                    matched_symbol = candidate
                    matched_quote = quote_asset
                    method = f"heuristic_{quote_asset}"
                    break

            if matched_symbol is None:
                raise KeyError(
                    f"Could not map {pair} to a Binance spot symbol using quote_priority={tuple(quote_priority)}."
                )

            output[pair] = BinanceSymbolMapping(
                pair=pair,
                symbol=matched_symbol,
                quote_asset=str(matched_quote),
                method=method,
            )

        return output

    def fetch_klines(
        self,
        *,
        symbol: str,
        interval: str,
        start_open: pd.Timestamp,
        end_open: pd.Timestamp,
    ) -> pd.DataFrame:
        if start_open > end_open:
            return _standardize_kline_frame(pd.DataFrame(columns=range(12)))

        rows: list[list[Any]] = []
        current_start_ms = int(start_open.timestamp() * 1000)
        end_open_ms = int(end_open.timestamp() * 1000)

        while current_start_ms <= end_open_ms:
            remaining_bars = int(((end_open_ms - current_start_ms) / ONE_HOUR_MS) + 1)
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start_ms,
                "endTime": end_open_ms + ONE_HOUR_MS - 1,
                "limit": min(1000, remaining_bars),
            }
            batch = self._request_json("/api/v3/klines", params=params)
            if not isinstance(batch, list):
                raise RuntimeError(f"Unexpected Binance klines payload: {batch!r}")
            if not batch:
                break

            rows.extend(batch)
            last_open = int(batch[-1][0])
            next_start_ms = last_open + ONE_HOUR_MS
            if next_start_ms <= current_start_ms:
                break
            current_start_ms = next_start_ms

            if len(batch) < params["limit"]:
                break
            if self.pause_seconds > 0:
                time.sleep(self.pause_seconds)

        return _standardize_kline_frame(pd.DataFrame(rows))

    @staticmethod
    def last_closed_hour_open(as_of: pd.Timestamp | None = None) -> pd.Timestamp:
        now = _to_utc_timestamp(as_of) or pd.Timestamp.now(tz="UTC")
        now = now.tz_convert("UTC") if now.tzinfo is not None else now.tz_localize("UTC")
        return now.floor("h") - pd.Timedelta(hours=1)

    def fetch_recent_closed_panel(
        self,
        pair_to_symbol: Mapping[str, BinanceSymbolMapping | str],
        *,
        bars: int,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if bars <= 0:
            raise ValueError("bars must be positive.")

        last_closed_open = self.last_closed_hour_open(as_of=as_of)
        start_open = last_closed_open - pd.Timedelta(hours=bars - 1)

        close_frames: dict[str, pd.Series] = {}
        for pair, mapping in pair_to_symbol.items():
            symbol = mapping.symbol if isinstance(mapping, BinanceSymbolMapping) else str(mapping)
            history = self.fetch_klines(
                symbol=symbol,
                interval="1h",
                start_open=start_open,
                end_open=last_closed_open,
            )
            if history.empty:
                raise RuntimeError(f"No Binance 1h history was returned for {pair} ({symbol}).")

            history = history.set_index("timestamp").sort_index()
            close_frames[pair] = history["close"].astype("float64")

        close_panel = pd.DataFrame(close_frames).sort_index()
        if close_panel.empty:
            raise RuntimeError("The Binance close panel is empty.")
        if len(close_panel) < bars:
            raise RuntimeError(
                f"Expected at least {bars} closed 1h bars but only received {len(close_panel)}."
            )
        return close_panel.iloc[-bars:].copy()
