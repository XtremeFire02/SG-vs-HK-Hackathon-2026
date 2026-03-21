from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from .config import DEFAULT_BINANCE_QUOTE_PRIORITY, Strategy2DataConfig, Strategy2ExecutionConfig


BINANCE_DATA_API_BASE = "https://data-api.binance.vision/api/v3"
BINANCE_ARCHIVE_BASE = "https://data.binance.vision/data/spot"
ROOSTOO_DEFAULT_BASE_URL = "https://mock-api.roostoo.com"
COINPAPRIKA_BASE_URL = "https://api.coinpaprika.com/v1"
ONE_HOUR_MS = 60 * 60 * 1000
_BINANCE_KLINE_COLUMNS = [
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


@dataclass
class Strategy2MarketDataBundle:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    quote_volume: pd.DataFrame
    universe: pd.DataFrame
    market_cap: Optional[pd.DataFrame] = None
    market_cap_metadata: Optional[pd.DataFrame] = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _to_utc_timestamp(value: Optional[str | pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("h")


def resolve_strategy2_backtest_range(data_config: Strategy2DataConfig) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive hourly open timestamps for the requested backtest range."""

    now_hour = pd.Timestamp.now(tz="UTC").floor("h")
    default_end = now_hour - pd.Timedelta(hours=1)
    end_open = _to_utc_timestamp(data_config.end_date) or default_end
    start_open = _to_utc_timestamp(data_config.start_date)
    if start_open is None:
        start_open = (end_open - pd.DateOffset(years=5)).floor("h")
    if start_open > end_open:
        raise ValueError("start_date must be earlier than end_date.")
    return start_open, end_open


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().all():
        raise ValueError("Timestamp column could not be parsed.")

    median = float(values.dropna().median())
    if median > 1e14:
        values = values / 1000.0  # microseconds -> milliseconds
    elif median < 1e11:
        values = values * 1000.0  # seconds -> milliseconds

    return values.round().astype("int64")


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        ms = _normalize_timestamp_series(series)
        return pd.to_datetime(ms, unit="ms", utc=True).floor("h")

    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().all():
        raise ValueError("Could not parse datetime-like values from the market-cap CSV.")
    return parsed.dt.floor("h")


def _standardize_binance_kline_frame(df: pd.DataFrame) -> pd.DataFrame:
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
    if frame.shape[1] < 6:
        raise ValueError("Unexpected Binance kline shape; expected at least 6 columns.")

    rename_map = {i: col for i, col in enumerate(_BINANCE_KLINE_COLUMNS[: frame.shape[1]])}
    frame = frame.rename(columns=rename_map)

    for missing_col in _BINANCE_KLINE_COLUMNS:
        if missing_col not in frame.columns:
            frame[missing_col] = np.nan

    frame["open_time"] = _normalize_timestamp_series(frame["open_time"])
    if frame["close_time"].isna().all():
        frame["close_time"] = frame["open_time"] + ONE_HOUR_MS - 1
    else:
        frame["close_time"] = _normalize_timestamp_series(frame["close_time"])

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "num_trades",
    ]
    for col in numeric_cols:
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


def _month_starts_between(start_open: pd.Timestamp, end_open: pd.Timestamp) -> List[pd.Timestamp]:
    month_start = start_open.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_month = end_open.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months: List[pd.Timestamp] = []
    current = month_start
    while current <= end_month:
        months.append(current)
        current = current + pd.DateOffset(months=1)
    return months


def fetch_roostoo_exchange_info(
    base_url: str = ROOSTOO_DEFAULT_BASE_URL,
    timeout: int = 20,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    client = session or requests.Session()
    url = f"{base_url.rstrip('/')}/v3/exchangeInfo"
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_or_load_roostoo_exchange_info(
    data_config: Strategy2DataConfig,
    base_url: str = ROOSTOO_DEFAULT_BASE_URL,
) -> Dict[str, Any]:
    cache_path = Path(data_config.cache_dir) / "metadata" / "roostoo_exchange_info.json"
    if cache_path.exists() and not data_config.refresh_roostoo_exchange_info:
        return _read_json(cache_path)

    payload = fetch_roostoo_exchange_info(
        base_url=base_url,
        timeout=data_config.request_timeout_seconds,
    )
    _write_json(cache_path, payload)
    return payload


def parse_roostoo_universe(exchange_info: Mapping[str, Any], quote_asset: str = "USD") -> pd.DataFrame:
    trade_pairs = exchange_info.get("TradePairs", {})
    rows: List[Dict[str, Any]] = []
    for pair, rules in trade_pairs.items():
        if not isinstance(pair, str) or "/" not in pair:
            continue
        base_asset, quote = pair.split("/", 1)
        rows.append(
            {
                "pair": pair,
                "base_asset": base_asset,
                "quote_asset": quote,
                "can_trade": bool(rules.get("CanTrade", False)),
                "price_precision": int(rules.get("PricePrecision", 0) or 0),
                "amount_precision": int(rules.get("AmountPrecision", 0) or 0),
                "mini_order": float(rules.get("MiniOrder", 0.0) or 0.0),
            }
        )

    universe = pd.DataFrame(rows)
    if universe.empty:
        return universe

    universe = universe[universe["quote_asset"].eq(quote_asset)].copy()
    universe = universe[universe["can_trade"]].copy()
    universe = universe.sort_values("pair").reset_index(drop=True)
    return universe


def fetch_binance_exchange_info(
    timeout: int = 20,
    session: Optional[requests.Session] = None,
    base_url: str = BINANCE_DATA_API_BASE,
) -> Dict[str, Any]:
    client = session or requests.Session()
    url = f"{base_url.rstrip('/')}/exchangeInfo"
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_or_load_binance_exchange_info(data_config: Strategy2DataConfig) -> Dict[str, Any]:
    cache_path = Path(data_config.cache_dir) / "metadata" / "binance_spot_exchange_info.json"
    if cache_path.exists() and not data_config.refresh_binance_exchange_info:
        return _read_json(cache_path)

    payload = fetch_binance_exchange_info(timeout=data_config.request_timeout_seconds)
    _write_json(cache_path, payload)
    return payload


def build_binance_symbol_table(exchange_info: Mapping[str, Any]) -> pd.DataFrame:
    symbols = exchange_info.get("symbols", [])
    rows: List[Dict[str, Any]] = []
    for item in symbols:
        rows.append(
            {
                "symbol": item.get("symbol"),
                "base_asset": item.get("baseAsset"),
                "quote_asset": item.get("quoteAsset"),
                "status": item.get("status"),
                "spot_allowed": bool(item.get("isSpotTradingAllowed", False)),
            }
        )
    return pd.DataFrame(rows)


def map_roostoo_to_binance_symbols(
    roostoo_universe: pd.DataFrame,
    binance_symbol_table: pd.DataFrame,
    quote_priority: Sequence[str] = DEFAULT_BINANCE_QUOTE_PRIORITY,
    overrides: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    overrides = dict(overrides or {})
    available = binance_symbol_table.copy()
    if not available.empty:
        available = available[available["spot_allowed"]].copy()
        available = available[available["status"].eq("TRADING")].copy()

    by_symbol = set(available["symbol"].dropna().tolist())
    symbol_meta: Dict[str, Dict[str, Any]] = {
        row["symbol"]: row for row in available.to_dict(orient="records") if isinstance(row.get("symbol"), str)
    }

    rows: List[Dict[str, Any]] = []
    for row in roostoo_universe.to_dict(orient="records"):
        pair = row["pair"]
        base_asset = row["base_asset"]
        mapped_symbol: Optional[str] = None
        mapped_quote: Optional[str] = None
        mapping_method = "unmapped"

        if pair in overrides:
            candidate = overrides[pair]
            if candidate in by_symbol:
                mapped_symbol = candidate
                mapped_quote = symbol_meta[candidate]["quote_asset"]
                mapping_method = "manual_override"
        else:
            for quote in quote_priority:
                candidate = f"{base_asset}{quote}"
                if candidate in by_symbol:
                    mapped_symbol = candidate
                    mapped_quote = quote
                    mapping_method = f"heuristic_{quote}"
                    break

        rows.append(
            {
                **row,
                "binance_symbol": mapped_symbol,
                "binance_quote_asset": mapped_quote,
                "mapping_found": mapped_symbol is not None,
                "mapping_method": mapping_method,
            }
        )

    return pd.DataFrame(rows)


def _market_cap_features_requested(data_config: Strategy2DataConfig) -> bool:
    return bool(data_config.include_market_cap or data_config.market_cap_csv_path)


def fetch_coinpaprika_tickers(
    timeout: int = 20,
    session: Optional[requests.Session] = None,
    base_url: str = COINPAPRIKA_BASE_URL,
) -> List[Dict[str, Any]]:
    client = session or requests.Session()
    url = f"{base_url.rstrip('/')}/tickers"
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected CoinPaprika response: {payload}")
    return payload


def fetch_or_load_coinpaprika_tickers(data_config: Strategy2DataConfig) -> List[Dict[str, Any]]:
    cache_path = Path(data_config.cache_dir) / "metadata" / "coinpaprika_tickers.json"
    if cache_path.exists() and not data_config.refresh_market_cap_cache:
        payload = _read_json(cache_path)
        if isinstance(payload, list):
            return payload

    payload = fetch_coinpaprika_tickers(timeout=data_config.request_timeout_seconds)
    _write_json(cache_path, payload)
    return payload


def build_coinpaprika_market_cap_table(tickers: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in tickers:
        usd_quote = (item.get("quotes") or {}).get("USD") or {}
        price = _safe_float(usd_quote.get("price"), default=np.nan)
        market_cap = _safe_float(usd_quote.get("market_cap"), default=np.nan)
        approx_supply = np.nan
        if pd.notna(price) and price > 0 and pd.notna(market_cap):
            approx_supply = market_cap / price

        rows.append(
            {
                "coin_id": item.get("id"),
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "rank": _safe_float(item.get("rank"), default=np.nan),
                "current_price_usd": price,
                "market_cap_usd": market_cap,
                "approx_circulating_supply": approx_supply,
                "last_updated": item.get("last_updated"),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["symbol_norm"] = out["symbol"].astype(str).str.strip().str.lower()
    out["rank_sort"] = out["rank"].fillna(np.inf)
    out = out.sort_values(
        ["symbol_norm", "rank_sort", "market_cap_usd"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return out


def map_roostoo_to_market_cap_metadata(
    roostoo_universe: pd.DataFrame,
    market_cap_table: pd.DataFrame,
    overrides: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    overrides = dict(overrides or {})
    by_coin_id = {
        row["coin_id"]: row
        for row in market_cap_table.to_dict(orient="records")
        if isinstance(row.get("coin_id"), str)
    }
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for row in market_cap_table.to_dict(orient="records"):
        symbol_norm = str(row.get("symbol_norm") or "").strip().lower()
        if not symbol_norm:
            continue
        by_symbol.setdefault(symbol_norm, []).append(row)

    rows: List[Dict[str, Any]] = []
    for row in roostoo_universe.to_dict(orient="records"):
        pair = row["pair"]
        base_asset = str(row.get("base_asset") or "").strip().lower()
        chosen: Optional[Dict[str, Any]] = None
        mapping_method = "unmapped"

        if pair in overrides and overrides[pair] in by_coin_id:
            chosen = by_coin_id[overrides[pair]]
            mapping_method = "manual_override"
        else:
            candidates = by_symbol.get(base_asset, [])
            if candidates:
                chosen = candidates[0]
                mapping_method = "symbol_top_market_cap"

        rows.append(
            {
                "pair": pair,
                "market_cap_coin_id": chosen.get("coin_id") if chosen else None,
                "market_cap_name": chosen.get("name") if chosen else None,
                "market_cap_rank": chosen.get("rank") if chosen else np.nan,
                "market_cap_snapshot_usd": chosen.get("market_cap_usd") if chosen else np.nan,
                "approx_circulating_supply": chosen.get("approx_circulating_supply") if chosen else np.nan,
                "market_cap_mapping_found": chosen is not None,
                "market_cap_mapping_method": mapping_method,
            }
        )

    return pd.DataFrame(rows)


def _alias_lookup_for_market_cap_csv(universe_table: pd.DataFrame) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for row in universe_table.to_dict(orient="records"):
        pair = row["pair"]
        aliases = {
            str(row.get("pair") or "").strip().lower(),
            str(row.get("base_asset") or "").strip().lower(),
            str(row.get("binance_symbol") or "").strip().lower(),
        }
        aliases.discard("")
        for alias in aliases:
            lookup.setdefault(alias, pair)
    return lookup


def load_market_cap_frame_from_csv(
    csv_path: str,
    index: pd.DatetimeIndex,
    universe_table: pd.DataFrame,
) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Market-cap CSV not found: {csv_path}")

    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(index=index, columns=universe_table["pair"].tolist(), dtype="float64")

    normalized_cols = {str(col).strip().lower(): col for col in raw.columns}
    ts_candidates = ["timestamp", "date", "datetime", "time", "open_time"]
    ts_key = next((name for name in ts_candidates if name in normalized_cols), None)
    if ts_key is None:
        raise ValueError("Market-cap CSV must contain a timestamp/date column.")

    ts_col = normalized_cols[ts_key]
    raw = raw.copy()
    raw["_timestamp"] = _parse_datetime_series(raw[ts_col])

    value_candidates = ["market_cap", "market_cap_usd", "marketcap", "mktcap", "cap"]
    id_candidates = ["pair", "symbol", "asset", "base_asset", "binance_symbol", "ticker"]

    value_key = next((name for name in value_candidates if name in normalized_cols), None)
    id_key = next((name for name in id_candidates if name in normalized_cols), None)

    if value_key is not None and id_key is not None:
        alias_lookup = _alias_lookup_for_market_cap_csv(universe_table)
        value_col = normalized_cols[value_key]
        id_col = normalized_cols[id_key]
        long_df = raw[["_timestamp", id_col, value_col]].copy()
        long_df["_pair"] = long_df[id_col].astype(str).str.strip().str.lower().map(alias_lookup)
        long_df["_market_cap"] = pd.to_numeric(long_df[value_col], errors="coerce")
        wide = long_df.dropna(subset=["_timestamp", "_pair"]).pivot_table(
            index="_timestamp",
            columns="_pair",
            values="_market_cap",
            aggfunc="last",
        )
    else:
        wide = pd.DataFrame(index=raw["_timestamp"])
        for row in universe_table.to_dict(orient="records"):
            pair = row["pair"]
            candidates = [
                str(row.get("pair") or ""),
                str(row.get("base_asset") or ""),
                str(row.get("binance_symbol") or ""),
            ]
            match_col = None
            for candidate in candidates:
                candidate_norm = candidate.strip().lower()
                if candidate_norm and candidate_norm in normalized_cols and candidate_norm != ts_key:
                    match_col = normalized_cols[candidate_norm]
                    break
            wide[pair] = pd.to_numeric(raw[match_col], errors="coerce") if match_col is not None else np.nan

    wide = wide.groupby(level=0).last().sort_index()
    wide = wide.reindex(index.union(wide.index)).sort_index().ffill().reindex(index)
    wide = wide.reindex(columns=universe_table["pair"].tolist())
    return wide.astype("float64")


def summarize_market_cap_frame(
    universe_table: pd.DataFrame,
    market_cap_frame: pd.DataFrame,
    mapping_method: str,
) -> pd.DataFrame:
    latest = market_cap_frame.ffill().iloc[-1] if not market_cap_frame.empty else pd.Series(dtype="float64")
    rows: List[Dict[str, Any]] = []
    for row in universe_table.to_dict(orient="records"):
        pair = row["pair"]
        latest_cap = latest.get(pair, np.nan)
        rows.append(
            {
                "pair": pair,
                "market_cap_coin_id": None,
                "market_cap_name": None,
                "market_cap_rank": np.nan,
                "market_cap_snapshot_usd": latest_cap,
                "approx_circulating_supply": np.nan,
                "market_cap_mapping_found": pd.notna(latest_cap),
                "market_cap_mapping_method": mapping_method,
            }
        )
    return pd.DataFrame(rows)


def build_market_cap_proxy_frame(close: pd.DataFrame, market_cap_metadata: pd.DataFrame) -> pd.DataFrame:
    if market_cap_metadata.empty:
        return pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")

    supply = (
        market_cap_metadata.set_index("pair")["approx_circulating_supply"]
        .reindex(close.columns)
        .astype("float64")
    )
    supply = supply.where(supply > 0.0)
    return close.astype("float64").mul(supply, axis=1)


def _empty_market_cap_metadata(universe_table: pd.DataFrame, *, mapping_method: str, warning: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for row in universe_table.to_dict(orient="records"):
        rows.append(
            {
                "pair": row["pair"],
                "market_cap_coin_id": None,
                "market_cap_name": None,
                "market_cap_rank": np.nan,
                "market_cap_snapshot_usd": np.nan,
                "approx_circulating_supply": np.nan,
                "market_cap_mapping_found": False,
                "market_cap_mapping_method": mapping_method,
                "market_cap_warning": warning,
            }
        )
    return pd.DataFrame(rows)


def _archive_zip_path(cache_dir: Path, symbol: str, interval: str, month_start: pd.Timestamp) -> Path:
    file_name = f"{symbol}-{interval}-{month_start.strftime('%Y-%m')}.zip"
    return cache_dir / "raw" / "binance" / "monthly" / "klines" / symbol / interval / file_name


def _archive_zip_url(symbol: str, interval: str, month_start: pd.Timestamp) -> str:
    file_name = f"{symbol}-{interval}-{month_start.strftime('%Y-%m')}.zip"
    return f"{BINANCE_ARCHIVE_BASE}/monthly/klines/{symbol}/{interval}/{file_name}"


def _read_zip_csv(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [name for name in zf.namelist() if not name.endswith("/")]
        if not members:
            raise RuntimeError(f"Zip archive is empty: {zip_path}")
        with zf.open(members[0]) as handle:
            df = pd.read_csv(handle, header=None)
    return _standardize_binance_kline_frame(df)


def _download_archive_if_needed(
    symbol: str,
    interval: str,
    month_start: pd.Timestamp,
    cache_dir: Path,
    session: requests.Session,
    timeout: int,
) -> Optional[Path]:
    target_path = _archive_zip_path(cache_dir, symbol, interval, month_start)
    if target_path.exists():
        return target_path

    url = _archive_zip_url(symbol, interval, month_start)
    response = session.get(url, timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    _ensure_parent(target_path)
    with open(target_path, "wb") as handle:
        handle.write(response.content)
    return target_path


def fetch_binance_klines_rest(
    symbol: str,
    interval: str,
    start_open: pd.Timestamp,
    end_open: pd.Timestamp,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
    pause_seconds: float = 0.0,
    base_url: str = BINANCE_DATA_API_BASE,
) -> pd.DataFrame:
    if start_open > end_open:
        return _standardize_binance_kline_frame(pd.DataFrame(columns=range(12)))

    client = session or requests.Session()
    rows: List[List[Any]] = []
    current_start_ms = int(start_open.timestamp() * 1000)
    end_open_ms = int(end_open.timestamp() * 1000)

    while current_start_ms <= end_open_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start_ms,
            "endTime": end_open_ms + ONE_HOUR_MS - 1,
            "limit": 1000,
        }
        response = client.get(f"{base_url.rstrip('/')}/klines", params=params, timeout=timeout)
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected Binance REST response: {batch}")
        if not batch:
            break

        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_start = last_open + ONE_HOUR_MS
        if next_start <= current_start_ms:
            break
        current_start_ms = next_start

        if len(batch) < 1000:
            break
        if pause_seconds > 0:
            import time

            time.sleep(pause_seconds)

    if not rows:
        return _standardize_binance_kline_frame(pd.DataFrame(columns=range(12)))

    df = pd.DataFrame(rows)
    return _standardize_binance_kline_frame(df)


def load_symbol_history(
    symbol: str,
    data_config: Strategy2DataConfig,
    start_open: pd.Timestamp,
    end_open: pd.Timestamp,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Load one Binance symbol's hourly history, caching a standardized CSV locally."""

    if data_config.interval != "1h":
        raise ValueError("The Strategy 2 implementation currently supports interval='1h' only.")

    cache_dir = Path(data_config.cache_dir)
    processed_path = cache_dir / "processed" / f"{symbol}_{data_config.interval}.csv"

    if processed_path.exists() and not data_config.refresh_history:
        cached = pd.read_csv(processed_path)
        cached = _standardize_binance_kline_frame(cached)
        if not cached.empty:
            cached_min = cached["timestamp"].min()
            cached_max = cached["timestamp"].max()
            if cached_min <= start_open and cached_max >= end_open:
                return cached[
                    (cached["timestamp"] >= start_open) & (cached["timestamp"] <= end_open)
                ].reset_index(drop=True)

    client = session or requests.Session()
    frames: List[pd.DataFrame] = []

    if data_config.use_archive_downloads:
        for month_start in _month_starts_between(start_open, end_open):
            zip_path = _download_archive_if_needed(
                symbol=symbol,
                interval=data_config.interval,
                month_start=month_start,
                cache_dir=cache_dir,
                session=client,
                timeout=data_config.request_timeout_seconds,
            )
            if zip_path is None:
                continue
            try:
                frames.append(_read_zip_csv(zip_path))
            except zipfile.BadZipFile:
                zip_path.unlink(missing_ok=True)
                retry_path = _download_archive_if_needed(
                    symbol=symbol,
                    interval=data_config.interval,
                    month_start=month_start,
                    cache_dir=cache_dir,
                    session=client,
                    timeout=data_config.request_timeout_seconds,
                )
                if retry_path is not None:
                    frames.append(_read_zip_csv(retry_path))

    history = (
        pd.concat(frames, ignore_index=True)
        if frames
        else _standardize_binance_kline_frame(pd.DataFrame(columns=range(12)))
    )

    if data_config.include_rest_topup:
        if history.empty:
            rest_start = start_open
        else:
            rest_start = history["timestamp"].max() + pd.Timedelta(hours=1)
            rest_start = max(rest_start, start_open)

        if rest_start <= end_open:
            rest = fetch_binance_klines_rest(
                symbol=symbol,
                interval=data_config.interval,
                start_open=rest_start,
                end_open=end_open,
                session=client,
                timeout=data_config.request_timeout_seconds,
                pause_seconds=data_config.request_pause_seconds,
            )
            if not rest.empty:
                history = pd.concat([history, rest], ignore_index=True)

    history = _standardize_binance_kline_frame(history)
    if not history.empty:
        _ensure_parent(processed_path)
        history.to_csv(processed_path, index=False)

    return history[(history["timestamp"] >= start_open) & (history["timestamp"] <= end_open)].reset_index(drop=True)


def load_histories_for_universe(
    mapped_universe: pd.DataFrame,
    data_config: Strategy2DataConfig,
) -> Dict[str, pd.DataFrame]:
    start_open, end_open = resolve_strategy2_backtest_range(data_config)
    histories: Dict[str, pd.DataFrame] = {}
    session = requests.Session()

    for row in mapped_universe.to_dict(orient="records"):
        pair = row["pair"]
        symbol = row.get("binance_symbol")
        if not symbol:
            histories[pair] = _standardize_binance_kline_frame(pd.DataFrame(columns=range(12)))
            continue

        histories[pair] = load_symbol_history(
            symbol=symbol,
            data_config=data_config,
            start_open=start_open,
            end_open=end_open,
            session=session,
        )

    return histories


def summarize_universe_coverage(
    mapped_universe: pd.DataFrame,
    histories: Mapping[str, pd.DataFrame],
    data_config: Strategy2DataConfig,
    execution_config: Strategy2ExecutionConfig,
) -> pd.DataFrame:
    start_open, end_open = resolve_strategy2_backtest_range(data_config)
    expected_hours = int(((end_open - start_open) / pd.Timedelta(hours=1)) + 1)

    rows: List[Dict[str, Any]] = []
    for row in mapped_universe.to_dict(orient="records"):
        pair = row["pair"]
        hist = histories.get(pair)
        if hist is None or hist.empty:
            history_hours = 0
            history_days = 0.0
            missing_ratio = 1.0
            start_ts = pd.NaT
            end_ts = pd.NaT
        else:
            history_hours = int(hist["timestamp"].nunique())
            history_days = history_hours / 24.0
            start_ts = hist["timestamp"].min()
            end_ts = hist["timestamp"].max()
            missing_ratio = 1.0 - min(1.0, history_hours / max(1, expected_hours))

        mini_order_ok = float(row.get("mini_order", 0.0) or 0.0) <= execution_config.initial_cash * execution_config.mini_order_nav_fraction
        enough_history = history_days >= data_config.min_history_days
        mapping_found = bool(row.get("mapping_found", False))
        passes_static_filter = bool(mapping_found and mini_order_ok and enough_history)

        rows.append(
            {
                **row,
                "history_start": start_ts,
                "history_end": end_ts,
                "history_hours": history_hours,
                "history_days": history_days,
                "full_sample_missing_ratio": missing_ratio,
                "mini_order_ok": mini_order_ok,
                "enough_history": enough_history,
                "passes_static_filter": passes_static_filter,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["passes_static_filter", "pair"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_market_data_bundle(
    universe_table: pd.DataFrame,
    histories: Mapping[str, pd.DataFrame],
    data_config: Strategy2DataConfig,
    market_cap: Optional[pd.DataFrame] = None,
    market_cap_metadata: Optional[pd.DataFrame] = None,
) -> Strategy2MarketDataBundle:
    start_open, end_open = resolve_strategy2_backtest_range(data_config)
    index = pd.date_range(start=start_open, end=end_open, freq="1h", tz="UTC")

    def wide_field(field: str) -> pd.DataFrame:
        data: Dict[str, pd.Series] = {}
        for row in universe_table.to_dict(orient="records"):
            pair = row["pair"]
            hist = histories.get(pair)
            if hist is None or hist.empty:
                data[pair] = pd.Series(index=index, dtype="float64")
                continue
            series = hist.set_index("timestamp")[field].astype("float64").reindex(index)
            data[pair] = series
        return pd.DataFrame(data, index=index, dtype="float64")

    ordered_pairs = universe_table["pair"].tolist()
    market_cap_frame = None
    if market_cap is not None:
        market_cap_frame = market_cap.reindex(index=index, columns=ordered_pairs).astype("float64")

    return Strategy2MarketDataBundle(
        open=wide_field("open"),
        high=wide_field("high"),
        low=wide_field("low"),
        close=wide_field("close"),
        volume=wide_field("volume"),
        quote_volume=wide_field("quote_asset_volume"),
        universe=universe_table.copy().reset_index(drop=True),
        market_cap=market_cap_frame,
        market_cap_metadata=None if market_cap_metadata is None else market_cap_metadata.copy().reset_index(drop=True),
    )


def prepare_strategy2_market_data_environment(
    data_config: Strategy2DataConfig,
    execution_config: Strategy2ExecutionConfig,
    *,
    roostoo_base_url: str = ROOSTOO_DEFAULT_BASE_URL,
    symbol_overrides: Optional[Mapping[str, str]] = None,
    market_cap_overrides: Optional[Mapping[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Strategy2MarketDataBundle]:
    """Build the raw Strategy 2 market-data environment without using phase2_backtest.py."""

    roostoo_exchange_info = fetch_or_load_roostoo_exchange_info(data_config, base_url=roostoo_base_url)
    roostoo_universe = parse_roostoo_universe(roostoo_exchange_info, quote_asset=data_config.roostoo_quote_asset)

    binance_exchange_info = fetch_or_load_binance_exchange_info(data_config)
    binance_symbol_table = build_binance_symbol_table(binance_exchange_info)

    mapped = map_roostoo_to_binance_symbols(
        roostoo_universe,
        binance_symbol_table,
        quote_priority=data_config.binance_quote_priority,
        overrides=symbol_overrides,
    )

    histories = load_histories_for_universe(mapped, data_config)
    summary = summarize_universe_coverage(mapped, histories, data_config, execution_config)
    filtered_universe = summary[summary["passes_static_filter"]].copy().reset_index(drop=True)

    market_cap_frame: Optional[pd.DataFrame] = None
    market_cap_metadata: Optional[pd.DataFrame] = None

    temp_bundle = build_market_data_bundle(filtered_universe, histories, data_config)

    if _market_cap_features_requested(data_config):
        if data_config.market_cap_csv_path:
            market_cap_frame = load_market_cap_frame_from_csv(
                csv_path=data_config.market_cap_csv_path,
                index=temp_bundle.close.index,
                universe_table=filtered_universe,
            )
            market_cap_metadata = summarize_market_cap_frame(filtered_universe, market_cap_frame, mapping_method="csv")
            summary = summary.merge(market_cap_metadata, on="pair", how="left")
        else:
            try:
                tickers = fetch_or_load_coinpaprika_tickers(data_config)
                market_cap_table = build_coinpaprika_market_cap_table(tickers)
                market_cap_metadata_all = map_roostoo_to_market_cap_metadata(
                    summary,
                    market_cap_table,
                    overrides=market_cap_overrides,
                )
                summary = summary.merge(market_cap_metadata_all, on="pair", how="left")
                market_cap_metadata = market_cap_metadata_all[
                    market_cap_metadata_all["pair"].isin(filtered_universe["pair"])
                ].copy().reset_index(drop=True)
                market_cap_frame = build_market_cap_proxy_frame(temp_bundle.close, market_cap_metadata)
            except Exception as exc:  # pragma: no cover - network failures are environment-dependent.
                if not data_config.allow_missing_market_cap:
                    raise
                market_cap_metadata = _empty_market_cap_metadata(
                    filtered_universe,
                    mapping_method="unavailable",
                    warning=str(exc),
                )
                summary = summary.merge(market_cap_metadata, on="pair", how="left")
                market_cap_frame = None

    data_bundle = build_market_data_bundle(
        filtered_universe,
        histories,
        data_config,
        market_cap=market_cap_frame,
        market_cap_metadata=market_cap_metadata,
    )
    return summary, histories, data_bundle


__all__ = [
    "DEFAULT_BINANCE_QUOTE_PRIORITY",
    "ROOSTOO_DEFAULT_BASE_URL",
    "Strategy2MarketDataBundle",
    "build_binance_symbol_table",
    "build_coinpaprika_market_cap_table",
    "build_market_cap_proxy_frame",
    "build_market_data_bundle",
    "fetch_binance_exchange_info",
    "fetch_coinpaprika_tickers",
    "fetch_or_load_binance_exchange_info",
    "fetch_or_load_coinpaprika_tickers",
    "fetch_or_load_roostoo_exchange_info",
    "fetch_roostoo_exchange_info",
    "fetch_binance_klines_rest",
    "load_histories_for_universe",
    "load_market_cap_frame_from_csv",
    "load_symbol_history",
    "map_roostoo_to_binance_symbols",
    "map_roostoo_to_market_cap_metadata",
    "parse_roostoo_universe",
    "prepare_strategy2_market_data_environment",
    "resolve_strategy2_backtest_range",
    "summarize_market_cap_frame",
    "summarize_universe_coverage",
]
