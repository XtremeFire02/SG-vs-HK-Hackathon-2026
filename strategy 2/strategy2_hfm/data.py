from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Strategy2Config
from .market_data_pipeline import (
    ROOSTOO_DEFAULT_BASE_URL,
    Strategy2MarketDataBundle,
    prepare_strategy2_market_data_environment,
)


_REQUIRED_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")


def _to_utc_timestamp(value: Optional[str | pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("h")


def _slice_history_frame(
    history: pd.DataFrame,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if history is None or history.empty:
        return history.copy()

    frame = history.copy()
    if "timestamp" not in frame.columns:
        return frame

    ts = pd.to_datetime(frame["timestamp"], utc=True)
    mask = pd.Series(True, index=frame.index, dtype=bool)
    if start is not None:
        mask &= ts >= start
    if end is not None:
        mask &= ts <= end
    return frame.loc[mask].reset_index(drop=True)


@dataclass
class Strategy2PreparedData:
    """Cleaned and aligned market data for the Strategy 2 backtest."""

    summary_table: pd.DataFrame
    histories: Dict[str, pd.DataFrame]
    universe_table: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    quote_volume: pd.DataFrame
    market_cap: Optional[pd.DataFrame] = None
    market_cap_metadata: Optional[pd.DataFrame] = None
    source_config: Optional[Strategy2Config] = None

    @property
    def pairs(self) -> pd.Index:
        return self.close.columns

    @property
    def start_timestamp(self) -> pd.Timestamp:
        return self.close.index.min()

    @property
    def end_timestamp(self) -> pd.Timestamp:
        return self.close.index.max()

    def copy(self) -> "Strategy2PreparedData":
        return Strategy2PreparedData(
            summary_table=self.summary_table.copy(),
            histories={k: v.copy() for k, v in self.histories.items()},
            universe_table=self.universe_table.copy(),
            open=self.open.copy(),
            high=self.high.copy(),
            low=self.low.copy(),
            close=self.close.copy(),
            volume=self.volume.copy(),
            quote_volume=self.quote_volume.copy(),
            market_cap=None if self.market_cap is None else self.market_cap.copy(),
            market_cap_metadata=None if self.market_cap_metadata is None else self.market_cap_metadata.copy(),
            source_config=self.source_config,
        )

    def field_frames(self, *, include_optional: bool = True) -> Dict[str, pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
        }
        if include_optional and self.market_cap is not None:
            frames["market_cap"] = self.market_cap
        return frames

    def select_assets(self, pairs: Sequence[str]) -> "Strategy2PreparedData":
        pairs = list(dict.fromkeys(pairs))
        missing = [pair for pair in pairs if pair not in self.close.columns]
        if missing:
            raise KeyError(f"Unknown asset pair(s): {missing}")

        def subset(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if frame is None:
                return None
            return frame.loc[:, pairs].copy()

        summary = self.summary_table.copy()
        if not summary.empty and "pair" in summary.columns:
            summary = summary[summary["pair"].isin(pairs)].copy().reset_index(drop=True)

        universe = self.universe_table.copy()
        if not universe.empty and "pair" in universe.columns:
            universe = universe[universe["pair"].isin(pairs)].copy().reset_index(drop=True)

        market_cap_metadata = None
        if self.market_cap_metadata is not None:
            market_cap_metadata = self.market_cap_metadata.copy()
            if "pair" in market_cap_metadata.columns:
                market_cap_metadata = market_cap_metadata[
                    market_cap_metadata["pair"].isin(pairs)
                ].copy().reset_index(drop=True)

        histories = {pair: self.histories[pair].copy() for pair in pairs if pair in self.histories}
        return Strategy2PreparedData(
            summary_table=summary,
            histories=histories,
            universe_table=universe,
            open=subset(self.open),
            high=subset(self.high),
            low=subset(self.low),
            close=subset(self.close),
            volume=subset(self.volume),
            quote_volume=subset(self.quote_volume),
            market_cap=subset(self.market_cap),
            market_cap_metadata=market_cap_metadata,
            source_config=self.source_config,
        )

    def slice_time(
        self,
        *,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
    ) -> "Strategy2PreparedData":
        start_ts = _to_utc_timestamp(start)
        end_ts = _to_utc_timestamp(end)

        def subset(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if frame is None:
                return None
            index = frame.index
            mask = pd.Series(True, index=index, dtype=bool)
            if start_ts is not None:
                mask &= index >= start_ts
            if end_ts is not None:
                mask &= index <= end_ts
            return frame.loc[mask].copy()

        histories = {
            pair: _slice_history_frame(history, start=start_ts, end=end_ts)
            for pair, history in self.histories.items()
        }

        return Strategy2PreparedData(
            summary_table=self.summary_table.copy(),
            histories=histories,
            universe_table=self.universe_table.copy(),
            open=subset(self.open),
            high=subset(self.high),
            low=subset(self.low),
            close=subset(self.close),
            volume=subset(self.volume),
            quote_volume=subset(self.quote_volume),
            market_cap=subset(self.market_cap),
            market_cap_metadata=None if self.market_cap_metadata is None else self.market_cap_metadata.copy(),
            source_config=self.source_config,
        )

    def get_market_cap_snapshot(
        self,
        *,
        timestamp: Optional[str | pd.Timestamp] = None,
    ) -> pd.Series:
        if self.market_cap is not None and not self.market_cap.empty:
            if timestamp is None:
                return self.market_cap.ffill().iloc[-1].reindex(self.close.columns)
            ts = _to_utc_timestamp(timestamp)
            eligible = self.market_cap.loc[self.market_cap.index <= ts]
            if eligible.empty:
                raise ValueError("Requested market-cap snapshot is earlier than the prepared panel.")
            return eligible.ffill().iloc[-1].reindex(self.close.columns)

        if self.market_cap_metadata is not None and not self.market_cap_metadata.empty:
            if "pair" not in self.market_cap_metadata.columns or "market_cap_snapshot_usd" not in self.market_cap_metadata.columns:
                raise ValueError("market_cap_metadata does not contain the expected columns.")
            return (
                self.market_cap_metadata.set_index("pair")["market_cap_snapshot_usd"]
                .astype("float64")
                .reindex(self.close.columns)
            )

        raise ValueError("No market-cap information is attached to this prepared data object.")

    def filter_by_market_cap(
        self,
        *,
        min_market_cap_usd: Optional[float] = None,
        max_market_cap_usd: Optional[float] = None,
        timestamp: Optional[str | pd.Timestamp] = None,
    ) -> "Strategy2PreparedData":
        snapshot = self.get_market_cap_snapshot(timestamp=timestamp)
        mask = snapshot.notna()
        if min_market_cap_usd is not None:
            mask &= snapshot >= float(min_market_cap_usd)
        if max_market_cap_usd is not None:
            mask &= snapshot <= float(max_market_cap_usd)
        selected_pairs = snapshot.index[mask].tolist()
        return self.select_assets(selected_pairs)

    def split_insample_oos(
        self,
        *,
        insample_fraction: Optional[float] = None,
        oos_start_date: Optional[str | pd.Timestamp] = None,
        warmup_hours: int = 0,
    ) -> Tuple["Strategy2PreparedData", "Strategy2PreparedData", pd.Timestamp]:
        if self.close.empty:
            raise ValueError("Cannot split an empty prepared data object.")

        if oos_start_date is None and self.source_config is not None:
            oos_start_date = self.source_config.data.oos_start_date
        if insample_fraction is None and self.source_config is not None:
            insample_fraction = self.source_config.data.insample_fraction
        if insample_fraction is None:
            insample_fraction = 0.70

        index = self.close.index
        if oos_start_date is not None:
            oos_start = _to_utc_timestamp(oos_start_date)
        else:
            split_idx = max(1, int(len(index) * float(insample_fraction)))
            split_idx = min(split_idx, len(index) - 1)
            oos_start = index[split_idx]

        if oos_start <= index.min() or oos_start > index.max():
            raise ValueError("oos_start_date must lie strictly inside the prepared data range.")

        warmup_hours = max(0, int(warmup_hours))
        in_sample_end = oos_start - pd.Timedelta(hours=1)
        oos_data_start = max(index.min(), oos_start - pd.Timedelta(hours=warmup_hours))

        in_sample = self.slice_time(end=in_sample_end)
        out_of_sample = self.slice_time(start=oos_data_start)
        return in_sample, out_of_sample, oos_start


def clean_and_align_panel(
    field_frames: Mapping[str, pd.DataFrame],
    *,
    drop_incomplete_timestamps: bool = True,
) -> MutableMapping[str, pd.DataFrame]:
    """Align field frames and optionally drop timestamps with any missing values."""

    if not field_frames:
        raise ValueError("field_frames cannot be empty.")

    base_index = None
    base_columns = None
    for name, frame in field_frames.items():
        if base_index is None:
            base_index = frame.index
            base_columns = frame.columns
            continue
        if not frame.index.equals(base_index):
            raise ValueError(f"Field '{name}' does not share the same timestamp index.")
        if not frame.columns.equals(base_columns):
            raise ValueError(f"Field '{name}' does not share the same column order.")

    cleaned = {name: frame.copy() for name, frame in field_frames.items()}
    if not drop_incomplete_timestamps:
        return cleaned

    common_mask = pd.Series(True, index=base_index, dtype=bool)
    for frame in cleaned.values():
        common_mask &= frame.notna().all(axis=1)

    if not bool(common_mask.any()):
        raise RuntimeError("No complete common timestamps remain after alignment.")

    return {name: frame.loc[common_mask].copy() for name, frame in cleaned.items()}


def prepare_strategy2_environment(
    config: Strategy2Config,
    roostoo_base_url: str = ROOSTOO_DEFAULT_BASE_URL,
    symbol_overrides: Optional[Mapping[str, str]] = None,
    market_cap_overrides: Optional[Mapping[str, str]] = None,
) -> Strategy2PreparedData:
    """Build the Strategy 2 prepared dataset without depending on phase2_backtest.py."""

    config.validate()
    summary_table, histories, bundle = prepare_strategy2_market_data_environment(
        data_config=config.data,
        execution_config=config.execution,
        roostoo_base_url=roostoo_base_url,
        symbol_overrides=symbol_overrides,
        market_cap_overrides=market_cap_overrides,
    )

    cleaned = clean_and_align_panel(
        {
            "open": bundle.open,
            "high": bundle.high,
            "low": bundle.low,
            "close": bundle.close,
            "volume": bundle.volume,
            "quote_volume": bundle.quote_volume,
        },
        drop_incomplete_timestamps=config.data.drop_incomplete_timestamps,
    )

    market_cap = None
    if bundle.market_cap is not None:
        market_cap = bundle.market_cap.reindex(
            index=cleaned["close"].index,
            columns=cleaned["close"].columns,
        ).astype("float64")

    prepared = Strategy2PreparedData(
        summary_table=summary_table.copy(),
        histories={k: v.copy() for k, v in histories.items()},
        universe_table=bundle.universe.copy(),
        open=cleaned["open"],
        high=cleaned["high"],
        low=cleaned["low"],
        close=cleaned["close"],
        volume=cleaned["volume"],
        quote_volume=cleaned["quote_volume"],
        market_cap=market_cap,
        market_cap_metadata=None if bundle.market_cap_metadata is None else bundle.market_cap_metadata.copy(),
        source_config=config,
    )
    return prepared


def filter_prepared_data(
    prepared: Strategy2PreparedData,
    *,
    pairs: Optional[Sequence[str]] = None,
    start: Optional[str | pd.Timestamp] = None,
    end: Optional[str | pd.Timestamp] = None,
    min_market_cap_usd: Optional[float] = None,
    max_market_cap_usd: Optional[float] = None,
    market_cap_timestamp: Optional[str | pd.Timestamp] = None,
) -> Strategy2PreparedData:
    out = prepared
    if pairs is not None:
        out = out.select_assets(pairs)
    if start is not None or end is not None:
        out = out.slice_time(start=start, end=end)
    if min_market_cap_usd is not None or max_market_cap_usd is not None:
        out = out.filter_by_market_cap(
            min_market_cap_usd=min_market_cap_usd,
            max_market_cap_usd=max_market_cap_usd,
            timestamp=market_cap_timestamp,
        )
    return out


def split_prepared_data_insample_oos(
    prepared: Strategy2PreparedData,
    *,
    insample_fraction: Optional[float] = None,
    oos_start_date: Optional[str | pd.Timestamp] = None,
    warmup_hours: int = 0,
) -> Tuple[Strategy2PreparedData, Strategy2PreparedData, pd.Timestamp]:
    return prepared.split_insample_oos(
        insample_fraction=insample_fraction,
        oos_start_date=oos_start_date,
        warmup_hours=warmup_hours,
    )


def build_panel_quality_table(prepared: Strategy2PreparedData) -> pd.DataFrame:
    market_cap_snapshot = None
    market_cap_rank = None
    market_cap_name = None
    if prepared.market_cap is not None or prepared.market_cap_metadata is not None:
        try:
            market_cap_snapshot = prepared.get_market_cap_snapshot()
        except ValueError:
            market_cap_snapshot = None
    if prepared.market_cap_metadata is not None and not prepared.market_cap_metadata.empty:
        meta = prepared.market_cap_metadata.set_index("pair")
        if "market_cap_rank" in meta.columns:
            market_cap_rank = meta["market_cap_rank"]
        if "market_cap_name" in meta.columns:
            market_cap_name = meta["market_cap_name"]

    mini_order_series = (
        prepared.universe_table.set_index("pair").get("mini_order", pd.Series(dtype=float))
        if not prepared.universe_table.empty
        else pd.Series(dtype=float)
    )

    rows = []
    for pair in prepared.close.columns:
        rows.append(
            {
                "pair": pair,
                "aligned_hours": int(prepared.close[pair].notna().sum()),
                "aligned_days": float(prepared.close[pair].notna().sum() / 24.0),
                "start_timestamp": prepared.close.index.min(),
                "end_timestamp": prepared.close.index.max(),
                "mini_order": float(mini_order_series.get(pair, np.nan)),
                "latest_market_cap_usd": float(market_cap_snapshot.get(pair, np.nan)) if market_cap_snapshot is not None else np.nan,
                "market_cap_rank": float(market_cap_rank.get(pair, np.nan)) if market_cap_rank is not None else np.nan,
                "market_cap_name": market_cap_name.get(pair) if market_cap_name is not None else None,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("pair").reset_index(drop=True)
    return out


def build_indexed_prices(close: pd.DataFrame) -> pd.DataFrame:
    if close.empty:
        raise ValueError("close panel is empty.")
    first_row = close.iloc[0]
    if first_row.isna().any():
        raise ValueError("close panel must not contain NaNs in the first row after cleaning.")
    indexed = close.divide(first_row, axis="columns")
    return indexed.astype("float64")


def compute_log_returns(indexed_price: pd.DataFrame) -> pd.DataFrame:
    if indexed_price.empty:
        raise ValueError("indexed_price panel is empty.")
    log_price = np.log(indexed_price.astype("float64"))
    return log_price.diff()


__all__ = [
    "ROOSTOO_DEFAULT_BASE_URL",
    "Strategy2PreparedData",
    "build_indexed_prices",
    "build_panel_quality_table",
    "clean_and_align_panel",
    "compute_log_returns",
    "filter_prepared_data",
    "prepare_strategy2_environment",
    "split_prepared_data_insample_oos",
]
