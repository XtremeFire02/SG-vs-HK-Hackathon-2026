from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd

from .config import Strategy2Config, with_strategy2_updates
from .data import Strategy2PreparedData
from .metrics import build_backtest_metrics
from .portfolio import Strategy2TargetWeights, build_long_only_ts_target_weights
from .signal import Strategy2SignalResult, compute_hfm_signal


@dataclass
class Strategy2BacktestResult:
    config: Strategy2Config
    signal_result: Strategy2SignalResult
    target_weights: pd.DataFrame
    realized_weights: pd.DataFrame
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]
    universe_table: pd.DataFrame



def _apply_weight_buffer_and_min_order(
    desired_weights: np.ndarray,
    current_weights: np.ndarray,
    nav: float,
    mini_orders: np.ndarray,
    tradable_mask: np.ndarray,
    rebalance_buffer: float,
) -> np.ndarray:
    adjusted = desired_weights.copy()
    adjusted[~tradable_mask] = current_weights[~tradable_mask]

    if rebalance_buffer > 0.0:
        small_delta = np.abs(adjusted - current_weights) < rebalance_buffer
        adjusted[small_delta] = current_weights[small_delta]

    if nav > 0.0:
        trade_notional = np.abs(adjusted - current_weights) * nav
        adjusted[trade_notional < mini_orders] = current_weights[trade_notional < mini_orders]

    adjusted = np.clip(adjusted, 0.0, 1.0)
    total = float(np.nansum(adjusted))
    if total > 1.0:
        adjusted = adjusted / total
    return adjusted



def _apply_notional_buffer_and_min_order(
    desired_notional: np.ndarray,
    current_notional: np.ndarray,
    nav: float,
    mini_orders: np.ndarray,
    tradable_mask: np.ndarray,
    rebalance_buffer: float,
) -> np.ndarray:
    adjusted = desired_notional.copy()
    locked_mask = ~tradable_mask
    adjusted[locked_mask] = current_notional[locked_mask]

    if nav <= 0.0:
        return np.zeros_like(adjusted, dtype="float64")

    if rebalance_buffer > 0.0:
        small_delta = np.abs(adjusted - current_notional) < (rebalance_buffer * nav)
        adjusted[small_delta] = current_notional[small_delta]
        locked_mask = locked_mask | small_delta

    trade_notional = np.abs(adjusted - current_notional)
    below_min_order = trade_notional < mini_orders
    adjusted[below_min_order] = current_notional[below_min_order]
    locked_mask = locked_mask | below_min_order

    adjusted = np.clip(adjusted, 0.0, None)

    locked_total = float(np.nansum(adjusted[locked_mask]))
    free_mask = ~locked_mask
    free_total = float(np.nansum(adjusted[free_mask]))
    available_nav = max(nav - locked_total, 0.0)

    if free_total > available_nav and free_total > 0.0:
        adjusted[free_mask] = adjusted[free_mask] * (available_nav / free_total)

    return adjusted.astype("float64")



def _infer_trade_timestamp(index: pd.DatetimeIndex, signal_timestamp: pd.Timestamp) -> pd.Timestamp:
    signal_loc = index.get_loc(signal_timestamp)
    if isinstance(signal_loc, slice):
        signal_loc = signal_loc.start
    trade_loc = int(signal_loc) + 1
    if trade_loc >= len(index):
        raise RuntimeError("No next-bar execution timestamp exists after the first valid signal.")
    return index[trade_loc]



def _build_rebalance_targets(
    *,
    prev_ts: pd.Timestamp,
    nav_before: float,
    columns: Sequence[str],
    signal_result: Strategy2SignalResult,
    config: Strategy2Config,
    target_weights_result: Optional[Strategy2TargetWeights],
) -> tuple[np.ndarray, np.ndarray]:
    position_rule = config.portfolio.position_rule

    if position_rule == "fixed_usd_per_active_asset":
        signal_row = signal_result.signal.loc[prev_ts].reindex(columns).fillna(0.0)
        active_mask = signal_row.gt(config.portfolio.long_signal_threshold).to_numpy(dtype="float64")
        desired_notional = active_mask * float(config.portfolio.bet_size)
        desired_weights = np.zeros(len(columns), dtype="float64")
        if nav_before > 0.0:
            desired_weights = desired_notional / nav_before
        return desired_weights, desired_notional

    if target_weights_result is None:
        raise RuntimeError(
            "target_weights_result must be provided for weight-based portfolio sizing rules."
        )

    desired_weights = (
        target_weights_result.weights.loc[prev_ts]
        .reindex(columns)
        .fillna(0.0)
        .to_numpy(dtype="float64")
    )
    desired_notional = desired_weights * nav_before
    return desired_weights, desired_notional



def run_long_only_hfm_backtest(
    prepared_data: Strategy2PreparedData,
    config: Strategy2Config,
    *,
    signal_result: Optional[Strategy2SignalResult] = None,
    target_weights_result: Optional[Strategy2TargetWeights] = None,
) -> Strategy2BacktestResult:
    config.validate()

    if signal_result is None:
        signal_result = compute_hfm_signal(prepared_data.close, config.signal)
    if target_weights_result is None and config.portfolio.position_rule != "fixed_usd_per_active_asset":
        target_weights_result = build_long_only_ts_target_weights(
            signal_result.signal,
            long_signal_threshold=config.portfolio.long_signal_threshold,
            position_rule=config.portfolio.position_rule,
            bet_size=config.portfolio.bet_size,
        )

    if signal_result.first_valid_timestamp is None:
        raise RuntimeError("No valid signal timestamp was produced. Check the warm-up window and data.")

    trade_start_timestamp = _infer_trade_timestamp(prepared_data.close.index, signal_result.first_valid_timestamp)
    start_idx = int(prepared_data.close.index.get_loc(trade_start_timestamp))

    columns = list(prepared_data.close.columns)
    n_assets = len(columns)
    if n_assets == 0:
        raise RuntimeError("The prepared data contains no assets.")

    mini_order_series = (
        prepared_data.universe_table.set_index("pair").get("mini_order", pd.Series(dtype="float64"))
        if not prepared_data.universe_table.empty
        else pd.Series(dtype="float64")
    )
    mini_orders = np.array([float(mini_order_series.get(pair, 0.0) or 0.0) for pair in columns], dtype="float64")

    cash = float(config.execution.initial_cash)
    units = np.zeros(n_assets, dtype="float64")

    equity_records = []
    trade_records = []
    target_weight_records = []
    realized_weight_records = []
    total_turnover_notional = 0.0

    open_prices = prepared_data.open.astype("float64")
    close_prices = prepared_data.close.astype("float64")

    for idx in range(start_idx, len(prepared_data.close.index)):
        ts = prepared_data.close.index[idx]
        prev_ts = prepared_data.close.index[idx - 1]

        open_now = open_prices.iloc[idx].to_numpy(dtype="float64")
        close_now = close_prices.iloc[idx].to_numpy(dtype="float64")
        current_notional_at_open = units * open_now
        nav_before = float(cash + current_notional_at_open.sum())

        current_weights = np.zeros(n_assets, dtype="float64")
        if nav_before > 0.0:
            current_weights = current_notional_at_open / nav_before

        applied_target_weights = current_weights.copy()
        should_rebalance = bool(ts.hour % config.portfolio.rebalance_hours == 0)

        if should_rebalance:
            raw_target_weights, raw_target_notional = _build_rebalance_targets(
                prev_ts=prev_ts,
                nav_before=nav_before,
                columns=columns,
                signal_result=signal_result,
                config=config,
                target_weights_result=target_weights_result,
            )
            tradable_mask = np.isfinite(open_now) & (open_now > 0.0)

            if config.portfolio.position_rule == "fixed_usd_per_active_asset":
                desired_notional = _apply_notional_buffer_and_min_order(
                    desired_notional=raw_target_notional,
                    current_notional=current_notional_at_open,
                    nav=nav_before,
                    mini_orders=mini_orders,
                    tradable_mask=tradable_mask,
                    rebalance_buffer=config.execution.rebalance_buffer,
                )
                applied_target_weights = np.zeros(n_assets, dtype="float64")
                if nav_before > 0.0:
                    applied_target_weights = desired_notional / nav_before
            else:
                applied_target_weights = _apply_weight_buffer_and_min_order(
                    desired_weights=raw_target_weights,
                    current_weights=current_weights,
                    nav=nav_before,
                    mini_orders=mini_orders,
                    tradable_mask=tradable_mask,
                    rebalance_buffer=config.execution.rebalance_buffer,
                )
                desired_notional = applied_target_weights * nav_before

            sell_mask = desired_notional + 1e-12 < current_notional_at_open
            for asset_idx in np.where(sell_mask)[0]:
                mark_price = float(open_now[asset_idx])
                if not math.isfinite(mark_price) or mark_price <= 0.0:
                    continue
                marked_reduction = float(current_notional_at_open[asset_idx] - desired_notional[asset_idx])
                if marked_reduction <= 0.0:
                    continue

                quantity = min(float(units[asset_idx]), marked_reduction / mark_price)
                if quantity <= 0.0:
                    continue

                exec_price = mark_price * (1.0 - config.execution.slippage_bps / 10_000.0)
                gross_proceeds = quantity * exec_price
                fee = gross_proceeds * config.execution.fee_rate
                net_proceeds = gross_proceeds - fee

                units[asset_idx] -= quantity
                cash += net_proceeds
                total_turnover_notional += marked_reduction
                trade_records.append(
                    {
                        "timestamp": ts,
                        "signal_timestamp": prev_ts,
                        "pair": columns[asset_idx],
                        "side": "SELL",
                        "quantity": quantity,
                        "mark_price": mark_price,
                        "exec_price": exec_price,
                        "trade_notional_mark": marked_reduction,
                        "fee": fee,
                        "signal": float(signal_result.signal.loc[prev_ts, columns[asset_idx]]),
                    }
                )

            current_notional_after_sells = units * open_now
            buy_mask = desired_notional > current_notional_after_sells + 1e-12
            buy_order = np.argsort(-desired_notional)

            for asset_idx in buy_order:
                if not buy_mask[asset_idx]:
                    continue
                mark_price = float(open_now[asset_idx])
                if not math.isfinite(mark_price) or mark_price <= 0.0:
                    continue

                marked_increase = float(desired_notional[asset_idx] - current_notional_after_sells[asset_idx])
                if marked_increase <= 0.0 or cash <= 0.0:
                    continue

                exec_price = mark_price * (1.0 + config.execution.slippage_bps / 10_000.0)
                unit_cash_cost = exec_price * (1.0 + config.execution.fee_rate)
                if unit_cash_cost <= 0.0:
                    continue

                desired_quantity = marked_increase / mark_price
                max_affordable_quantity = cash / unit_cash_cost
                quantity = min(desired_quantity, max_affordable_quantity)
                if quantity <= 0.0:
                    continue

                gross_cost = quantity * exec_price
                fee = gross_cost * config.execution.fee_rate
                total_cash_used = gross_cost + fee
                if total_cash_used > cash + 1e-9:
                    quantity = cash / unit_cash_cost
                    gross_cost = quantity * exec_price
                    fee = gross_cost * config.execution.fee_rate
                    total_cash_used = gross_cost + fee
                if quantity <= 0.0:
                    continue

                units[asset_idx] += quantity
                cash -= total_cash_used
                total_turnover_notional += quantity * mark_price
                trade_records.append(
                    {
                        "timestamp": ts,
                        "signal_timestamp": prev_ts,
                        "pair": columns[asset_idx],
                        "side": "BUY",
                        "quantity": quantity,
                        "mark_price": mark_price,
                        "exec_price": exec_price,
                        "trade_notional_mark": quantity * mark_price,
                        "fee": fee,
                        "signal": float(signal_result.signal.loc[prev_ts, columns[asset_idx]]),
                    }
                )

        close_notional = units * close_now
        nav_close = float(cash + close_notional.sum())
        realized_weights = np.zeros(n_assets, dtype="float64")
        if nav_close > 0.0:
            realized_weights = close_notional / nav_close

        cash_weight = 0.0 if nav_close <= 0.0 else cash / nav_close
        gross_exposure = float(np.nansum(realized_weights))
        holding_count = int((realized_weights > 1e-12).sum())

        equity_records.append(
            {
                "timestamp": ts,
                "equity": nav_close,
                "cash": cash,
                "cash_weight": cash_weight,
                "gross_exposure": gross_exposure,
                "holding_count": holding_count,
                "rebalanced_from_prev_signal": should_rebalance,
            }
        )
        target_weight_records.append(pd.Series(applied_target_weights, index=columns, name=ts))
        realized_weight_records.append(pd.Series(realized_weights, index=columns, name=ts))

    equity_curve = pd.DataFrame(equity_records)
    if equity_curve.empty:
        raise RuntimeError("Backtest produced no equity records.")

    trades = pd.DataFrame(trade_records)
    target_weights_df = pd.DataFrame(target_weight_records)
    realized_weights_df = pd.DataFrame(realized_weight_records)
    metrics = build_backtest_metrics(
        equity_curve=equity_curve,
        trades=trades,
        turnover_notional=total_turnover_notional,
    )

    return Strategy2BacktestResult(
        config=config,
        signal_result=signal_result,
        target_weights=target_weights_df,
        realized_weights=realized_weights_df,
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        universe_table=prepared_data.universe_table.copy(),
    )



def run_normalization_sweep(
    prepared_data: Strategy2PreparedData,
    base_config: Strategy2Config,
    normalization_pairs: Sequence[Tuple[int, int]] = ((12, 168), (24, 168), (12, 720), (24, 720)),
) -> pd.DataFrame:
    rows = []
    for short_norm, long_norm in normalization_pairs:
        trial_config = with_strategy2_updates(
            base_config,
            signal={
                "short_norm_hours": short_norm,
                "long_norm_hours": long_norm,
            },
        )
        result = run_long_only_hfm_backtest(prepared_data, trial_config)
        rows.append(
            {
                "short_norm_hours": short_norm,
                "long_norm_hours": long_norm,
                **result.metrics,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["sharpe", "annualized_return"], ascending=[False, False]).reset_index(drop=True)
    return out


__all__ = [
    "Strategy2BacktestResult",
    "run_long_only_hfm_backtest",
    "run_normalization_sweep",
]
