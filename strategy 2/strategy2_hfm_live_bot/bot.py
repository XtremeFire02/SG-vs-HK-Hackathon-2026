from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

from .binance_client import BinanceSpotMarketDataClient, BinanceSymbolMapping
from .config import LiveBotSettings
from .roostoo_client import RoostooClient
from .signal_engine import LiveSignalResult, compute_live_signal


_CONSOLE_WIDTH = 100
_CONSOLE_SEPARATOR = "=" * _CONSOLE_WIDTH
_CONSOLE_SUB_SEPARATOR = "-" * _CONSOLE_WIDTH


def _is_scalar_like(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, pd.Timestamp, Real))


def _format_real(value: Real) -> str:
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"

    abs_number = abs(number)
    if abs_number == 0:
        return "0"
    if abs_number >= 1_000_000:
        rendered = f"{number:,.2f}"
    elif abs_number >= 1:
        rendered = f"{number:,.6f}"
    elif abs_number >= 1e-2:
        rendered = f"{number:.6f}"
    elif abs_number >= 1e-4:
        rendered = f"{number:.8f}"
    else:
        rendered = f"{number:.6g}"

    if "e" not in rendered.lower():
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _format_console_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Real):
        return _format_real(value)
    return str(value)


def _format_console_lines(value: Any, *, indent: int = 0) -> list[str]:
    prefix = " " * indent

    if isinstance(value, Mapping):
        if not value:
            return [f"{prefix}{{}}"]

        scalar_items = [(str(key), item) for key, item in value.items() if _is_scalar_like(item)]
        nested_items = [(str(key), item) for key, item in value.items() if not _is_scalar_like(item)]

        lines: list[str] = []
        if scalar_items:
            key_width = max(len(key) for key, _ in scalar_items)
            for key, item in scalar_items:
                lines.append(f"{prefix}{key:<{key_width}} : {_format_console_value(item)}")

        for idx, (key, item) in enumerate(nested_items):
            if lines:
                lines.append("")
            lines.append(f"{prefix}{key}:")
            child_lines = _format_console_lines(item, indent=indent + 4)
            lines.extend(child_lines)
            if idx < len(nested_items) - 1:
                lines.append("")

        return lines

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if not items:
            return [f"{prefix}[]"]

        if all(_is_scalar_like(item) for item in items):
            inline = ", ".join(_format_console_value(item) for item in items)
            if len(inline) <= max(30, _CONSOLE_WIDTH - indent - 2):
                return [f"{prefix}{inline}"]

        lines = []
        for item in items:
            if _is_scalar_like(item):
                lines.append(f"{prefix}- {_format_console_value(item)}")
            else:
                lines.append(f"{prefix}-")
                lines.extend(_format_console_lines(item, indent=indent + 4))
        return lines

    return [f"{prefix}{_format_console_value(value)}"]


@dataclass
class BalanceAmount:
    free: float
    locked: float

    @property
    def total(self) -> float:
        return float(self.free + self.locked)


@dataclass
class MarketTicker:
    max_bid: float
    min_ask: float
    last_price: float
    change_24h: float


@dataclass
class AccountSnapshot:
    balances: Dict[str, BalanceAmount]
    tickers: Dict[str, MarketTicker]
    usd_asset: str = "USD"

    @property
    def usd_free(self) -> float:
        return self.balances.get(self.usd_asset, BalanceAmount(0.0, 0.0)).free

    @property
    def usd_total(self) -> float:
        return self.balances.get(self.usd_asset, BalanceAmount(0.0, 0.0)).total

    def free_qty_for_pair(self, pair: str) -> float:
        asset = pair.split("/", 1)[0]
        return self.balances.get(asset, BalanceAmount(0.0, 0.0)).free

    def total_qty_for_pair(self, pair: str) -> float:
        asset = pair.split("/", 1)[0]
        return self.balances.get(asset, BalanceAmount(0.0, 0.0)).total


@dataclass(frozen=True)
class SessionState:
    started_at_utc: pd.Timestamp
    active_pairs: tuple[str, ...]
    pair_to_binance_symbol: Dict[str, BinanceSymbolMapping]
    warmup_hours_lost: int
    required_history_bars: int
    configured_cash_reserve_usd: float
    effective_cash_reserve_usd: float
    current_market_value_usd: float
    max_position_value_per_asset_usd: float


class HfmFixedUsdLiveBot:
    """Live spot bot for the long-only fixed-USD-per-active-asset adaptation."""

    def __init__(self, settings: LiveBotSettings) -> None:
        settings.validate()
        self.settings = settings

        self.roostoo = RoostooClient(
            api_key=settings.roostoo.resolved_api_key,
            secret_key=settings.roostoo.resolved_secret_key,
            base_url=settings.roostoo.resolved_base_url,
            timeout=settings.roostoo.timeout_seconds,
            max_retries=settings.roostoo.max_retries,
            retry_backoff_seconds=settings.roostoo.retry_backoff_seconds,
        )
        self.binance = BinanceSpotMarketDataClient(
            base_url=settings.binance.base_url,
            timeout=settings.binance.timeout_seconds,
            pause_seconds=settings.binance.pause_seconds,
        )

        self.exchange_info: dict[str, Any] = {}
        self.trade_pair_rules: dict[str, dict[str, Any]] = {}
        self.session_state: Optional[SessionState] = None
        self.session_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")

    # ------------------------------
    # Logging helpers
    # ------------------------------
    def _log(self, event: str, message: str, **details: Any) -> None:
        now = pd.Timestamp.now(tz="UTC")

        console_lines = [
            "",
            _CONSOLE_SEPARATOR,
            f"[{now.isoformat()}] {event.upper()}",
            message,
        ]
        if details:
            console_lines.append(_CONSOLE_SUB_SEPARATOR)
            console_lines.extend(_format_console_lines(details, indent=2))
        console_lines.append(_CONSOLE_SEPARATOR)
        print("\n".join(console_lines), flush=True)

        log_path = Path(self.settings.logging.jsonl_path)
        if log_path.parent != Path("."):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": now.isoformat(),
            "session_id": self.session_id,
            "event": event,
            "message": message,
            "details": details,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")

    # ------------------------------
    # Time helpers
    # ------------------------------
    def _wait_until_safe_to_use_latest_closed_bar(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        safe_time = now.floor("h") + pd.Timedelta(seconds=self.settings.trading.bar_close_delay_seconds)
        if now >= safe_time:
            return

        wait_seconds = max(0.0, (safe_time - now).total_seconds())
        self._log(
            "sleep",
            "Waiting for the latest 1h Binance bar to be safely closed.",
            wait_seconds=round(wait_seconds, 3),
            target_time_utc=safe_time.isoformat(),
        )
        while True:
            remaining = (safe_time - pd.Timestamp.now(tz="UTC")).total_seconds()
            if remaining <= 0:
                break
            time.sleep(min(self.settings.trading.loop_poll_seconds, max(0.25, remaining)))

    def _sleep_until_next_cycle(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        next_safe_time = now.floor("h") + pd.Timedelta(hours=1, seconds=self.settings.trading.bar_close_delay_seconds)
        self._log(
            "sleep",
            "Sleeping until the next hourly rebalance window.",
            next_cycle_utc=next_safe_time.isoformat(),
        )
        while True:
            remaining = (next_safe_time - pd.Timestamp.now(tz="UTC")).total_seconds()
            if remaining <= 0:
                break
            time.sleep(min(self.settings.trading.loop_poll_seconds, max(0.25, remaining)))

    # ------------------------------
    # Parsing / validation helpers
    # ------------------------------
    def _require_success(self, payload: Mapping[str, Any], *, context: str) -> None:
        success = payload.get("Success")
        if success is False:
            err = payload.get("ErrMsg", "")
            raise RuntimeError(f"{context} failed: {err}")

    def _load_exchange_info(self) -> None:
        self.exchange_info = self.roostoo.get_exchange_info()
        trade_pairs = self.exchange_info.get("TradePairs", {})
        if not isinstance(trade_pairs, dict):
            raise RuntimeError("Roostoo exchangeInfo did not contain a valid TradePairs mapping.")

        self.trade_pair_rules = {}
        for pair in self.settings.trading.active_pairs:
            rule = trade_pairs.get(pair)
            if rule is None:
                raise KeyError(f"Active pair {pair} was not found in Roostoo exchangeInfo.")
            if not bool(rule.get("CanTrade", False)):
                raise RuntimeError(f"Active pair {pair} exists but CanTrade=false on Roostoo.")
            self.trade_pair_rules[pair] = dict(rule)

    def _map_active_pairs_to_binance(self) -> dict[str, BinanceSymbolMapping]:
        mappings = self.binance.map_roostoo_pairs_to_spot_symbols(
            self.settings.trading.active_pairs,
            quote_priority=self.settings.binance.quote_priority,
            symbol_overrides=self.settings.binance.symbol_overrides,
        )
        for pair, mapping in mappings.items():
            self._log(
                "mapping",
                "Mapped Roostoo pair to Binance spot symbol.",
                pair=pair,
                binance_symbol=mapping.symbol,
                quote_asset=mapping.quote_asset,
                method=mapping.method,
            )
        return mappings

    def _fetch_balances(self) -> dict[str, BalanceAmount]:
        payload = self.roostoo.get_balance()
        self._require_success(payload, context="GET /v3/balance")

        # Roostoo test/live responses may expose balances under SpotWallet
        # rather than Wallet.
        wallet = payload.get("SpotWallet")
        if wallet is None:
            wallet = payload.get("Wallet", {})

        if not isinstance(wallet, dict):
            raise RuntimeError(
                "Roostoo balance payload did not include a valid SpotWallet/Wallet mapping."
            )

        balances: dict[str, BalanceAmount] = {}
        for asset, amounts in wallet.items():
            if not isinstance(amounts, dict):
                continue
            free = float(amounts.get("Free", 0.0) or 0.0)
            locked = float(amounts.get("Lock", 0.0) or 0.0)
            balances[str(asset)] = BalanceAmount(free=free, locked=locked)

        return balances

    def _fetch_tickers(self) -> dict[str, MarketTicker]:
        payload = self.roostoo.get_ticker()
        self._require_success(payload, context="GET /v3/ticker")
        data = payload.get("Data", {})
        if not isinstance(data, dict):
            raise RuntimeError("Roostoo ticker payload did not include a valid Data mapping.")

        tickers: dict[str, MarketTicker] = {}
        for pair, values in data.items():
            if not isinstance(values, dict):
                continue
            tickers[str(pair)] = MarketTicker(
                max_bid=float(values.get("MaxBid", 0.0) or 0.0),
                min_ask=float(values.get("MinAsk", 0.0) or 0.0),
                last_price=float(values.get("LastPrice", 0.0) or 0.0),
                change_24h=float(values.get("Change", 0.0) or 0.0),
            )
        return tickers

    def _fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances=self._fetch_balances(),
            tickers=self._fetch_tickers(),
            usd_asset=self.settings.trading.usd_asset,
        )

    def _safe_last_price(self, account: AccountSnapshot, pair: str) -> float:
        ticker = account.tickers.get(pair)
        if ticker is None or ticker.last_price <= 0:
            raise RuntimeError(f"Missing positive LastPrice ticker for {pair}.")
        return ticker.last_price

    def _buy_reference_price(self, account: AccountSnapshot, pair: str) -> float:
        ticker = account.tickers.get(pair)
        if ticker is None:
            raise RuntimeError(f"Missing ticker for {pair}.")
        reference = ticker.min_ask if ticker.min_ask > 0 else ticker.last_price
        if reference <= 0:
            raise RuntimeError(f"Could not derive a positive buy reference price for {pair}.")
        return reference

    def _sell_reference_price(self, account: AccountSnapshot, pair: str) -> float:
        ticker = account.tickers.get(pair)
        if ticker is None:
            raise RuntimeError(f"Missing ticker for {pair}.")
        reference = ticker.max_bid if ticker.max_bid > 0 else ticker.last_price
        if reference <= 0:
            raise RuntimeError(f"Could not derive a positive sell reference price for {pair}.")
        return reference

    def _pair_from_asset(self, asset: str) -> str:
        return f"{asset}/{self.settings.trading.usd_asset}"

    def _amount_precision(self, pair: str) -> int:
        if pair in self.trade_pair_rules:
            return int(self.trade_pair_rules[pair].get("AmountPrecision", 0) or 0)
        trade_pairs = self.exchange_info.get("TradePairs", {})
        return int((trade_pairs.get(pair) or {}).get("AmountPrecision", 0) or 0)

    def _mini_order_value(self, pair: str) -> float:
        if pair in self.trade_pair_rules:
            return float(self.trade_pair_rules[pair].get("MiniOrder", 0.0) or 0.0)
        trade_pairs = self.exchange_info.get("TradePairs", {})
        return float((trade_pairs.get(pair) or {}).get("MiniOrder", 0.0) or 0.0)

    def _floor_quantity(self, quantity: float, pair: str) -> float:
        precision = self._amount_precision(pair)
        quant = Decimal("1").scaleb(-precision)
        rounded = Decimal(str(quantity)).quantize(quant, rounding=ROUND_DOWN)
        return float(rounded)

    def _quantity_to_string(self, quantity: float, pair: str) -> str:
        precision = self._amount_precision(pair)
        floored = self._floor_quantity(quantity, pair)
        if floored <= 0:
            return "0"
        rendered = f"{floored:.{precision}f}" if precision > 0 else str(int(floored))
        rendered = rendered.rstrip("0").rstrip(".")
        return rendered or "0"

    def _position_market_value(self, account: AccountSnapshot, pair: str, *, use_total_qty: bool = True) -> float:
        quantity = account.total_qty_for_pair(pair) if use_total_qty else account.free_qty_for_pair(pair)
        return quantity * self._safe_last_price(account, pair)

    def _build_signal_result(self) -> LiveSignalResult:
        if self.session_state is None:
            raise RuntimeError("Session state has not been initialized.")
        close_panel = self.binance.fetch_recent_closed_panel(
            self.session_state.pair_to_binance_symbol,
            bars=self.session_state.required_history_bars,
        )
        return compute_live_signal(close_panel, self.settings.signal)

    def _summarize_portfolio_market_value(self, account: AccountSnapshot) -> float:
        total = account.usd_total
        for asset, bal in account.balances.items():
            if asset == self.settings.trading.usd_asset:
                continue
            if abs(bal.total) <= self.settings.trading.balance_epsilon:
                continue
            pair = self._pair_from_asset(asset)
            ticker = account.tickers.get(pair)
            if ticker is None or ticker.last_price <= 0:
                self._log(
                    "warning",
                    "Could not value a wallet asset because no valid Roostoo ticker was found.",
                    asset=asset,
                    pair=pair,
                    quantity_total=bal.total,
                )
                continue
            total += bal.total * ticker.last_price
        return float(total)

    # ------------------------------
    # Trade execution helpers
    # ------------------------------
    def _maybe_place_market_sell(self, pair: str, quantity: float, reason: str) -> bool:
        floored_quantity = self._floor_quantity(quantity, pair)
        if floored_quantity <= self.settings.trading.balance_epsilon:
            self._log("decision", "Skip SELL because floored quantity is zero.", pair=pair, quantity=quantity, reason=reason)
            return False

        quantity_str = self._quantity_to_string(floored_quantity, pair)
        if self.settings.trading.execute_trades:
            response = self.roostoo.place_order(pair=pair, side="SELL", quantity=quantity_str, order_type="MARKET")
            self._require_success(response, context=f"SELL {pair}")
            order_detail = response.get("OrderDetail", {})
            self._log(
                "order",
                "Executed market SELL.",
                pair=pair,
                quantity=quantity_str,
                reason=reason,
                status=order_detail.get("Status"),
                filled_average_price=order_detail.get("FilledAverPrice"),
                commission=order_detail.get("CommissionChargeValue"),
                order_id=order_detail.get("OrderID"),
            )
        else:
            self._log(
                "dry_run",
                "Would execute market SELL.",
                pair=pair,
                quantity=quantity_str,
                reason=reason,
            )
        return True

    def _maybe_place_market_buy(self, pair: str, notional_usd: float, account: AccountSnapshot, reason: str) -> bool:
        if notional_usd <= 0:
            self._log("decision", "Skip BUY because requested notional is not positive.", pair=pair, notional_usd=notional_usd, reason=reason)
            return False

        reference_price = self._buy_reference_price(account, pair)
        quantity = notional_usd / reference_price
        floored_quantity = self._floor_quantity(quantity, pair)
        if floored_quantity <= self.settings.trading.balance_epsilon:
            self._log(
                "decision",
                "Skip BUY because floored quantity is zero after AmountPrecision rounding.",
                pair=pair,
                requested_notional_usd=notional_usd,
                reference_price=reference_price,
                reason=reason,
            )
            return False

        estimated_order_value = floored_quantity * reference_price
        if estimated_order_value <= self._mini_order_value(pair):
            self._log(
                "decision",
                "Skip BUY because estimated order value does not clear MiniOrder.",
                pair=pair,
                estimated_order_value_usd=estimated_order_value,
                mini_order_usd=self._mini_order_value(pair),
                reason=reason,
            )
            return False

        quantity_str = self._quantity_to_string(floored_quantity, pair)
        if self.settings.trading.execute_trades:
            response = self.roostoo.place_order(pair=pair, side="BUY", quantity=quantity_str, order_type="MARKET")
            self._require_success(response, context=f"BUY {pair}")
            order_detail = response.get("OrderDetail", {})
            self._log(
                "order",
                "Executed market BUY.",
                pair=pair,
                quantity=quantity_str,
                requested_notional_usd=notional_usd,
                estimated_order_value_usd=estimated_order_value,
                reason=reason,
                status=order_detail.get("Status"),
                filled_average_price=order_detail.get("FilledAverPrice"),
                commission=order_detail.get("CommissionChargeValue"),
                order_id=order_detail.get("OrderID"),
            )
        else:
            self._log(
                "dry_run",
                "Would execute market BUY.",
                pair=pair,
                quantity=quantity_str,
                requested_notional_usd=notional_usd,
                estimated_order_value_usd=estimated_order_value,
                reason=reason,
            )
        return True

    # ------------------------------
    # Startup
    # ------------------------------
    def _connectivity_check(self) -> None:
        server_time = self.roostoo.get_server_time()
        self._log(
            "startup",
            "Connectivity check succeeded.",
            roostoo_mode=self.settings.roostoo.mode.upper(),
            roostoo_base_url=self.settings.roostoo.resolved_base_url,
            server_time=server_time,
        )

    def _close_non_active_positions(self) -> AccountSnapshot:
        account = self._fetch_account_snapshot()
        active_pairs_set = set(self.settings.trading.active_pairs)

        for asset, balance in sorted(account.balances.items()):
            if asset == self.settings.trading.usd_asset:
                continue
            if abs(balance.total) <= self.settings.trading.balance_epsilon:
                continue

            pair = self._pair_from_asset(asset)
            if pair in active_pairs_set:
                self._log(
                    "startup",
                    "Keeping startup position because the pair is active.",
                    pair=pair,
                    free_quantity=balance.free,
                    locked_quantity=balance.locked,
                )
                continue

            if pair not in self.exchange_info.get("TradePairs", {}):
                self._log(
                    "warning",
                    "Found a non-active asset but no matching Roostoo USD trade pair exists, so it cannot be auto-closed.",
                    asset=asset,
                    free_quantity=balance.free,
                    locked_quantity=balance.locked,
                )
                continue

            if balance.free <= self.settings.trading.balance_epsilon:
                self._log(
                    "warning",
                    "Found a non-active asset with no free quantity to sell. Locked quantity remains untouched.",
                    pair=pair,
                    free_quantity=balance.free,
                    locked_quantity=balance.locked,
                )
                continue

            estimated_order_value = balance.free * self._sell_reference_price(account, pair)
            if estimated_order_value <= self._mini_order_value(pair):
                self._log(
                    "warning",
                    "Non-active startup position is below MiniOrder, so it is being left as dust.",
                    pair=pair,
                    free_quantity=balance.free,
                    estimated_order_value_usd=estimated_order_value,
                    mini_order_usd=self._mini_order_value(pair),
                )
                continue

            self._log(
                "startup",
                "Closing non-active startup position.",
                pair=pair,
                free_quantity=balance.free,
                locked_quantity=balance.locked,
                estimated_order_value_usd=estimated_order_value,
            )
            traded = self._maybe_place_market_sell(pair, balance.free, reason="startup_non_active_position")
            if traded and self.settings.trading.execute_trades:
                account = AccountSnapshot(
                    balances=self._fetch_balances(),
                    tickers=account.tickers,
                    usd_asset=self.settings.trading.usd_asset,
                )

        if self.settings.trading.execute_trades:
            account = self._fetch_account_snapshot()
        return account

    def _initialize_session_state(self, account: AccountSnapshot, pair_to_symbol: dict[str, BinanceSymbolMapping]) -> None:
        current_market_value_usd = self._summarize_portfolio_market_value(account)
        effective_cash_reserve = min(self.settings.trading.cash_reserve_usd, max(account.usd_free, 0.0))
        max_position_value_per_asset = max(
            0.0,
            (current_market_value_usd - effective_cash_reserve) / float(len(self.settings.trading.active_pairs)),
        )

        self.session_state = SessionState(
            started_at_utc=pd.Timestamp.now(tz="UTC"),
            active_pairs=tuple(self.settings.trading.active_pairs),
            pair_to_binance_symbol=pair_to_symbol,
            warmup_hours_lost=self.settings.signal.warmup_hours_lost,
            required_history_bars=self.settings.signal.required_history_bars,
            configured_cash_reserve_usd=self.settings.trading.cash_reserve_usd,
            effective_cash_reserve_usd=effective_cash_reserve,
            current_market_value_usd=current_market_value_usd,
            max_position_value_per_asset_usd=max_position_value_per_asset,
        )

        active_holdings = {}
        for pair in self.settings.trading.active_pairs:
            active_holdings[pair] = {
                "quantity_free": account.free_qty_for_pair(pair),
                "quantity_total": account.total_qty_for_pair(pair),
                "market_value_usd": self._position_market_value(account, pair),
                "last_price": self._safe_last_price(account, pair),
            }

        self._log(
            "startup",
            "Initialized session state and fixed per-asset market-value cap.",
            started_at_utc=self.session_state.started_at_utc.isoformat(),
            active_pairs=list(self.session_state.active_pairs),
            warmup_hours_lost=self.session_state.warmup_hours_lost,
            required_history_bars=self.session_state.required_history_bars,
            configured_cash_reserve_usd=self.session_state.configured_cash_reserve_usd,
            effective_cash_reserve_usd=self.session_state.effective_cash_reserve_usd,
            current_market_value_usd=self.session_state.current_market_value_usd,
            max_position_value_per_asset_usd=self.session_state.max_position_value_per_asset_usd,
            usd_free=account.usd_free,
            usd_total=account.usd_total,
            active_holdings=active_holdings,
        )

    def startup(self) -> None:
        self._connectivity_check()
        self._load_exchange_info()
        pair_to_symbol = self._map_active_pairs_to_binance()
        post_reconciliation_account = self._close_non_active_positions()
        self._initialize_session_state(post_reconciliation_account, pair_to_symbol)

    # ------------------------------
    # Hourly cycle
    # ------------------------------
    def _build_working_account_copy(self, account: AccountSnapshot) -> AccountSnapshot:
        copied_balances = {
            asset: BalanceAmount(free=bal.free, locked=bal.locked)
            for asset, bal in account.balances.items()
        }
        copied_tickers = {
            pair: MarketTicker(
                max_bid=t.max_bid,
                min_ask=t.min_ask,
                last_price=t.last_price,
                change_24h=t.change_24h,
            )
            for pair, t in account.tickers.items()
        }
        return AccountSnapshot(balances=copied_balances, tickers=copied_tickers, usd_asset=account.usd_asset)

    def _update_working_account_after_sell(self, account: AccountSnapshot, pair: str, quantity: float) -> None:
        asset = pair.split("/", 1)[0]
        price = self._sell_reference_price(account, pair)
        proceeds = quantity * price * (1.0 - self.settings.trading.estimated_fee_rate)

        base_balance = account.balances.get(asset, BalanceAmount(0.0, 0.0))
        usd_balance = account.balances.get(account.usd_asset, BalanceAmount(0.0, 0.0))

        new_free_asset = max(0.0, base_balance.free - quantity)
        account.balances[asset] = BalanceAmount(free=new_free_asset, locked=base_balance.locked)
        account.balances[account.usd_asset] = BalanceAmount(
            free=usd_balance.free + proceeds,
            locked=usd_balance.locked,
        )

    def _update_working_account_after_buy(self, account: AccountSnapshot, pair: str, notional_usd: float) -> None:
        asset = pair.split("/", 1)[0]
        price = self._buy_reference_price(account, pair)
        quantity = self._floor_quantity(notional_usd / price, pair)
        if quantity <= 0:
            return

        spend = quantity * price * (1.0 + self.settings.trading.estimated_fee_rate)
        base_balance = account.balances.get(asset, BalanceAmount(0.0, 0.0))
        usd_balance = account.balances.get(account.usd_asset, BalanceAmount(0.0, 0.0))

        account.balances[asset] = BalanceAmount(
            free=base_balance.free + quantity,
            locked=base_balance.locked,
        )
        account.balances[account.usd_asset] = BalanceAmount(
            free=max(0.0, usd_balance.free - spend),
            locked=usd_balance.locked,
        )

    def run_cycle(self, *, label: str) -> None:
        if self.session_state is None:
            raise RuntimeError("Session state is not initialized. Call startup() first.")

        signal_result = self._build_signal_result()
        latest_ts = signal_result.latest_signal_timestamp
        latest_snapshot = signal_result.latest_signal_snapshot()

        actual_account = self._fetch_account_snapshot()
        working_account = self._build_working_account_copy(actual_account)

        per_pair_summary = {}
        for pair in self.settings.trading.active_pairs:
            per_pair_summary[pair] = {
                "signal": float(latest_snapshot.loc[pair, "signal"]),
                "free_qty": working_account.free_qty_for_pair(pair),
                "total_qty": working_account.total_qty_for_pair(pair),
                "position_value_usd": self._position_market_value(working_account, pair),
                "last_price": self._safe_last_price(working_account, pair),
                "min_ask": working_account.tickers[pair].min_ask,
                "max_bid": working_account.tickers[pair].max_bid,
            }

        self._log(
            "cycle",
            "Starting hourly signal evaluation.",
            label=label,
            signal_timestamp_utc=latest_ts.isoformat(),
            usd_free=working_account.usd_free,
            usd_total=working_account.usd_total,
            effective_cash_reserve_usd=self.session_state.effective_cash_reserve_usd,
            max_position_value_per_asset_usd=self.session_state.max_position_value_per_asset_usd,
            pairs=per_pair_summary,
        )

        for pair in self.settings.trading.active_pairs:
            signal_value = float(latest_snapshot.loc[pair, "signal"])
            free_qty = working_account.free_qty_for_pair(pair)
            total_qty = working_account.total_qty_for_pair(pair)
            position_value = self._position_market_value(working_account, pair)
            free_usd = working_account.usd_free
            threshold = self.settings.signal.signal_threshold

            if signal_value <= threshold:
                if free_qty <= self.settings.trading.balance_epsilon:
                    self._log(
                        "decision",
                        "Hold cash / no SELL needed because free quantity is zero or dust.",
                        pair=pair,
                        signal=signal_value,
                        threshold=threshold,
                        free_quantity=free_qty,
                        total_quantity=total_qty,
                    )
                    continue

                estimated_order_value = free_qty * self._sell_reference_price(working_account, pair)
                if estimated_order_value <= self._mini_order_value(pair):
                    self._log(
                        "decision",
                        "Signal says SELL but free position is below MiniOrder, so it is being left as dust.",
                        pair=pair,
                        signal=signal_value,
                        threshold=threshold,
                        free_quantity=free_qty,
                        estimated_order_value_usd=estimated_order_value,
                        mini_order_usd=self._mini_order_value(pair),
                    )
                    continue

                traded = self._maybe_place_market_sell(
                    pair=pair,
                    quantity=free_qty,
                    reason="signal_below_or_equal_threshold",
                )
                if traded:
                    if self.settings.trading.execute_trades:
                        refreshed_balances = self._fetch_balances()
                        working_account = AccountSnapshot(
                            balances=refreshed_balances,
                            tickers=working_account.tickers,
                            usd_asset=working_account.usd_asset,
                        )
                    else:
                        self._update_working_account_after_sell(working_account, pair, free_qty)
                continue

            proposed_position_value = position_value + self.settings.trading.bet_size_usd
            if proposed_position_value > self.session_state.max_position_value_per_asset_usd + 1e-12:
                self._log(
                    "decision",
                    "Skip BUY because the fixed per-asset market-value cap would be breached.",
                    pair=pair,
                    signal=signal_value,
                    threshold=threshold,
                    current_position_value_usd=position_value,
                    proposed_position_value_usd=proposed_position_value,
                    max_position_value_per_asset_usd=self.session_state.max_position_value_per_asset_usd,
                )
                continue

            buy_price = self._buy_reference_price(working_account, pair)
            estimated_cash_needed = self.settings.trading.bet_size_usd * (1.0 + self.settings.trading.estimated_fee_rate)
            spendable_usd = max(0.0, free_usd - self.session_state.effective_cash_reserve_usd)
            if spendable_usd + 1e-12 < estimated_cash_needed:
                self._log(
                    "decision",
                    "Skip BUY because it would violate the USD cash reserve.",
                    pair=pair,
                    signal=signal_value,
                    threshold=threshold,
                    free_usd=free_usd,
                    spendable_usd=spendable_usd,
                    estimated_cash_needed=estimated_cash_needed,
                )
                continue

            estimated_order_value = self.settings.trading.bet_size_usd
            if estimated_order_value <= self._mini_order_value(pair):
                self._log(
                    "decision",
                    "Skip BUY because bet_size_usd is below the pair MiniOrder.",
                    pair=pair,
                    signal=signal_value,
                    threshold=threshold,
                    bet_size_usd=self.settings.trading.bet_size_usd,
                    mini_order_usd=self._mini_order_value(pair),
                )
                continue

            traded = self._maybe_place_market_buy(
                pair=pair,
                notional_usd=self.settings.trading.bet_size_usd,
                account=working_account,
                reason="signal_above_threshold_and_under_cap",
            )
            if traded:
                if self.settings.trading.execute_trades:
                    refreshed_balances = self._fetch_balances()
                    working_account = AccountSnapshot(
                        balances=refreshed_balances,
                        tickers=working_account.tickers,
                        usd_asset=working_account.usd_asset,
                    )
                else:
                    self._update_working_account_after_buy(working_account, pair, self.settings.trading.bet_size_usd)

        final_summary = {}
        for pair in self.settings.trading.active_pairs:
            final_summary[pair] = {
                "signal": float(latest_snapshot.loc[pair, "signal"]),
                "free_qty": working_account.free_qty_for_pair(pair),
                "total_qty": working_account.total_qty_for_pair(pair),
                "position_value_usd": self._position_market_value(working_account, pair),
            }

        self._log(
            "cycle",
            "Completed hourly signal evaluation.",
            label=label,
            signal_timestamp_utc=latest_ts.isoformat(),
            usd_free=working_account.usd_free,
            usd_total=working_account.usd_total,
            final_positions=final_summary,
        )

    # ------------------------------
    # Public runner
    # ------------------------------
    def run(self) -> None:
        self.startup()

        # Immediate first cycle using the latest fully closed bar.
        self._wait_until_safe_to_use_latest_closed_bar()
        self.run_cycle(label="startup")

        if not self.settings.trading.run_forever:
            self._log("shutdown", "RUN_FOREVER is false, so the bot is exiting after the startup cycle.")
            return

        while True:
            self._sleep_until_next_cycle()
            self._wait_until_safe_to_use_latest_closed_bar()
            self.run_cycle(label="hourly")
