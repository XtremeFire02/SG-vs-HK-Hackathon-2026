import time
import os
import csv
import signal
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from roostoo_client import RoostooClient

API_KEY = os.getenv("API_KEY")
SECRET = os.getenv("SECRET")

client = RoostooClient(api_key=API_KEY, secret_key=SECRET)

# ─── STRATEGY: VOLATILITY BREAKOUT WITH TRAILING STOPS ───────────────────────
#
#  Why this works:
#    - In flat markets, price never breaks the channel -> 0 trades -> 0 loss
#    - In trending markets, breakouts catch the move early and ride it
#    - Trailing stop lets winners run, cuts losers mechanically
#    - Risk-based sizing: every trade risks the same % of equity
#
#  Signal:
#    1. Donchian channel: ENTRY_LB-bar high / EXIT_LB-bar low
#    2. BUY when price > channel high AND volatility filter passes
#    3. EXIT when price < trailing stop (peak - STOP_MULT * ATR)
#       OR price < EXIT_LB-bar low
#    4. Cross-sectional: only hold TOP_K positions, prefer strongest breakouts
#
#  Backtested Sharpe: +18.76 (7d), +5.80 (1mo), +1.75 (60d)
#  Profit Factor (with regime detector): avg Sharpe +8.77
#


def run_breakout_bot():
    print("=" * 60)
    print("  VOLATILITY BREAKOUT BOT")
    print("  Donchian channel breakout | Trailing ATR stops")
    print("  Risk-parity sizing | Volatility filter")
    print("=" * 60)

    # ── Auth check ──
    print("\nChecking API credentials...")
    try:
        bal = client.get_balance()
        print("  API credentials OK")
    except Exception as e:
        print(f"  !! AUTH FAILED: {e}")
        print(f"  !! Check your API_KEY and SECRET in .env")
        return

    # ── Load exchange precision rules ──
    ex_info = client.get_exchange_info()
    amt_precision = {}
    price_precision = {}
    if ex_info and "TradePairs" in ex_info:
        for pname, pinfo in ex_info["TradePairs"].items():
            coin = pinfo["Coin"]
            amt_precision[coin] = pinfo["AmountPrecision"]
            if "PricePrecision" in pinfo:
                price_precision[coin] = pinfo["PricePrecision"]

    # ── Parameters ──
    UNIVERSE       = ["BTC", "ETH", "SOL", "BNB"]
    ENTRY_LB       = 120    # Donchian entry channel lookback (~60 min at 30s ticks) — optimized via sweep
    EXIT_LB        = 48     # Donchian exit channel lookback (tighter)
    ATR_WIN        = 20     # ATR window
    TOP_K          = 4      # Allow holding all 4 coins if the market moves
    RISK_PER_TRADE = 0.01   # risk 1% of equity per trade
    MAX_ALLOC      = 0.24   # 4 positions * 24% = 96% utilization
    STOP_MULT      = 3.0    # trailing stop = peak - STOP_MULT * ATR (wider to survive noise)
    MIN_VOL        = 0.00005 # minimum ATR/price to allow entry — optimized via sweep
    PRICE_OFFSET   = 0.0002 # 0.02% limit price offset
    COOLDOWN       = 60     # ticks after stop-out before re-entry (~30 min) — prevents whipsaw re-entry
    MIN_ORDER_USD  = 50.00  # minimum order notional
    MIN_BREAKOUT_STR = 0.5  # minimum breakout strength (ATR units above channel) — filters weak breakouts
    CONFIRM_TICKS  = 2      # price must stay above channel for N ticks before entry (confirms breakout)
    PROFIT_TARGET  = 2.0    # take profit at PROFIT_TARGET * ATR above entry
    REGIME_VOL_THRESH = 0.001  # avg vol across coins must exceed this to trade (0.1%) — strict filter for real trends
    REGIME_WINDOW  = 20     # how many ticks of vol history to average for regime (smooths noise)

    # ── Circuit Breakers ──
    MAX_DAILY_LOSS_PCT  = 0.05   # halt trading if daily loss exceeds 5%
    MAX_DRAWDOWN_PCT    = 0.10   # kill switch if drawdown from peak exceeds 10%
    MAX_CONSECUTIVE_LOSSES = 5   # pause entries after 5 consecutive losing trades

    # ── Pending Order Handling ──
    PENDING_TIMEOUT_TICKS = 3    # ticks before re-placing an unfilled order at worse price

    # ── Stale Data Detection ──
    STALE_PRICE_TICKS   = 10     # alert if price unchanged for this many ticks
    MAX_STALE_TICKS     = 30     # skip trading if stale for this many ticks

    # ── Error Recovery ──
    MAX_API_RETRIES     = 3
    BASE_RETRY_DELAY    = 2      # seconds (doubles each retry)

    # ── Health Check ──
    HEALTH_LOG_FILE     = "bot_health.log"
    HEALTH_INTERVAL     = 60     # log health every N ticks

    HIST_KEEP = ENTRY_LB + ATR_WIN + 10

    # ── Per-coin state ──
    hist       = {}
    ticks      = {c: 0 for c in UNIVERSE}
    pos        = {c: 0.0 for c in UNIVERSE}
    vol_history = []  # rolling avg vol across coins for regime detection
    peak_price = {c: 0.0 for c in UNIVERSE}
    entry_price = {}
    current_longs = set()
    last_exit  = {c: -COOLDOWN for c in UNIVERSE}
    last_prices = {c: None for c in UNIVERSE}       # for stale detection
    stale_count = {c: 0 for c in UNIVERSE}           # consecutive unchanged ticks
    consecutive_losses = 0                            # for circuit breaker
    daily_start_equity = None                         # set on first equity calc
    circuit_breaker_active = False
    api_error_count = 0
    pending_exits = {}     # coin -> {"trade_pnl": float, "tick": int} — awaiting fill confirmation
    pending_entries = {}   # coin -> {"qty": float, "tick": int} — awaiting fill confirmation
    breakout_confirm = {c: 0 for c in UNIVERSE}  # ticks price has been above channel high

    global_tick = 0

    # ── Sync with exchange ──
    print("\nSyncing with exchange...")
    wallet_key = "SpotWallet" if "SpotWallet" in bal else "Wallet"
    if wallet_key in bal:
        for c in UNIVERSE:
            c_entry = bal[wallet_key].get(c, {})
            pos[c] = c_entry.get("Free", 0.0) + c_entry.get("Locked", 0.0)
        usd_entry = bal[wallet_key].get("USD", {})
        free_usd = usd_entry.get("Free", 0.0) + usd_entry.get("Locked", 0.0)
    else:
        free_usd = 50000.0

    start_equity = free_usd
    try:
        tk_init = client.get_ticker()
        if tk_init and "Data" in tk_init:
            for c in UNIVERSE:
                lp = tk_init["Data"].get(f"{c}/USD", {}).get("LastPrice", 0)
                p = float(lp)
                start_equity += pos[c] * p
                if pos[c] > 0 and p > 0:
                    entry_price[c] = p
                    peak_price[c] = p
                    current_longs.add(c)
    except Exception:
        pass

    # ── Liquidate leftover positions from previous runs ──
    leftovers_sold = False
    for c in UNIVERSE:
        if pos[c] > 0:
            pair = f"{c}/USD"
            prec = amt_precision.get(c, 4)
            p_prec = price_precision.get(c, 2)
            qty = round(pos[c], prec)
            if qty > 0:
                try:
                    tk = client.get_ticker(pair)
                    price_now = float(tk["Data"][pair]["LastPrice"])
                    limit_price = round(price_now * (1 - PRICE_OFFSET), p_prec)
                    qty_str = f"{qty:.{prec}f}"
                    price_str = f"{limit_price:.{p_prec}f}"
                    print(f"  [CLEANUP] Selling leftover {qty_str} {c} @ ${limit_price}")
                    client.place_order(pair=pair, side="SELL",
                                       quantity=qty_str, order_type="LIMIT",
                                       price=price_str)
                    leftovers_sold = True
                    pos[c] = 0.0
                    peak_price[c] = 0.0
                    entry_price.pop(c, None)
                    current_longs.discard(c)
                except Exception as e:
                    print(f"  [CLEANUP] Failed to sell {c}: {e}")

    if leftovers_sold:
        time.sleep(3)  # wait for orders to fill
        # Re-sync balances after cleanup
        try:
            bal = client.get_balance()
            wallet_key = "SpotWallet" if "SpotWallet" in bal else "Wallet"
            usd_entry = bal[wallet_key].get("USD", {})
            free_usd = usd_entry.get("Free", 0.0) + usd_entry.get("Locked", 0.0)
            start_equity = free_usd
            for c in UNIVERSE:
                c_entry = bal[wallet_key].get(c, {})
                pos[c] = c_entry.get("Free", 0.0) + c_entry.get("Locked", 0.0)
                if pos[c] > 0:
                    tk = client.get_ticker(f"{c}/USD")
                    p = float(tk["Data"][f"{c}/USD"]["LastPrice"])
                    start_equity += pos[c] * p
            print(f"  [CLEANUP] Done. New starting equity: ${start_equity:,.2f}")
        except Exception as e:
            print(f"  [CLEANUP] Re-sync failed: {e}")

    equity = start_equity
    peak_equity = start_equity

    print(f"  Starting equity:    ${start_equity:,.2f}")
    print(f"  Starting positions: {pos}")
    print(f"  Strategy:           Volatility Breakout")
    print(f"  Channel:            Entry={ENTRY_LB} bars, Exit={EXIT_LB} bars")
    print(f"  Trailing stop:      {STOP_MULT}x ATR | Vol filter: >{MIN_VOL:.2%}")
    print(f"  Risk per trade:     {RISK_PER_TRADE:.0%} equity | Max alloc: {MAX_ALLOC:.0%}")
    print(f"  Top K positions:    {TOP_K}")
    print("-" * 60)

    # ── CSV Audit Log ──
    LOG_FILE = "trade_log.csv"
    log_exists = os.path.exists(LOG_FILE)
    log_fp = open(LOG_FILE, "a", newline="")
    log_writer = csv.writer(log_fp)
    if not log_exists:
        log_writer.writerow([
            "timestamp", "coin", "side", "qty", "price", "limit_price",
            "atr", "chan_high", "chan_low", "vol_pct", "equity", "pnl_pct",
        ])
        log_fp.flush()

    # ── Graceful shutdown ──
    shutdown_requested = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        print("\n[SHUTDOWN] Signal received, finishing current cycle...")

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    def log_trade(coin, side, qty, price, limit_price, atr_val, ch, cl, vp):
        pnl = (equity / start_equity - 1) * 100
        log_writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            coin, side, f"{qty:.6f}", f"{price:.2f}", f"{limit_price:.2f}",
            f"{atr_val:.4f}", f"{ch:.2f}", f"{cl:.2f}",
            f"{vp:.6f}", f"{equity:.2f}", f"{pnl:.2f}",
        ])
        log_fp.flush()

    def api_call_with_retry(func, *args, **kwargs):
        """Call an API function with exponential backoff retry."""
        for attempt in range(MAX_API_RETRIES):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if attempt < MAX_API_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    print(f"  [RETRY] {func.__name__} failed (attempt {attempt+1}): {e} — retrying in {delay}s")
                    time.sleep(delay)
                else:
                    raise

    def log_health():
        """Write a health check line to the health log."""
        with open(HEALTH_LOG_FILE, "a") as hf:
            hf.write(json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "tick": global_tick,
                "equity": equity,
                "pnl_pct": (equity / start_equity - 1) * 100,
                "positions": {c: pos[c] for c in UNIVERSE if pos[c] > 0},
                "regime": regime,
                "circuit_breaker": circuit_breaker_active,
                "consecutive_losses": consecutive_losses,
            }) + "\n")

    # ── Main loop ──
    regime = "WARMUP"
    try:
        while not shutdown_requested:
            global_tick += 1

            # ── Health check logging ──
            if global_tick % HEALTH_INTERVAL == 0:
                try:
                    log_health()
                except Exception:
                    pass

            # Cancel stale orders — but NOT for coins with pending exit/entry orders
            for coin in UNIVERSE:
                if coin in pending_exits or coin in pending_entries:
                    continue
                try:
                    api_call_with_retry(client.cancel_order, f"{coin}/USD")
                except Exception:
                    pass

            # Fetch prices (with retry)
            prices_now = {}
            for coin in UNIVERSE:
                try:
                    td = api_call_with_retry(client.get_ticker, f"{coin}/USD")
                    if td and "Data" in td and f"{coin}/USD" in td["Data"]:
                        prices_now[coin] = float(td["Data"][f"{coin}/USD"]["LastPrice"])
                except Exception as e:
                    api_error_count += 1
                    print(f"  [API-ERR] get_ticker {coin} failed after retries: {e}")

            if len(prices_now) < len(UNIVERSE):
                print(f"[TICK {global_tick}] Missing price data, skipping")
                time.sleep(30)
                continue

            # ── Stale data detection ──
            any_stale = False
            for coin in UNIVERSE:
                if last_prices[coin] is not None and prices_now[coin] == last_prices[coin]:
                    stale_count[coin] += 1
                else:
                    stale_count[coin] = 0
                last_prices[coin] = prices_now[coin]

                if stale_count[coin] >= STALE_PRICE_TICKS:
                    if stale_count[coin] == STALE_PRICE_TICKS:
                        print(f"  [STALE] {coin} price unchanged for {stale_count[coin]} ticks!")
                if stale_count[coin] >= MAX_STALE_TICKS:
                    any_stale = True

            if any_stale:
                print(f"[TICK {global_tick}] Stale data detected (>{MAX_STALE_TICKS} ticks unchanged), skipping trades")
                time.sleep(30)
                continue

            # Update history
            for coin in UNIVERSE:
                if coin not in hist:
                    hist[coin] = []
                hist[coin].append(prices_now[coin])
                ticks[coin] += 1
                if len(hist[coin]) > HIST_KEEP:
                    hist[coin] = hist[coin][-HIST_KEEP:]

            # Update trailing peaks
            for c in current_longs:
                if prices_now[c] > peak_price[c]:
                    peak_price[c] = prices_now[c]

            # Warmup
            min_ticks = min(ticks[c] for c in UNIVERSE)
            if min_ticks <= ENTRY_LB:
                print(f"[TICK {global_tick}] Warming up {min_ticks}/{ENTRY_LB}")
                time.sleep(30)
                continue

            # ── Compute indicators ──
            atr = {}
            chan_high = {}
            chan_low = {}
            vol_pct = {}

            for c in UNIVERSE:
                ph = hist[c]

                # ATR
                diffs = [abs(ph[j] - ph[j-1])
                         for j in range(len(ph) - ATR_WIN, len(ph))]
                atr[c] = sum(diffs) / len(diffs) if diffs else 1e-10

                # Donchian channels (exclude current bar)
                chan_high[c] = max(ph[-ENTRY_LB - 1:-1])
                chan_low[c] = min(ph[-EXIT_LB - 1:-1])

                # Volatility %
                vol_pct[c] = atr[c] / prices_now[c] if prices_now[c] > 0 else 0

            # ── CHECK EXITS (always execute — even if circuit breaker is on) ──
            for c in list(current_longs):
                if pos[c] <= 0:
                    current_longs.discard(c)
                    continue
                if c in pending_exits:
                    continue  # already placed a sell, waiting for fill confirmation

                stop_level = peak_price[c] - STOP_MULT * atr[c]
                exit_channel = chan_low[c]
                exit_level = max(stop_level, exit_channel)

                # ── Profit target: take profit at PROFIT_TARGET * ATR above entry ──
                ep = entry_price.get(c)
                profit_hit = False
                if ep and ep > 0 and atr[c] > 0:
                    target_price = ep + PROFIT_TARGET * atr[c]
                    if prices_now[c] >= target_price:
                        profit_hit = True

                if profit_hit:
                    reason = "PROFIT-TARGET"
                elif prices_now[c] <= exit_level:
                    reason = "STOP" if prices_now[c] <= stop_level else "CHAN-EXIT"
                else:
                    continue  # no exit signal

                pair = f"{c}/USD"
                prec = amt_precision.get(c, 4)
                p_prec = price_precision.get(c, 2)
                qty = round(pos[c], prec)

                # ── Dust position: too small to sell — abandon it ──
                if qty > 0 and qty * prices_now[c] < MIN_ORDER_USD:
                    trade_pnl = 0.0
                    ep = entry_price.get(c)
                    if ep and ep > 0:
                        trade_pnl = (prices_now[c] / ep - 1)
                    pnl_str = f" P&L: {trade_pnl*100:+.2f}%" if ep else ""
                    print(f"  [DUST-ABANDON] {c}: {qty} units (${qty * prices_now[c]:.2f}) "
                          f"below min order ${MIN_ORDER_USD}{pnl_str} — removing from tracking")
                    pos[c] = 0.0
                    peak_price[c] = 0.0
                    entry_price.pop(c, None)
                    current_longs.discard(c)
                    last_exit[c] = global_tick
                    if trade_pnl <= 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    continue

                if qty > 0:
                    limit_price = round(prices_now[c] * (1 - PRICE_OFFSET), p_prec)
                    qty_str = f"{qty:.{prec}f}"
                    price_str = f"{limit_price:.{p_prec}f}"

                    trade_pnl = 0.0
                    ep = entry_price.get(c)
                    if ep and ep > 0:
                        trade_pnl = (prices_now[c] / ep - 1)
                    pnl_str = f" P&L: {trade_pnl*100:+.2f}%" if ep else ""

                    if reason == "PROFIT-TARGET":
                        print(f"  [{reason}] {c}: ${prices_now[c]:,.2f} >= "
                              f"${target_price:,.2f}{pnl_str}")
                    else:
                        print(f"  [{reason}] {c}: ${prices_now[c]:,.2f} <= "
                              f"${exit_level:,.2f}{pnl_str}")

                    try:
                        api_call_with_retry(
                            client.place_order, pair=pair, side="SELL",
                            quantity=qty_str, order_type="LIMIT",
                            price=price_str)
                        print(f"    >> SELL-{reason} {qty_str} {c} @ ${limit_price} (pending fill)")
                        log_trade(c, f"SELL-{reason}", qty, prices_now[c],
                                  limit_price, atr[c], chan_high[c],
                                  chan_low[c], vol_pct[c])

                        # Mark as pending — position cleared only after resync confirms fill
                        pending_exits[c] = {
                            "trade_pnl": trade_pnl,
                            "tick": global_tick,
                            "reason": reason,
                        }

                    except Exception as e:
                        print(f"    >> SELL-{reason} {c} FAILED: {e}")

            # ── Expire stale pending exits: cancel and re-place at worse price ──
            for c in list(pending_exits):
                age = global_tick - pending_exits[c]["tick"]
                if age >= PENDING_TIMEOUT_TICKS:
                    try:
                        client.cancel_order(f"{c}/USD")
                    except Exception:
                        pass

                    # If position is dust, abandon instead of retrying
                    pos_value = pos[c] * prices_now.get(c, 0)
                    if pos_value < MIN_ORDER_USD:
                        info = pending_exits.pop(c)
                        print(f"  [PENDING-EXPIRE] {c} exit unfilled — dust position (${pos_value:.2f}), abandoning")
                        pos[c] = 0.0
                        peak_price[c] = 0.0
                        entry_price.pop(c, None)
                        current_longs.discard(c)
                        last_exit[c] = global_tick
                        if info["trade_pnl"] <= 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
                    else:
                        print(f"  [PENDING-EXPIRE] {c} exit unfilled after {age} ticks, canceling and retrying")
                        del pending_exits[c]
                        # Will re-trigger exit check next tick since coin is still in current_longs

            # ── REGIME DETECTION ──
            avg_vol = sum(vol_pct[c] for c in UNIVERSE) / len(UNIVERSE)
            vol_history.append(avg_vol)
            if len(vol_history) > REGIME_WINDOW:
                vol_history.pop(0)
            smoothed_vol = sum(vol_history) / len(vol_history)
            regime = "ACTIVE" if smoothed_vol >= REGIME_VOL_THRESH else "FLAT"

            # ── CIRCUIT BREAKER CHECKS ──
            # Check daily loss limit
            if daily_start_equity is None:
                daily_start_equity = equity
            daily_loss = (equity / daily_start_equity - 1) if daily_start_equity > 0 else 0

            # Check max drawdown from all-time peak
            dd_from_peak = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

            if daily_loss <= -MAX_DAILY_LOSS_PCT:
                if not circuit_breaker_active:
                    print(f"  [CIRCUIT-BREAKER] Daily loss {daily_loss:.2%} exceeds limit {-MAX_DAILY_LOSS_PCT:.1%} — HALTING entries")
                circuit_breaker_active = True
            elif dd_from_peak >= MAX_DRAWDOWN_PCT:
                if not circuit_breaker_active:
                    print(f"  [CIRCUIT-BREAKER] Drawdown {dd_from_peak:.2%} exceeds limit {MAX_DRAWDOWN_PCT:.1%} — HALTING entries")
                circuit_breaker_active = True
            elif consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                if not circuit_breaker_active:
                    print(f"  [CIRCUIT-BREAKER] {consecutive_losses} consecutive losses — HALTING entries")
                circuit_breaker_active = True
            else:
                if circuit_breaker_active:
                    print(f"  [CIRCUIT-BREAKER] Conditions cleared — resuming entries")
                circuit_breaker_active = False

            # ── UPDATE BREAKOUT CONFIRMATION COUNTERS ──
            for c in UNIVERSE:
                if prices_now[c] > chan_high[c]:
                    breakout_confirm[c] += 1
                else:
                    breakout_confirm[c] = 0

            # ── CHECK ENTRIES (only in ACTIVE regime + no circuit breaker) ──
            breakout_candidates = []
            if regime == "ACTIVE" and not circuit_breaker_active:
                for c in UNIVERSE:
                    if c in current_longs:
                        continue
                    if (global_tick - last_exit.get(c, -COOLDOWN)) < COOLDOWN:
                        continue
                    if vol_pct[c] < MIN_VOL:
                        continue
                    if breakout_confirm[c] >= CONFIRM_TICKS:
                        strength = ((prices_now[c] - chan_high[c]) / atr[c]
                                    if atr[c] > 0 else 0)
                        if strength >= MIN_BREAKOUT_STR:
                            breakout_candidates.append((c, strength))

            breakout_candidates.sort(key=lambda x: x[1], reverse=True)
            slots = TOP_K - len(current_longs)

            for c, strength in breakout_candidates[:slots]:
                stop_distance = STOP_MULT * atr[c]
                if stop_distance <= 0:
                    continue

                # Risk-based sizing
                qty_by_risk = (equity * RISK_PER_TRADE) / stop_distance
                qty_by_alloc = (equity * MAX_ALLOC) / prices_now[c]
                qty = min(qty_by_risk, qty_by_alloc)

                # Budget cap (with retry)
                try:
                    bal_now = api_call_with_retry(client.get_balance)
                    wk = "SpotWallet" if "SpotWallet" in bal_now else "Wallet"
                    free_cash = bal_now.get(wk, {}).get("USD", {}).get("Free", 0.0)
                except Exception:
                    free_cash = 0.0

                fill_price = prices_now[c] * (1 + PRICE_OFFSET)
                max_qty = (free_cash * 0.95) / fill_price if fill_price > 0 else 0
                qty = min(qty, max_qty)

                pair = f"{c}/USD"
                prec = amt_precision.get(c, 4)
                p_prec = price_precision.get(c, 2)
                qty = round(qty, prec)

                if qty <= 0 or qty * prices_now[c] < MIN_ORDER_USD:
                    continue

                limit_price = round(fill_price, p_prec)
                qty_str = f"{qty:.{prec}f}"
                price_str = f"{limit_price:.{p_prec}f}"

                print(f"  [BREAKOUT] {c}: ${prices_now[c]:,.2f} > "
                      f"${chan_high[c]:,.2f} (ch_high) | "
                      f"strength={strength:.2f} vol={vol_pct[c]:.3%}")

                try:
                    api_call_with_retry(
                        client.place_order, pair=pair, side="BUY",
                        quantity=qty_str, order_type="LIMIT",
                        price=price_str)
                    print(f"    >> BUY {qty_str} {c} @ ${limit_price} (pending fill)")
                    log_trade(c, "BUY-BREAKOUT", qty, prices_now[c],
                              limit_price, atr[c], chan_high[c],
                              chan_low[c], vol_pct[c])

                    # Mark as pending — resync will confirm fill from exchange balance
                    pending_entries[c] = {
                        "qty": qty,
                        "price": prices_now[c],
                        "tick": global_tick,
                    }
                    # Optimistically reserve the slot so we don't double-buy
                    current_longs.add(c)

                except Exception as e:
                    print(f"    >> BUY {c} FAILED: {e}")

            # ── Expire stale pending entries ──
            for c in list(pending_entries):
                age = global_tick - pending_entries[c]["tick"]
                if age >= PENDING_TIMEOUT_TICKS:
                    print(f"  [PENDING-EXPIRE] {c} entry unfilled after {age} ticks, canceling")
                    try:
                        client.cancel_order(f"{c}/USD")
                    except Exception:
                        pass
                    current_longs.discard(c)
                    del pending_entries[c]

            # ── Status display ──
            for c in sorted(UNIVERSE):
                status = "LONG" if c in current_longs else "flat"
                stop_info = ""
                if c in current_longs and pos[c] > 0:
                    sl = peak_price[c] - STOP_MULT * atr[c]
                    stop_info = f" stop=${sl:,.2f}"
                cd_info = ""
                if c not in current_longs:
                    cd_remaining = COOLDOWN - (global_tick - last_exit.get(c, -COOLDOWN))
                    if cd_remaining > 0:
                        cd_info = f" cd={cd_remaining}"
                bo = "BREAK" if prices_now[c] > chan_high[c] else "     "
                stale_tag = f" STALE({stale_count[c]})" if stale_count[c] >= STALE_PRICE_TICKS else ""
                print(f"  [{c}] ${prices_now[c]:,.2f} | ch=[${chan_low[c]:,.2f}, "
                      f"${chan_high[c]:,.2f}] | vol={vol_pct[c]:.3%} "
                      f"{bo} | {status}{stop_info}{cd_info}{stale_tag}")

            # ── Re-sync equity & confirm pending fills ──
            try:
                bal = api_call_with_retry(client.get_balance)
                wallet_key = "SpotWallet" if "SpotWallet" in bal else "Wallet"
                if wallet_key in bal:
                    for c in UNIVERSE:
                        exchange_pos = bal[wallet_key].get(c, {})
                        exchange_qty = exchange_pos.get("Free", 0.0) + exchange_pos.get("Locked", 0.0)

                        # ── Confirm pending exit fills ──
                        if c in pending_exits:
                            if exchange_qty < 0.0001:  # position gone → sell filled
                                info = pending_exits.pop(c)
                                print(f"    >> [CONFIRMED] {c} exit filled")
                                # Now safe to clear position state
                                pos[c] = 0.0
                                peak_price[c] = 0.0
                                entry_price.pop(c, None)
                                current_longs.discard(c)
                                last_exit[c] = global_tick
                                # Track consecutive losses
                                if info["trade_pnl"] <= 0:
                                    consecutive_losses += 1
                                else:
                                    consecutive_losses = 0
                            else:
                                # Still holding — sell not filled yet
                                pos[c] = exchange_qty

                        # ── Confirm pending entry fills ──
                        elif c in pending_entries:
                            if exchange_qty > 0.0001:  # got coins → buy filled
                                info = pending_entries.pop(c)
                                print(f"    >> [CONFIRMED] {c} entry filled ({exchange_qty:.6f})")
                                pos[c] = exchange_qty
                                entry_price[c] = info["price"]
                                peak_price[c] = info["price"]
                                # current_longs already has c (added optimistically)
                            else:
                                # No coins yet — buy not filled
                                pos[c] = 0.0

                        # ── Active tracked position ──
                        elif c in current_longs:
                            pos[c] = exchange_qty

                        # ── Untracked dust cleanup ──
                        elif exchange_qty > 0 and exchange_qty * prices_now[c] > MIN_ORDER_USD:
                            pair = f"{c}/USD"
                            prec = amt_precision.get(c, 4)
                            p_prec = price_precision.get(c, 2)
                            dust_qty = round(exchange_qty, prec)
                            if dust_qty > 0:
                                try:
                                    limit_price = round(prices_now[c] * (1 - PRICE_OFFSET), p_prec)
                                    qty_str = f"{dust_qty:.{prec}f}"
                                    price_str = f"{limit_price:.{p_prec}f}"
                                    client.place_order(pair=pair, side="SELL",
                                                       quantity=qty_str, order_type="LIMIT",
                                                       price=price_str)
                                    print(f"  [DUST-CLEANUP] Sold {qty_str} {c} @ ${limit_price}")
                                except Exception:
                                    pass
                            pos[c] = 0.0
                        else:
                            pos[c] = 0.0

                    usd_entry = bal[wallet_key].get("USD", {})
                    usd = usd_entry.get("Free", 0.0) + usd_entry.get("Locked", 0.0)

                    coin_val = sum(pos[c] * prices_now[c] for c in UNIVERSE)
                    equity = usd + coin_val
                    peak_equity = max(peak_equity, equity)
                    pnl = (equity / start_equity - 1) * 100
                    dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

                    cb_tag = " [CB-HALT]" if circuit_breaker_active else ""
                    loss_tag = f" losses={consecutive_losses}" if consecutive_losses > 0 else ""
                    pending_tag = ""
                    if pending_exits or pending_entries:
                        pe = [f"{c}:exit" for c in pending_exits]
                        pb = [f"{c}:buy" for c in pending_entries]
                        pending_tag = f" | Pending: {', '.join(pe + pb)}"
                    print(f"\n{'=' * 60}")
                    print(f"EQUITY: ${equity:,.2f} | P&L: {pnl:+.2f}% "
                          f"| DD: {dd:.1%} | Tick: {global_tick}{cb_tag}")
                    print(f"Holdings: {', '.join(sorted(current_longs)) if current_longs else 'ALL CASH'}"
                          f" | Regime: {regime} (vol={smoothed_vol:.3%}){loss_tag}{pending_tag}")
                    print(f"{'=' * 60}\n")
            except Exception as e:
                print(f"[RESYNC] Failed: {e}")

            time.sleep(30)

    finally:
        pnl_final = (equity / start_equity - 1) * 100
        print(f"\n{'=' * 60}")
        print(f"[SHUTDOWN] Final equity: ${equity:,.2f} | P&L: {pnl_final:+.2f}%")
        print(f"[SHUTDOWN] Positions: {pos}")
        print(f"[SHUTDOWN] Ticks: {global_tick}")
        print(f"[SHUTDOWN] Circuit breaker was active: {circuit_breaker_active}")
        print(f"[SHUTDOWN] Consecutive losses at exit: {consecutive_losses}")
        print(f"{'=' * 60}")
        log_fp.close()
        print("[SHUTDOWN] Trade log saved and closed.")
        # Final health log
        try:
            log_health()
        except Exception:
            pass


if __name__ == '__main__':
    run_breakout_bot()
