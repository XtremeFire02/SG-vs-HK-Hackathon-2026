import time
import os
import csv
import signal
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from roostoo_client import RoostooClient

API_KEY = os.getenv("API_KEY")
SECRET = os.getenv("SECRET")

client = RoostooClient(api_key=API_KEY, secret_key=SECRET)

# ─── STRATEGY: MULTI-FACTOR ENSEMBLE REGIME-SWITCHING BOT ────────────────────
#
#  Regime detection  (per-coin, smoothed to avoid thrashing)
#    - Bollinger Band width = (upper - lower) / mean = 4σ / mean
#    - Wide bands  (BB_width > BREAKOUT_THRESH) → TRENDING market
#    - Narrow bands (BB_width ≤ BREAKOUT_THRESH) → RANGING  market
#
#  TRENDING  → Momentum strategy (gated by RSI overbought filter)
#    - EMA crossover (fast vs slow) sets direction
#    - BTC regime filter: only go long when BTC is also trending up
#    - RSI gate: block new longs when overbought
#    - Gradual exit on signal loss (halve toward target, don't dump)
#
#  RANGING   → Mean Reversion strategy (gated by RSI + bandwidth filter)
#    - Bollinger Bands on 20-period rolling window
#    - Position = full_size × (upper - price) / (upper - lower)
#    - RSI gate: only buy near lower band when RSI confirms oversold
#    - Bandwidth gate: skip when bands too narrow to profit
#    - Emergency exit: if MR-SKIP but position losing > threshold, exit
#
#  Risk management
#    - Vol scaling: inverted for MR (scale DOWN when vol is low)
#    - Dynamic sizing based on current equity (not initial capital)
#    - Drawdown-based position scaling (reduce exposure as losses mount)
#    - Per-coin max exposure cap prevents concentration
#    - Minimum order notional prevents dust trades
#    - Trade cooldown prevents churn on same coin
#    - Per-cycle budget cap prevents over-allocation
#    - Cancel-before-recompute prevents stale order fills
#    - Warmup ramp avoids noisy early-EMA signals
#    - Graceful shutdown with final state logging
#


def compute_rsi(prices, period=14):
    """SMA-based RSI from raw price list. Returns 50.0 if not enough data or flat."""
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(len(prices) - period, len(prices))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    # FIX #1: flat market (no movement) → neutral RSI, not 100
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def run_portfolio_bot():
    print("=" * 60)
    print("  MULTI-FACTOR ENSEMBLE REGIME-SWITCHING BOT")
    print("  Trending market → EMA Momentum + RSI filter")
    print("  Ranging  market → Mean Reversion (BB + RSI)")
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
    UNIVERSE   = ["BTC", "ETH", "SOL", "BNB"]
    FAST       = 5        # fast EMA period (ticks)
    SLOW       = 20       # slow EMA period (ticks)
    VOL_WIN    = 20       # realized vol lookback
    TARGET_VOL = 0.0002   # per-tick vol target (~15% annualized)
    MAX_SCALE  = 1.5      # cap on vol scalar
    ALLOC      = 0.20     # base allocation per coin (fraction of equity)

    # Regime detection
    REGIME_WIN      = 20    # BB window for regime detection
    BREAKOUT_THRESH = 0.025 # BB width > 2.5% → trending; ≤ 2.5% → ranging
    REGIME_SMOOTH   = 0.2   # EMA alpha for smoothing regime signal

    # Mean reversion
    MR_BB_K   = 2.0   # Bollinger Band standard-deviation multiplier
    MR_ALLOC  = 0.15  # slightly smaller allocation for mean reversion
    MR_MIN_BW = 0.002  # minimum BB width (0.2%) — need 5x cost-to-bandwidth ratio
    MR_SKIP_EXIT_PCT = 0.02  # exit MR-SKIP position if losing more than 2%

    # RSI (multi-factor confluence filter)
    RSI_WIN        = 14   # lookback period for RSI
    RSI_OVERBOUGHT = 75   # trending: block buys above this
    RSI_OVERSOLD_MR = 55  # ranging: only require RSI confluence near lower band

    # Risk management
    DRAWDOWN_START  = 0.03   # start reducing size at 3% drawdown from peak
    DRAWDOWN_FLOOR  = 0.20   # minimum scalar (20% of normal size) at ~19%+ drawdown
    PRICE_OFFSET    = 0.0002 # 0.02% limit price offset for better fills
    WARMUP_RAMP     = 10     # ticks after warmup to ramp to full size
    MAX_COIN_ALLOC  = 0.30   # hard cap: never put more than 30% equity in one coin
    MIN_ORDER_USD   = 150.00 # minimum order notional to avoid dust trades
    TRADE_COOLDOWN  = 40     # minimum cycles between trades on same coin (~20 min)

    k_f = 2.0 / (FAST + 1)
    k_s = 2.0 / (SLOW + 1)

    HIST_KEEP = max(VOL_WIN, REGIME_WIN, SLOW, RSI_WIN) + 10

    # ── Per-coin state ──
    ema_f      = {}   # fast EMA
    ema_s      = {}   # slow EMA
    hist       = {}   # price history
    ticks      = {}   # tick count
    vol_ema    = {}   # smoothed vol scalar
    regime_ema = {}   # smoothed BB-width for regime detection
    last_trade = {}   # tick of last trade per coin (cooldown tracking)
    entry_price = {}  # price when position was acquired (for MR-SKIP stop-loss)

    # ── Sync with exchange ──
    print("\nSyncing with exchange...")
    pos = {c: 0.0 for c in UNIVERSE}

    wallet_key = "SpotWallet" if "SpotWallet" in bal else "Wallet"
    if wallet_key in bal:
        for c in UNIVERSE:
            # FIX #7: capture Free + Locked on init, matching resync behavior
            c_entry = bal[wallet_key].get(c, {})
            pos[c] = c_entry.get("Free", 0.0) + c_entry.get("Locked", 0.0)
        usd_entry = bal[wallet_key].get("USD", {})
        free_usd = usd_entry.get("Free", 0.0) + usd_entry.get("Locked", 0.0)
    else:
        free_usd = 50000.0

    # Include coin value in initial equity so P&L baseline is correct
    start_usd = free_usd
    try:
        tk_init = client.get_ticker()
        if tk_init and "Data" in tk_init:
            for c in UNIVERSE:
                lp = tk_init["Data"].get(f"{c}/USD", {}).get("LastPrice", 0)
                coin_price = float(lp)
                start_usd += pos[c] * coin_price
                if pos[c] > 0 and coin_price > 0:
                    entry_price[c] = coin_price
    except Exception:
        pass

    equity = start_usd
    free_usd_balance = free_usd
    peak_equity = start_usd

    print(f"  Starting equity:    ${start_usd:,.2f} (USD + coin value)")
    print(f"  Starting positions: {pos}")
    print(f"  Strategy:           Multi-Factor Ensemble (Momentum | Mean Reversion)")
    print(f"  RSI filter:         Overbought > {RSI_OVERBOUGHT}, MR oversold < {RSI_OVERSOLD_MR}")
    print(f"  Regime threshold:   BB_width > {BREAKOUT_THRESH:.1%} = TRENDING")
    print(f"  Max coin exposure:  {MAX_COIN_ALLOC:.0%}")
    print(f"  Drawdown scaling:   starts at {DRAWDOWN_START:.0%}, floor at {DRAWDOWN_FLOOR:.0%}")
    print("-" * 60)

    # ── CSV Audit Log ──
    LOG_FILE = "trade_log.csv"
    log_exists = os.path.exists(LOG_FILE)
    log_fp = open(LOG_FILE, "a", newline="")
    log_writer = csv.writer(log_fp)
    if not log_exists:
        log_writer.writerow([
            "timestamp", "coin", "regime", "strategy", "rsi",
            "signal", "qty", "price", "limit_price",
            "equity", "pnl_pct", "bb_width", "vol_scalar", "dd_scalar",
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

    cycle_count = 0

    # ── Main loop ──
    try:
        while not shutdown_requested:
            cycle_count += 1

            # ── Drawdown-based position scaling ──
            drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
            dd_scalar = 1.0
            if drawdown > DRAWDOWN_START:
                dd_scalar = max(DRAWDOWN_FLOOR, 1.0 - (drawdown - DRAWDOWN_START) * 5)

            # Resync free_usd at TOP of cycle so cycle_budget is fresh
            try:
                bal = client.get_balance()
                wallet_key = "SpotWallet" if "SpotWallet" in bal else "Wallet"
                if wallet_key in bal:
                    usd_entry = bal[wallet_key].get("USD", {})
                    free_usd_balance = usd_entry.get("Free", 0.0)
            except Exception:
                pass

            cycle_budget = free_usd_balance

            if drawdown > DRAWDOWN_START:
                print(f"[RISK] Drawdown {drawdown:.1%} from peak ${peak_equity:,.2f} "
                      f"| Size scalar: {dd_scalar:.0%}")

            for coin in UNIVERSE:
                try:
                    pair = f"{coin}/USD"

                    try:
                        client.cancel_order(pair)
                    except Exception:
                        pass

                    td = client.get_ticker(pair)
                    if not (td and "Data" in td and pair in td["Data"]):
                        continue

                    price = float(td["Data"][pair]["LastPrice"])
                    if price <= 0:
                        continue

                    # First tick initialization
                    if coin not in ema_f:
                        ema_f[coin]      = price
                        ema_s[coin]      = price
                        hist[coin]       = []
                        ticks[coin]      = 0
                        # FIX #5: initialize regime_ema to None, seed on first computation
                        regime_ema[coin] = None
                        last_trade[coin] = -TRADE_COOLDOWN

                    # Update EMAs and history
                    ema_f[coin] = price * k_f + ema_f[coin] * (1 - k_f)
                    ema_s[coin] = price * k_s + ema_s[coin] * (1 - k_s)
                    hist[coin].append(price)
                    ticks[coin] += 1

                    if len(hist[coin]) > HIST_KEEP:
                        hist[coin] = hist[coin][-HIST_KEEP:]

                    signal_dir = "HOLD"
                    target_qty = 0.0
                    rsi = 50.0
                    strat_tag = "WARMUP"
                    regime_lbl = "INIT"
                    bb_width = 0.0
                    vol_scalar = 1.0

                    if ticks[coin] >= SLOW:
                        ph = hist[coin]

                        # ─── REGIME DETECTION ───────────────────────────────────
                        rp   = ph[-REGIME_WIN:]
                        rm   = sum(rp) / len(rp)
                        rstd = (sum((p - rm) ** 2 for p in rp) / len(rp)) ** 0.5
                        raw_bb_width = (4 * rstd) / rm if rm > 0 else 0.0

                        # FIX #5: seed regime_ema on first computation
                        if regime_ema[coin] is None:
                            regime_ema[coin] = raw_bb_width
                        else:
                            regime_ema[coin] = (REGIME_SMOOTH * raw_bb_width
                                                + (1 - REGIME_SMOOTH) * regime_ema[coin])
                        bb_width    = regime_ema[coin]
                        is_trending = bb_width > BREAKOUT_THRESH
                        regime_lbl  = "TREND" if is_trending else "RANGE"

                        # ─── VOL SCALING ────────────────────────────────────────
                        vol_scalar = 1.0
                        if len(ph) >= VOL_WIN + 1:
                            rets = [(ph[j] - ph[j-1]) / ph[j-1]
                                    for j in range(len(ph) - VOL_WIN, len(ph))]
                            avg = sum(rets) / len(rets)
                            var = sum((r - avg) ** 2 for r in rets) / len(rets)
                            vol = var ** 0.5
                            raw_scalar = min(TARGET_VOL / max(vol, 1e-10), MAX_SCALE)
                            if coin not in vol_ema:
                                vol_ema[coin] = raw_scalar
                            else:
                                vol_ema[coin] = 0.1 * raw_scalar + 0.9 * vol_ema[coin]
                            vol_scalar = vol_ema[coin]

                        # ─── WARMUP RAMP ────────────────────────────────────────
                        ticks_past = ticks[coin] - SLOW
                        warmup_scalar = min(1.0, 0.25 + 0.75 * ticks_past / WARMUP_RAMP)

                        # ─── RSI (MULTI-FACTOR FILTER) ──────────────────────────
                        rsi = compute_rsi(ph, RSI_WIN)

                        # ─── STRATEGY SELECTION ─────────────────────────────────
                        if is_trending:
                            # ── MOMENTUM STRATEGY ───────────────────────────────
                            # Combined scalar uses vol (trend: low vol = safe to size up)
                            combined = vol_scalar * dd_scalar * warmup_scalar

                            momentum = ema_f[coin] > ema_s[coin]
                            btc_ok = ema_f.get("BTC", 0) > ema_s.get("BTC", 1)
                            go_long = momentum and btc_ok

                            base_qty = equity * ALLOC / price

                            if go_long and rsi >= RSI_OVERBOUGHT:
                                target_qty = pos[coin]
                                strat_tag  = "LONG-RSI_BLOCK"
                            elif go_long:
                                target_qty = base_qty * combined
                                strat_tag  = "LONG"
                            else:
                                # FIX #6: gradual exit — halve toward zero, don't dump
                                target_qty = pos[coin] * 0.5
                                strat_tag  = "FLAT-FADE"

                        else:
                            # ── MEAN REVERSION STRATEGY ─────────────────────────
                            # FIX #2: for MR, clamp vol_scalar to ≤1.0
                            # Low vol = narrow bands = less profit → don't size up
                            mr_vol = min(vol_scalar, 1.0)
                            combined = mr_vol * dd_scalar * warmup_scalar

                            if bb_width < MR_MIN_BW:
                                # FIX #3: allow emergency exit if position is losing
                                ep = entry_price.get(coin)
                                if pos[coin] > 0 and ep and ep > 0:
                                    pnl_since_entry = (price - ep) / ep
                                    if pnl_since_entry < -MR_SKIP_EXIT_PCT:
                                        target_qty = 0.0
                                        strat_tag  = f"MR-STOP({pnl_since_entry:+.1%})"
                                    else:
                                        target_qty = pos[coin]
                                        strat_tag  = "MR-SKIP"
                                else:
                                    target_qty = pos[coin]
                                    strat_tag  = "MR-SKIP"
                            else:
                                bp     = ph[-SLOW:]
                                bb_mid = sum(bp) / len(bp)
                                bb_std = (sum((p - bb_mid) ** 2 for p in bp) / len(bp)) ** 0.5
                                bb_up  = bb_mid + MR_BB_K * bb_std
                                bb_dn  = bb_mid - MR_BB_K * bb_std

                                bb_pos = (bb_up - price) / max(bb_up - bb_dn, 1e-10)
                                bb_pos = max(0.0, min(1.0, bb_pos))

                                base_qty = equity * MR_ALLOC / price

                                # FIX #4: only require RSI confluence for strong buys
                                # (bb_pos > 0.7 = near lower band). Moderate positions
                                # don't need RSI confirmation. Threshold raised to 55.
                                if bb_pos > 0.7 and rsi >= RSI_OVERSOLD_MR:
                                    target_qty = pos[coin]
                                    strat_tag  = "MR-RSI_BLOCK"
                                else:
                                    target_qty = base_qty * combined * bb_pos
                                    if price <= bb_dn:
                                        strat_tag = "MR-BUY"
                                    elif price >= bb_up:
                                        strat_tag = "MR-SELL"
                                    else:
                                        strat_tag = f"MR({bb_pos:.0%})"

                        # Cap max position per coin
                        max_qty = (equity * MAX_COIN_ALLOC) / price
                        target_qty = min(target_qty, max_qty)

                        # ─── EXECUTE ────────────────────────────────────────────
                        trade_qty = target_qty - pos[coin]
                        print(f"[{coin}] ${price:,.2f} | {regime_lbl} | {strat_tag} | "
                              f"RSI={rsi:.1f} | vol={vol_scalar:.2f}x dd={dd_scalar:.0%} | "
                              f"BB_w={bb_width:.4f} | pos={pos[coin]:.6f} -> {target_qty:.6f}")

                        # Dead-band: skip tiny rebalances (saves fees)
                        min_rebalance = max(abs(target_qty) * 0.15, 0.0001)
                        if trade_qty > min_rebalance:
                            signal_dir = "BUY"
                        elif trade_qty < -min_rebalance:
                            signal_dir = "SELL"
                            trade_qty = abs(trade_qty)

                    else:
                        print(f"[{coin}] ${price:,.2f} | warming up {ticks[coin]}/{SLOW}")

                    # ── Place order ──
                    if signal_dir in ("BUY", "SELL"):
                        # Minimum order notional check
                        order_notional = abs(trade_qty) * price
                        if order_notional < MIN_ORDER_USD:
                            print(f"    >> {signal_dir} {coin} skipped: "
                                  f"${order_notional:.2f} < ${MIN_ORDER_USD} min notional")
                            time.sleep(1)
                            continue

                        # Trade cooldown — prevent churn
                        ticks_since = ticks.get(coin, 0) - last_trade.get(coin, -TRADE_COOLDOWN)
                        if ticks_since < TRADE_COOLDOWN:
                            print(f"    >> {signal_dir} {coin} skipped: "
                                  f"cooldown ({ticks_since}/{TRADE_COOLDOWN} ticks)")
                            time.sleep(1)
                            continue

                        prec   = amt_precision.get(coin, 4)
                        p_prec = price_precision.get(coin, 2)

                        # Cap BUY orders against remaining cycle budget
                        if signal_dir == "BUY":
                            max_affordable = cycle_budget * 0.95 / price
                            trade_qty = min(trade_qty, max_affordable)

                        trade_qty = round(trade_qty, prec)
                        if trade_qty > 0:
                            qty_str = f"{trade_qty:.{prec}f}"

                            if signal_dir == "BUY":
                                limit_price = round(price * (1 + PRICE_OFFSET), p_prec)
                            else:
                                limit_price = round(price * (1 - PRICE_OFFSET), p_prec)
                            price_str = f"{limit_price:.{p_prec}f}"

                            try:
                                client.place_order(
                                    pair=pair,
                                    side=signal_dir,
                                    quantity=qty_str,
                                    order_type="LIMIT",
                                    price=price_str,
                                )
                                print(f"    >> {signal_dir} {qty_str} {coin} "
                                      f"@ ${limit_price} (LIMIT) — OK")

                                last_trade[coin] = ticks[coin]

                                # Track entry price for stop-loss
                                if signal_dir == "BUY":
                                    entry_price[coin] = price
                                elif signal_dir == "SELL" and trade_qty >= pos[coin] * 0.9:
                                    # Full exit — clear entry price
                                    entry_price.pop(coin, None)

                                pnl_now = (equity / start_usd - 1) * 100
                                log_writer.writerow([
                                    datetime.now(timezone.utc).isoformat(),
                                    coin, regime_lbl, strat_tag, f"{rsi:.1f}",
                                    signal_dir, qty_str, f"{price:.2f}",
                                    f"{limit_price}", f"{equity:.2f}",
                                    f"{pnl_now:.2f}", f"{bb_width:.6f}",
                                    f"{vol_scalar:.4f}", f"{dd_scalar:.2f}",
                                ])
                                log_fp.flush()

                                if signal_dir == "BUY":
                                    cycle_budget -= trade_qty * limit_price
                            except Exception as e:
                                print(f"    >> {signal_dir} {qty_str} {coin} — FAILED: {e}")

                    time.sleep(1)

                except Exception as e:
                    print(f"[{coin}] ERROR: {e}")

            # ── Re-sync positions & report P&L ──
            try:
                bal = client.get_balance()
                wallet_key = "SpotWallet" if "SpotWallet" in bal else "Wallet"
                if wallet_key in bal:
                    for c in UNIVERSE:
                        entry  = bal[wallet_key].get(c, {})
                        pos[c] = entry.get("Free", 0.0) + entry.get("Locked", 0.0)

                    usd_entry = bal[wallet_key].get("USD", {})
                    free_usd_balance = usd_entry.get("Free", 0.0)
                    usd = free_usd_balance + usd_entry.get("Locked", 0.0)

                    tk = client.get_ticker()
                    if tk and "Data" in tk:
                        coin_val = sum(
                            pos[c] * float(
                                tk["Data"].get(f"{c}/USD", {}).get("LastPrice", 0)
                            )
                            for c in UNIVERSE
                        )
                        equity = usd + coin_val
                        peak_equity = max(peak_equity, equity)
                        pnl = (equity / start_usd - 1) * 100
                        dd  = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

                        regimes = {
                            c: ("TREND" if (regime_ema.get(c) or 0) > BREAKOUT_THRESH
                                else "RANGE")
                            for c in UNIVERSE if c in regime_ema
                        }

                        print(f"\n{'=' * 60}")
                        print(f"EQUITY: ${equity:,.2f} | P&L: {pnl:+.2f}% "
                              f"| Peak: ${peak_equity:,.2f} | DD: {dd:.1%}")
                        print(f"Cash: ${usd:,.2f} | Coins: ${coin_val:,.2f} "
                              f"| Cycle: {cycle_count}")
                        if regimes:
                            print(f"Regimes: {regimes}")
                        print(f"{'=' * 60}\n")
            except Exception as e:
                print(f"[RESYNC] Failed to sync positions: {e}")

            time.sleep(30)

    finally:
        print(f"\n{'=' * 60}")
        print(f"[SHUTDOWN] Final equity: ${equity:,.2f} | "
              f"P&L: {(equity / start_usd - 1) * 100:+.2f}%")
        print(f"[SHUTDOWN] Positions: {pos}")
        print(f"[SHUTDOWN] Cycles completed: {cycle_count}")
        print(f"{'=' * 60}")
        log_fp.close()
        print("[SHUTDOWN] Trade log saved and closed.")


if __name__ == '__main__':
    run_portfolio_bot()
