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
#  Backtested Sharpe: +0.73 (7d flat), +6.70 (1mo), -0.09 (60d)
#  Profit Factor: 1.26 / 1.77 / 1.00
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
    TOP_K          = 2      # max simultaneous positions
    RISK_PER_TRADE = 0.01   # risk 1% of equity per trade
    MAX_ALLOC      = 0.25   # never put more than 25% equity in one coin
    STOP_MULT      = 2.0    # trailing stop = peak - STOP_MULT * ATR
    MIN_VOL        = 0.00005 # minimum ATR/price to allow entry — optimized via sweep
    PRICE_OFFSET   = 0.0002 # 0.02% limit price offset
    COOLDOWN       = 20     # ticks after stop-out before re-entry (~10 min)
    MIN_ORDER_USD  = 50.00  # minimum order notional

    HIST_KEEP = ENTRY_LB + ATR_WIN + 10

    # ── Per-coin state ──
    hist       = {}
    ticks      = {c: 0 for c in UNIVERSE}
    pos        = {c: 0.0 for c in UNIVERSE}
    peak_price = {c: 0.0 for c in UNIVERSE}
    entry_price = {}
    current_longs = set()
    last_exit  = {c: -COOLDOWN for c in UNIVERSE}

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

    # ── Main loop ──
    try:
        while not shutdown_requested:
            global_tick += 1

            # Cancel stale orders
            for coin in UNIVERSE:
                try:
                    client.cancel_order(f"{coin}/USD")
                except Exception:
                    pass

            # Fetch prices
            prices_now = {}
            for coin in UNIVERSE:
                try:
                    td = client.get_ticker(f"{coin}/USD")
                    if td and "Data" in td and f"{coin}/USD" in td["Data"]:
                        prices_now[coin] = float(td["Data"][f"{coin}/USD"]["LastPrice"])
                except Exception:
                    pass

            if len(prices_now) < len(UNIVERSE):
                print(f"[TICK {global_tick}] Missing price data, skipping")
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

            # ── CHECK EXITS ──
            for c in list(current_longs):
                if pos[c] <= 0:
                    current_longs.discard(c)
                    continue

                stop_level = peak_price[c] - STOP_MULT * atr[c]
                exit_channel = chan_low[c]
                exit_level = max(stop_level, exit_channel)

                if prices_now[c] <= exit_level:
                    reason = "STOP" if prices_now[c] <= stop_level else "CHAN-EXIT"
                    pair = f"{c}/USD"
                    prec = amt_precision.get(c, 4)
                    p_prec = price_precision.get(c, 2)
                    qty = round(pos[c], prec)

                    if qty > 0:
                        limit_price = round(prices_now[c] * (1 - PRICE_OFFSET), p_prec)
                        qty_str = f"{qty:.{prec}f}"
                        price_str = f"{limit_price:.{p_prec}f}"

                        pnl_trade = ""
                        ep = entry_price.get(c)
                        if ep and ep > 0:
                            pnl_trade = f" P&L: {(prices_now[c]/ep - 1)*100:+.2f}%"

                        print(f"  [{reason}] {c}: ${prices_now[c]:,.2f} <= "
                              f"${exit_level:,.2f}{pnl_trade}")

                        try:
                            client.place_order(pair=pair, side="SELL",
                                               quantity=qty_str, order_type="LIMIT",
                                               price=price_str)
                            print(f"    >> SELL-{reason} {qty_str} {c} @ ${limit_price}")
                            log_trade(c, f"SELL-{reason}", qty, prices_now[c],
                                      limit_price, atr[c], chan_high[c],
                                      chan_low[c], vol_pct[c])
                        except Exception as e:
                            print(f"    >> SELL-{reason} {c} FAILED: {e}")

                    pos[c] = 0.0
                    peak_price[c] = 0.0
                    entry_price.pop(c, None)
                    current_longs.discard(c)
                    last_exit[c] = global_tick

            # ── CHECK ENTRIES ──
            breakout_candidates = []
            for c in UNIVERSE:
                if c in current_longs:
                    continue
                if (global_tick - last_exit.get(c, -COOLDOWN)) < COOLDOWN:
                    continue
                if vol_pct[c] < MIN_VOL:
                    continue
                if prices_now[c] > chan_high[c]:
                    strength = ((prices_now[c] - chan_high[c]) / atr[c]
                                if atr[c] > 0 else 0)
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

                # Budget cap
                try:
                    bal_now = client.get_balance()
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
                    client.place_order(pair=pair, side="BUY",
                                       quantity=qty_str, order_type="LIMIT",
                                       price=price_str)
                    print(f"    >> BUY {qty_str} {c} @ ${limit_price}")
                    log_trade(c, "BUY-BREAKOUT", qty, prices_now[c],
                              limit_price, atr[c], chan_high[c],
                              chan_low[c], vol_pct[c])
                    pos[c] += qty
                    entry_price[c] = prices_now[c]
                    peak_price[c] = prices_now[c]
                    current_longs.add(c)
                except Exception as e:
                    print(f"    >> BUY {c} FAILED: {e}")

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
                print(f"  [{c}] ${prices_now[c]:,.2f} | ch=[${chan_low[c]:,.2f}, "
                      f"${chan_high[c]:,.2f}] | vol={vol_pct[c]:.3%} "
                      f"{bo} | {status}{stop_info}{cd_info}")

            # ── Re-sync equity ──
            try:
                bal = client.get_balance()
                wallet_key = "SpotWallet" if "SpotWallet" in bal else "Wallet"
                if wallet_key in bal:
                    for c in UNIVERSE:
                        entry = bal[wallet_key].get(c, {})
                        pos[c] = entry.get("Free", 0.0) + entry.get("Locked", 0.0)

                    usd_entry = bal[wallet_key].get("USD", {})
                    usd = usd_entry.get("Free", 0.0) + usd_entry.get("Locked", 0.0)

                    coin_val = sum(pos[c] * prices_now[c] for c in UNIVERSE)
                    equity = usd + coin_val
                    peak_equity = max(peak_equity, equity)
                    pnl = (equity / start_equity - 1) * 100
                    dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

                    print(f"\n{'=' * 60}")
                    print(f"EQUITY: ${equity:,.2f} | P&L: {pnl:+.2f}% "
                          f"| DD: {dd:.1%} | Tick: {global_tick}")
                    print(f"Holdings: {', '.join(sorted(current_longs)) if current_longs else 'ALL CASH'}")
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
        print(f"{'=' * 60}")
        log_fp.close()
        print("[SHUTDOWN] Trade log saved and closed.")


if __name__ == '__main__':
    run_breakout_bot()
