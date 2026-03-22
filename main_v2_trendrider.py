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

# ─── STRATEGY: ADAPTIVE TREND RIDER WITH TRAILING STOPS ──────────────────────
#
#  Core idea: Don't predict. React. Wait for moves to prove themselves, ride them.
#
#  Signal generation:
#    - Momentum = return over LOOKBACK ticks per coin
#    - Rank coins by momentum, go long top TOP_K with positive momentum
#    - Size by inverse ATR (risk parity): volatile coins get smaller positions
#
#  Risk management:
#    - Trailing stop at STOP_ATR_MULT * ATR below peak price
#    - Rebalance cooldown prevents churn
#    - Min notional prevents dust trades
#    - Budget cap prevents over-allocation
#    - Graceful shutdown with final state logging
#


def run_trend_rider():
    print("=" * 60)
    print("  ADAPTIVE TREND RIDER")
    print("  Long top-K momentum coins | Trailing ATR stops")
    print("  Risk parity sizing | Rebalance on rank change")
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
    LOOKBACK       = 60     # momentum lookback (ticks, ~30 min at 30s cycles)
    ATR_WIN        = 20     # ATR window for vol sizing and stops
    TOP_K          = 2      # number of coins to hold long
    ALLOC          = 0.40   # total portfolio allocation (split among TOP_K)
    STOP_ATR_MULT  = 2.0    # trailing stop = peak - STOP_ATR_MULT * ATR
    REBAL_COOLDOWN = 30     # minimum ticks between rebalances (~15 min)
    PRICE_OFFSET   = 0.0002 # slippage per trade (0.02%)
    MIN_MOMENTUM   = 0.001  # minimum momentum (0.1%) to enter
    MIN_ORDER_USD  = 50.00  # minimum order notional

    # ── Per-coin state ──
    hist       = {}
    ticks      = {c: 0 for c in UNIVERSE}
    pos        = {c: 0.0 for c in UNIVERSE}
    peak_price = {c: 0.0 for c in UNIVERSE}
    entry_price = {}
    current_longs = set()
    last_rebal = -REBAL_COOLDOWN

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

    # Include coin value in starting equity
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
    print(f"  Strategy:           Trend Rider (LOOKBACK={LOOKBACK}, TOP_K={TOP_K})")
    print(f"  Trailing stop:      {STOP_ATR_MULT}x ATR")
    print(f"  Allocation:         {ALLOC:.0%} total, risk-parity sized")
    print(f"  Rebal cooldown:     {REBAL_COOLDOWN} ticks")
    print("-" * 60)

    # ── CSV Audit Log ──
    LOG_FILE = "trade_log.csv"
    log_exists = os.path.exists(LOG_FILE)
    log_fp = open(LOG_FILE, "a", newline="")
    log_writer = csv.writer(log_fp)
    if not log_exists:
        log_writer.writerow([
            "timestamp", "coin", "side", "qty", "price", "limit_price",
            "momentum", "atr", "equity", "pnl_pct",
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

    def log_trade(coin, side, qty, price, limit_price, mom, atr_val):
        pnl_now = (equity / start_equity - 1) * 100
        log_writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            coin, side, f"{qty:.6f}", f"{price:.2f}", f"{limit_price:.2f}",
            f"{mom:.4f}", f"{atr_val:.2f}", f"{equity:.2f}", f"{pnl_now:.2f}",
        ])
        log_fp.flush()

    def place_sell(coin, qty, price, side_label, mom, atr_val):
        """Place a sell order. Returns True on success."""
        pair = f"{coin}/USD"
        prec = amt_precision.get(coin, 4)
        p_prec = price_precision.get(coin, 2)
        qty = round(qty, prec)
        if qty <= 0:
            return False
        limit_price = round(price * (1 - PRICE_OFFSET), p_prec)
        qty_str = f"{qty:.{prec}f}"
        price_str = f"{limit_price:.{p_prec}f}"
        try:
            client.place_order(pair=pair, side="SELL", quantity=qty_str,
                               order_type="LIMIT", price=price_str)
            print(f"    >> {side_label} {qty_str} {coin} @ ${limit_price}")
            log_trade(coin, side_label, qty, price, limit_price, mom, atr_val)
            return True
        except Exception as e:
            print(f"    >> {side_label} {coin} FAILED: {e}")
            return False

    def place_buy(coin, qty, price, mom, atr_val):
        """Place a buy order. Returns True on success."""
        pair = f"{coin}/USD"
        prec = amt_precision.get(coin, 4)
        p_prec = price_precision.get(coin, 2)
        qty = round(qty, prec)
        if qty <= 0:
            return False
        limit_price = round(price * (1 + PRICE_OFFSET), p_prec)
        qty_str = f"{qty:.{prec}f}"
        price_str = f"{limit_price:.{p_prec}f}"
        try:
            client.place_order(pair=pair, side="BUY", quantity=qty_str,
                               order_type="LIMIT", price=price_str)
            print(f"    >> BUY {qty_str} {coin} @ ${limit_price}")
            log_trade(coin, "BUY", qty, price, limit_price, mom, atr_val)
            return True
        except Exception as e:
            print(f"    >> BUY {coin} FAILED: {e}")
            return False

    # ── Main loop ──
    try:
        while not shutdown_requested:
            global_tick += 1

            # Cancel all stale orders
            for coin in UNIVERSE:
                try:
                    client.cancel_order(f"{coin}/USD")
                except Exception:
                    pass

            # Fetch all prices
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
                if len(hist[coin]) > LOOKBACK + ATR_WIN + 10:
                    hist[coin] = hist[coin][-(LOOKBACK + ATR_WIN + 10):]

            # Update trailing stop peaks
            for c in current_longs:
                if prices_now[c] > peak_price[c]:
                    peak_price[c] = prices_now[c]

            # Warmup
            min_ticks = min(ticks[c] for c in UNIVERSE)
            if min_ticks < LOOKBACK:
                print(f"[TICK {global_tick}] Warming up {min_ticks}/{LOOKBACK}")
                time.sleep(30)
                continue

            # ── Compute ATR per coin ──
            atr = {}
            for c in UNIVERSE:
                ph = hist[c]
                diffs = [abs(ph[j] - ph[j - 1]) for j in range(len(ph) - ATR_WIN, len(ph))]
                atr[c] = sum(diffs) / len(diffs) if diffs else 1e-10

            # ── Check trailing stops ──
            stopped_out = set()
            for c in list(current_longs):
                if pos[c] > 0 and atr[c] > 0:
                    stop_level = peak_price[c] - STOP_ATR_MULT * atr[c]
                    if prices_now[c] <= stop_level:
                        print(f"  [STOP] {c} hit trailing stop: "
                              f"${prices_now[c]:,.2f} <= ${stop_level:,.2f}")
                        mom = (prices_now[c] - hist[c][-LOOKBACK]) / hist[c][-LOOKBACK]
                        if place_sell(c, pos[c], prices_now[c], "SELL-STOP", mom, atr[c]):
                            pos[c] = 0.0
                            peak_price[c] = 0.0
                            entry_price.pop(c, None)
                            stopped_out.add(c)
                            current_longs.discard(c)

            # ── Rank coins by momentum ──
            momentum = {}
            for c in UNIVERSE:
                ph = hist[c]
                p_now = ph[-1]
                p_prev = ph[-LOOKBACK]
                momentum[c] = (p_now - p_prev) / p_prev if p_prev > 0 else 0.0

            ranked = sorted(UNIVERSE, key=lambda c: momentum[c], reverse=True)
            candidates = [c for c in ranked if momentum[c] > MIN_MOMENTUM]
            new_longs = set(candidates[:TOP_K])

            # ── Rebalance if needed ──
            need_rebal = (new_longs != current_longs) or len(stopped_out) > 0
            cooldown_ok = (global_tick - last_rebal) >= REBAL_COOLDOWN

            # Print status
            for c in ranked:
                status = "LONG" if c in current_longs else "----"
                target = "-> LONG" if c in new_longs else "-> FLAT"
                stop_info = ""
                if c in current_longs and pos[c] > 0:
                    sl = peak_price[c] - STOP_ATR_MULT * atr[c]
                    stop_info = f" stop=${sl:,.2f}"
                print(f"  [{c}] ${prices_now[c]:,.2f} mom={momentum[c]:+.3%} "
                      f"ATR=${atr[c]:,.2f} | {status} {target}{stop_info}")

            if need_rebal and cooldown_ok:
                print(f"  [REBAL] Rankings changed: {current_longs} -> {new_longs}")

                # Sell exits
                for c in list(current_longs - new_longs):
                    if pos[c] > 0:
                        mom = momentum[c]
                        if place_sell(c, pos[c], prices_now[c], "SELL-REBAL", mom, atr[c]):
                            pos[c] = 0.0
                            peak_price[c] = 0.0
                            entry_price.pop(c, None)

                # Size new positions by inverse ATR (risk parity)
                if new_longs:
                    inv_atr = {c: 1.0 / max(atr[c], 1e-10) for c in new_longs}
                    total_inv_atr = sum(inv_atr.values())
                    alloc_per = {c: ALLOC * (inv_atr[c] / total_inv_atr) for c in new_longs}
                else:
                    alloc_per = {}

                # Buy / resize
                for c in new_longs:
                    target_val = equity * alloc_per[c]
                    target_qty = target_val / prices_now[c]
                    trade_qty = target_qty - pos[c]

                    if abs(trade_qty * prices_now[c]) < MIN_ORDER_USD:
                        continue

                    if trade_qty > 0:
                        if place_buy(c, trade_qty, prices_now[c], momentum[c], atr[c]):
                            pos[c] += round(trade_qty, amt_precision.get(c, 4))
                            entry_price[c] = prices_now[c]
                            peak_price[c] = max(peak_price[c], prices_now[c])
                    elif trade_qty < 0:
                        sell_qty = abs(trade_qty)
                        if place_sell(c, sell_qty, prices_now[c], "SELL-RESIZE",
                                      momentum[c], atr[c]):
                            pos[c] -= round(sell_qty, amt_precision.get(c, 4))

                current_longs = new_longs
                last_rebal = global_tick
            elif need_rebal and not cooldown_ok:
                remaining = REBAL_COOLDOWN - (global_tick - last_rebal)
                print(f"  [COOLDOWN] Rebal needed but {remaining} ticks remaining")

            # ── Re-sync positions & equity ──
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
                          f"| Peak: ${peak_equity:,.2f} | DD: {dd:.1%}")
                    print(f"Holdings: {', '.join(current_longs) if current_longs else 'CASH'} "
                          f"| Tick: {global_tick}")
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
    run_trend_rider()
