import time
import os

from dotenv import load_dotenv
load_dotenv()

from roostoo_client import RoostooClient

API_KEY = os.getenv("API_KEY")
SECRET = os.getenv("SECRET")

client = RoostooClient(api_key=API_KEY, secret_key=SECRET)

# ─── STRATEGY: ENSEMBLE REGIME-SWITCHING BOT ─────────────────────────────────
#
#  Regime detection  (per-coin, smoothed to avoid thrashing)
#    - Bollinger Band width = (upper - lower) / mean = 4σ / mean
#    - Wide bands  (BB_width > BREAKOUT_THRESH) → TRENDING market
#    - Narrow bands (BB_width ≤ BREAKOUT_THRESH) → RANGING  market
#
#  TRENDING  → Momentum strategy
#    - EMA crossover (fast vs slow) sets direction
#    - BTC regime filter: only go long when BTC is also trending up
#    - Vol scaling: size inversely with recent realized volatility
#
#  RANGING   → Mean Reversion strategy
#    - Bollinger Bands on 20-period rolling window
#    - Position = full_size × (upper - price) / (upper - lower)
#      → 100% long at lower band, 0% (flat) at upper band, 50% at mean
#    - Also vol-scaled so we size down when volatile
#
#  Risk management
#    - Dynamic sizing based on current equity (not initial capital)
#    - Drawdown-based position scaling (reduce exposure as losses mount)
#    - Limit price offsets for reliable fills
#    - Per-cycle budget cap prevents over-allocation
#    - Cancel-before-recompute prevents stale order fills
#    - Warmup ramp avoids noisy early-EMA signals
#


def run_portfolio_bot():
    print("=" * 60)
    print("  ENSEMBLE REGIME-SWITCHING BOT")
    print("  Trending market → EMA Momentum")
    print("  Ranging  market → Mean Reversion (Bollinger Bands)")
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
    MR_BB_K  = 2.0   # Bollinger Band standard-deviation multiplier
    MR_ALLOC = 0.15  # slightly smaller allocation for mean reversion
    MR_MIN_BW = 0.005 # minimum BB width (0.5%) to allow mean reversion trades

    # Risk management
    DRAWDOWN_START = 0.03   # start reducing size at 3% drawdown from peak
    DRAWDOWN_FLOOR = 0.20   # minimum scalar (20% of normal size) at ~19%+ drawdown
    PRICE_OFFSET   = 0.0002 # 0.02% limit price offset for better fills
    WARMUP_RAMP    = 10     # ticks after warmup to ramp to full size

    k_f = 2.0 / (FAST + 1)
    k_s = 2.0 / (SLOW + 1)

    # ── Per-coin state ──
    ema_f      = {}   # fast EMA
    ema_s      = {}   # slow EMA
    hist       = {}   # price history
    ticks      = {}   # tick count
    vol_ema    = {}   # smoothed vol scalar
    regime_ema = {}   # smoothed BB-width for regime detection

    # ── Sync with exchange ──
    print("\nSyncing with exchange...")
    pos = {c: 0.0 for c in UNIVERSE}
    start_usd = 50000.0

    wallet_key = "SpotWallet" if "SpotWallet" in bal else "Wallet"
    if wallet_key in bal:
        for c in UNIVERSE:
            pos[c] = bal[wallet_key].get(c, {}).get("Free", 0.0)
        start_usd = bal[wallet_key].get("USD", {}).get("Free", 50000.0)

    equity = start_usd
    free_usd = start_usd
    peak_equity = start_usd

    print(f"  Starting USD:       ${start_usd:,.2f}")
    print(f"  Starting positions: {pos}")
    print(f"  Strategy:           Ensemble (Momentum | Mean Reversion)")
    print(f"  Regime threshold:   BB_width > {BREAKOUT_THRESH:.1%} = TRENDING")
    print(f"  Drawdown scaling:   starts at {DRAWDOWN_START:.0%}, floor at {DRAWDOWN_FLOOR:.0%}")
    print("-" * 60)

    # ── Main loop ──
    while True:
        # ── Drawdown-based position scaling ──
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        dd_scalar = 1.0
        if drawdown > DRAWDOWN_START:
            dd_scalar = max(DRAWDOWN_FLOOR, 1.0 - (drawdown - DRAWDOWN_START) * 5)

        cycle_budget = free_usd

        if drawdown > DRAWDOWN_START:
            print(f"[RISK] Drawdown {drawdown:.1%} from peak ${peak_equity:,.2f} "
                  f"| Size scalar: {dd_scalar:.0%}")

        for coin in UNIVERSE:
            try:
                pair = f"{coin}/USD"

                # Cancel any stale open orders before computing new signals.
                # Prevents fills at outdated prices when signal flips to HOLD.
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
                    regime_ema[coin] = 0.0

                # Update EMAs and history
                ema_f[coin] = price * k_f + ema_f[coin] * (1 - k_f)
                ema_s[coin] = price * k_s + ema_s[coin] * (1 - k_s)
                hist[coin].append(price)
                ticks[coin] += 1
                keep = max(VOL_WIN, REGIME_WIN, SLOW) + 5
                if len(hist[coin]) > keep * 2:
                    hist[coin] = hist[coin][-keep:]

                signal     = "HOLD"
                target_qty = 0.0

                if ticks[coin] >= SLOW:
                    ph = hist[coin]

                    # ─── REGIME DETECTION ────────────────────────────────────────
                    rp   = ph[-REGIME_WIN:]
                    rm   = sum(rp) / len(rp)
                    rstd = (sum((p - rm) ** 2 for p in rp) / len(rp)) ** 0.5
                    raw_bb_width = (4 * rstd) / rm

                    regime_ema[coin] = (REGIME_SMOOTH * raw_bb_width
                                        + (1 - REGIME_SMOOTH) * regime_ema[coin])
                    bb_width    = regime_ema[coin]
                    is_trending = bb_width > BREAKOUT_THRESH
                    regime_lbl  = "TREND" if is_trending else "RANGE"

                    # ─── VOL SCALING ────────────────────────────────────────────
                    vol_scalar = 1.0
                    if len(ph) >= VOL_WIN + 1:
                        rets = [(ph[j] - ph[j-1]) / ph[j-1]
                                for j in range(len(ph) - VOL_WIN, len(ph))]
                        avg = sum(rets) / len(rets)
                        vol = (sum((r - avg) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
                        raw_scalar = min(TARGET_VOL / max(vol, 1e-10), MAX_SCALE)
                        if coin not in vol_ema:
                            vol_ema[coin] = raw_scalar
                        else:
                            vol_ema[coin] = 0.1 * raw_scalar + 0.9 * vol_ema[coin]
                        vol_scalar = vol_ema[coin]

                    # ─── WARMUP RAMP ────────────────────────────────────────────
                    # Ramp from 25% to 100% over WARMUP_RAMP ticks after warmup
                    # to avoid full-size trades on noisy early EMA signals
                    ticks_past = ticks[coin] - SLOW
                    warmup_scalar = min(1.0, 0.25 + 0.75 * ticks_past / WARMUP_RAMP)

                    # Combined scalar: vol × drawdown × warmup
                    combined = vol_scalar * dd_scalar * warmup_scalar

                    # ─── STRATEGY SELECTION ──────────────────────────────────────
                    if is_trending:
                        # ── MOMENTUM STRATEGY ────────────────────────────────────
                        momentum = ema_f[coin] > ema_s[coin]
                        btc_ok   = ema_f.get("BTC", 1) >= ema_s.get("BTC", 0)
                        go_long  = momentum and btc_ok

                        base_qty   = equity * ALLOC / price
                        target_qty = base_qty * combined if go_long else 0.0
                        strat_tag  = "LONG" if go_long else "FLAT"

                    else:
                        # ── MEAN REVERSION STRATEGY ──────────────────────────────
                        if bb_width < MR_MIN_BW:
                            # Band too narrow — any trade costs more than it can earn
                            target_qty = pos[coin]  # hold current position, no new trades
                            strat_tag  = "MR-SKIP"
                        else:
                            bp     = ph[-SLOW:]
                            bb_mid = sum(bp) / len(bp)
                            bb_std = (sum((p - bb_mid) ** 2 for p in bp) / len(bp)) ** 0.5
                            bb_up  = bb_mid + MR_BB_K * bb_std
                            bb_dn  = bb_mid - MR_BB_K * bb_std

                            bb_pos = (bb_up - price) / max(bb_up - bb_dn, 1e-10)
                            bb_pos = max(0.0, min(1.0, bb_pos))

                            base_qty   = equity * MR_ALLOC / price
                            target_qty = base_qty * combined * bb_pos

                            if price <= bb_dn:
                                strat_tag = "MR-BUY"
                            elif price >= bb_up:
                                strat_tag = "MR-SELL"
                            else:
                                strat_tag = f"MR({bb_pos:.0%})"

                    # ─── EXECUTE ─────────────────────────────────────────────────
                    trade_qty = target_qty - pos[coin]
                    print(f"[{coin}] ${price:,.2f} | {regime_lbl} | {strat_tag} | "
                          f"vol={vol_scalar:.2f}x dd={dd_scalar:.0%} | BB_w={bb_width:.4f} | "
                          f"pos={pos[coin]:.6f} -> {target_qty:.6f}")

                    # Dead-band: skip tiny rebalances (saves fees)
                    min_rebalance = max(abs(target_qty) * 0.05, 0.0001)
                    if trade_qty > min_rebalance:
                        signal = "BUY"
                    elif trade_qty < -min_rebalance:
                        signal = "SELL"
                        trade_qty = abs(trade_qty)

                else:
                    print(f"[{coin}] ${price:,.2f} | warming up {ticks[coin]}/{SLOW}")

                # ── Place order ──
                if signal in ("BUY", "SELL"):
                    prec   = amt_precision.get(coin, 4)
                    p_prec = price_precision.get(coin, 2)

                    # Cap BUY orders against remaining cycle budget
                    if signal == "BUY":
                        max_affordable = cycle_budget * 0.95 / price
                        trade_qty = min(trade_qty, max_affordable)

                    trade_qty = round(trade_qty, prec)
                    if trade_qty > 0:
                        qty_str = f"{trade_qty:.{prec}f}"

                        # Offset limit price for better fill probability:
                        # BUY slightly above last price, SELL slightly below
                        if signal == "BUY":
                            limit_price = round(price * (1 + PRICE_OFFSET), p_prec)
                        else:
                            limit_price = round(price * (1 - PRICE_OFFSET), p_prec)
                        price_str = f"{limit_price:.{p_prec}f}"

                        try:
                            client.place_order(
                                pair=pair,
                                side=signal,
                                quantity=qty_str,
                                order_type="LIMIT",
                                price=price_str,
                            )
                            print(f"    >> {signal} {qty_str} {coin} "
                                  f"@ ${limit_price} (LIMIT) — OK")
                            if signal == "BUY":
                                cycle_budget -= trade_qty * limit_price
                        except Exception as e:
                            print(f"    >> {signal} {qty_str} {coin} — FAILED: {e}")

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
                free_usd = usd_entry.get("Free", 0.0)
                usd = free_usd + usd_entry.get("Locked", 0.0)

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
                        c: ("TREND" if regime_ema.get(c, 0) > BREAKOUT_THRESH
                            else "RANGE")
                        for c in UNIVERSE if c in regime_ema
                    }

                    print(f"\n{'=' * 60}")
                    print(f"EQUITY: ${equity:,.2f} | P&L: {pnl:+.2f}% "
                          f"| Peak: ${peak_equity:,.2f} | DD: {dd:.1%}")
                    print(f"Cash: ${usd:,.2f} | Coins: ${coin_val:,.2f}")
                    if regimes:
                        print(f"Regimes: {regimes}")
                    print(f"{'=' * 60}\n")
        except Exception as e:
            print(f"[RESYNC] Failed to sync positions: {e}")

        time.sleep(30)


if __name__ == '__main__':
    run_portfolio_bot()
