# pip install yfinance pandas numpy

import yfinance as yf
import numpy as np
import pandas as pd


def compute_rsi(prices, period=14):
    """SMA-based RSI. Returns 50.0 if not enough data or flat market."""
    if len(prices) < period + 1:
        return 50.0
    deltas = [prices[i] - prices[i - 1] for i in range(len(prices) - period, len(prices))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    if avg_gain == 0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def backtest_ensemble(period="7d", interval="1m", capital=50000.0):
    """
    Portfolio-level backtest of the full multi-factor ensemble strategy.
    Mirrors main.py logic: regime detection, momentum + MR, RSI gates,
    vol scaling, drawdown scaling, cooldown, position caps, slippage.
    """
    UNIVERSE = ["BTC", "ETH", "SOL", "BNB"]

    # -- Download data --
    print("Downloading price data...")
    price_data = {}
    for coin in UNIVERSE:
        df = yf.download(f"{coin}-USD", period=period, interval=interval, progress=False)
        if df.empty:
            print(f"  !! No data for {coin}, skipping")
            continue
        price_data[coin] = df["Close"].values.flatten().astype(float)
        print(f"  {coin}: {len(price_data[coin])} bars")

    if not price_data:
        print("No data available.")
        return

    # Align to shortest series
    min_len = min(len(v) for v in price_data.values())
    for coin in list(price_data.keys()):
        price_data[coin] = price_data[coin][-min_len:]
    UNIVERSE = list(price_data.keys())
    print(f"  Aligned to {min_len} bars across {len(UNIVERSE)} coins\n")

    # -- Parameters (must match main.py) --
    FAST       = 5
    SLOW       = 20
    VOL_WIN    = 20
    TARGET_VOL = 0.0002
    MAX_SCALE  = 1.5
    ALLOC      = 0.20

    REGIME_WIN      = 20
    BREAKOUT_THRESH = 0.025
    REGIME_SMOOTH   = 0.2

    MR_BB_K         = 2.0
    MR_ALLOC        = 0.15
    MR_MIN_BW       = 0.002
    MR_SKIP_EXIT_PCT = 0.02

    RSI_WIN        = 14
    RSI_OVERBOUGHT = 75
    RSI_OVERSOLD_MR = 55

    DRAWDOWN_START  = 0.03
    DRAWDOWN_FLOOR  = 0.20
    PRICE_OFFSET    = 0.0002
    WARMUP_RAMP     = 10
    MAX_COIN_ALLOC  = 0.30
    MIN_ORDER_USD   = 150.00
    TRADE_COOLDOWN  = 40

    k_f = 2.0 / (FAST + 1)
    k_s = 2.0 / (SLOW + 1)
    HIST_KEEP = max(VOL_WIN, REGIME_WIN, SLOW, RSI_WIN) + 10

    # -- Per-coin state --
    ema_f      = {}
    ema_s      = {}
    hist       = {}
    ticks      = {}
    vol_ema    = {}
    regime_ema = {}
    last_trade = {}
    entry_price_track = {}

    pos = {c: 0.0 for c in UNIVERSE}
    cash = capital
    equity = capital
    peak_equity = capital

    equity_curve = []
    trade_log = []
    regime_log = {c: [] for c in UNIVERSE}

    # -- Run bar-by-bar --
    for i in range(min_len):
        # Compute equity
        coin_val = sum(pos[c] * price_data[c][i] for c in UNIVERSE)
        equity = cash + coin_val
        peak_equity = max(peak_equity, equity)
        equity_curve.append(equity)

        # Drawdown scaling
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        dd_scalar = 1.0
        if drawdown > DRAWDOWN_START:
            dd_scalar = max(DRAWDOWN_FLOOR, 1.0 - (drawdown - DRAWDOWN_START) * 5)

        cycle_budget = cash

        for coin in UNIVERSE:
            price = price_data[coin][i]
            if price <= 0:
                continue

            # Init
            if coin not in ema_f:
                ema_f[coin]      = price
                ema_s[coin]      = price
                hist[coin]       = []
                ticks[coin]      = 0
                regime_ema[coin] = None
                last_trade[coin] = -TRADE_COOLDOWN

            # Update EMAs and history
            ema_f[coin] = price * k_f + ema_f[coin] * (1 - k_f)
            ema_s[coin] = price * k_s + ema_s[coin] * (1 - k_s)
            hist[coin].append(price)
            ticks[coin] += 1

            if len(hist[coin]) > HIST_KEEP:
                hist[coin] = hist[coin][-HIST_KEEP:]

            if ticks[coin] < SLOW:
                regime_log[coin].append("WARMUP")
                continue

            ph = hist[coin]

            # -- REGIME DETECTION --
            rp   = ph[-REGIME_WIN:]
            rm   = sum(rp) / len(rp)
            rstd = (sum((p - rm) ** 2 for p in rp) / len(rp)) ** 0.5
            raw_bb_width = (4 * rstd) / rm if rm > 0 else 0.0

            if regime_ema[coin] is None:
                regime_ema[coin] = raw_bb_width
            else:
                regime_ema[coin] = (REGIME_SMOOTH * raw_bb_width
                                    + (1 - REGIME_SMOOTH) * regime_ema[coin])
            bb_width    = regime_ema[coin]
            is_trending = bb_width > BREAKOUT_THRESH
            regime_lbl  = "TREND" if is_trending else "RANGE"
            regime_log[coin].append(regime_lbl)

            # -- VOL SCALING --
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

            # -- WARMUP RAMP --
            ticks_past = ticks[coin] - SLOW
            warmup_scalar = min(1.0, 0.25 + 0.75 * ticks_past / WARMUP_RAMP)

            # -- RSI --
            rsi = compute_rsi(ph, RSI_WIN)

            # -- STRATEGY SELECTION --
            target_qty = 0.0
            strat_tag = ""

            if is_trending:
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
                    target_qty = pos[coin] * 0.5
                    strat_tag  = "FLAT-FADE"
            else:
                mr_vol = min(vol_scalar, 1.0)
                combined = mr_vol * dd_scalar * warmup_scalar

                if bb_width < MR_MIN_BW:
                    ep = entry_price_track.get(coin)
                    if pos[coin] > 0 and ep and ep > 0:
                        pnl_since_entry = (price - ep) / ep
                        if pnl_since_entry < -MR_SKIP_EXIT_PCT:
                            target_qty = 0.0
                            strat_tag  = "MR-STOP"
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

                    if bb_pos > 0.7 and rsi >= RSI_OVERSOLD_MR:
                        target_qty = pos[coin]
                        strat_tag  = "MR-RSI_BLOCK"
                    else:
                        target_qty = base_qty * combined * bb_pos
                        strat_tag  = "MR"

            # Cap max position per coin
            max_qty = (equity * MAX_COIN_ALLOC) / price
            target_qty = min(target_qty, max_qty)

            # -- EXECUTE --
            trade_qty = target_qty - pos[coin]

            # Dead-band
            min_rebalance = max(abs(target_qty) * 0.15, 0.0001)
            signal_dir = "HOLD"
            if trade_qty > min_rebalance:
                signal_dir = "BUY"
            elif trade_qty < -min_rebalance:
                signal_dir = "SELL"
                trade_qty = abs(trade_qty)

            if signal_dir in ("BUY", "SELL"):
                # Min notional
                if abs(trade_qty) * price < MIN_ORDER_USD:
                    continue

                # Cooldown
                ticks_since = ticks[coin] - last_trade.get(coin, -TRADE_COOLDOWN)
                if ticks_since < TRADE_COOLDOWN:
                    continue

                # Budget cap for buys
                if signal_dir == "BUY":
                    max_affordable = cycle_budget * 0.95 / price
                    trade_qty = min(trade_qty, max_affordable)

                if trade_qty <= 0:
                    continue

                # Apply slippage (PRICE_OFFSET)
                if signal_dir == "BUY":
                    fill_price = price * (1 + PRICE_OFFSET)
                    cost = trade_qty * fill_price
                    cash -= cost
                    pos[coin] += trade_qty
                    cycle_budget -= cost
                    entry_price_track[coin] = price
                else:
                    fill_price = price * (1 - PRICE_OFFSET)
                    proceeds = trade_qty * fill_price
                    cash += proceeds
                    pos[coin] -= trade_qty
                    if trade_qty >= pos[coin] * 0.9:
                        entry_price_track.pop(coin, None)

                last_trade[coin] = ticks[coin]

                trade_log.append({
                    "bar": i,
                    "coin": coin,
                    "side": signal_dir,
                    "qty": trade_qty,
                    "price": price,
                    "fill": fill_price,
                    "regime": regime_lbl,
                    "strategy": strat_tag,
                    "rsi": rsi,
                    "equity": equity,
                })

    # -- Compute metrics --
    eq = pd.Series(equity_curve)
    returns = eq.pct_change().dropna()

    # Sharpe ratio (annualized for 1-min bars: 525,600 mins/year)
    if returns.std() == 0 or len(returns) < 2:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(525_600)

    total_ret = (equity_curve[-1] / capital - 1) * 100
    peak = eq.expanding().max()
    max_dd = ((eq - peak) / peak).min() * 100

    # Win rate
    if trade_log:
        buys = {}
        wins = 0
        total_round_trips = 0
        for t in trade_log:
            c = t["coin"]
            if t["side"] == "BUY":
                buys[c] = t["fill"]
            elif t["side"] == "SELL" and c in buys:
                total_round_trips += 1
                if t["fill"] > buys[c]:
                    wins += 1
                del buys[c]
        win_rate = (wins / total_round_trips * 100) if total_round_trips > 0 else 0.0
    else:
        win_rate = 0.0
        total_round_trips = 0

    # Regime breakdown
    regime_counts = {}
    for coin in UNIVERSE:
        for r in regime_log.get(coin, []):
            regime_counts[r] = regime_counts.get(r, 0) + 1

    # Strategy breakdown
    strat_counts = {}
    for t in trade_log:
        s = t["strategy"]
        strat_counts[s] = strat_counts.get(s, 0) + 1

    # Per-coin P&L
    final_coin_val = {c: pos[c] * price_data[c][-1] for c in UNIVERSE}

    # -- Print report --
    print("=" * 62)
    print("  MULTI-FACTOR ENSEMBLE — BACKTEST RESULTS")
    print(f"  Period: {period} | Interval: {interval} | Bars: {min_len:,}")
    print("=" * 62)

    print(f"\n  {'PORTFOLIO METRICS':-<50}")
    print(f"  Starting Capital:   ${capital:>12,.2f}")
    print(f"  Final Equity:       ${equity_curve[-1]:>12,.2f}")
    print(f"  Total Return:       {total_ret:>+11.2f}%")
    print(f"  Sharpe Ratio:       {sharpe:>12.2f}  (annualized)")
    print(f"  Max Drawdown:       {max_dd:>+11.2f}%")
    print(f"  Total Trades:       {len(trade_log):>12,}")
    print(f"  Round Trips:        {total_round_trips:>12,}")
    print(f"  Win Rate:           {win_rate:>11.1f}%")

    print(f"\n  {'PER-COIN BREAKDOWN':-<50}")
    print(f"  {'Coin':<6} {'Position':>12} {'Value':>12} {'Trades':>8}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8}")
    for c in UNIVERSE:
        coin_trades = sum(1 for t in trade_log if t["coin"] == c)
        print(f"  {c:<6} {pos[c]:>12.6f} ${final_coin_val[c]:>10,.2f} {coin_trades:>8}")
    print(f"  {'Cash':<6} {'':>12} ${cash:>10,.2f}")

    print(f"\n  {'REGIME DISTRIBUTION':-<50}")
    total_bars = sum(regime_counts.values()) or 1
    for r, cnt in sorted(regime_counts.items()):
        print(f"  {r:<12} {cnt:>8,} bars  ({cnt/total_bars*100:.1f}%)")

    if strat_counts:
        print(f"\n  {'STRATEGY DISTRIBUTION (trades)':-<50}")
        for s, cnt in sorted(strat_counts.items(), key=lambda x: -x[1]):
            print(f"  {s:<20} {cnt:>6} trades")

    print(f"\n{'=' * 62}")

    return {
        "sharpe": sharpe,
        "total_return_pct": total_ret,
        "max_drawdown_pct": max_dd,
        "total_trades": len(trade_log),
        "win_rate": win_rate,
        "equity_curve": equity_curve,
        "trade_log": trade_log,
    }


if __name__ == "__main__":
    print("\n>>> BACKTEST 1: Last 7 days (recent flat market)")
    print(">>> This is the environment your bot has been running in.\n")
    backtest_ensemble(period="7d", interval="1m")

    print("\n\n>>> BACKTEST 2: Last 1 month (captures volatile periods)")
    print(">>> Tests whether the strategy has edge when vol returns.\n")
    backtest_ensemble(period="1mo", interval="5m")

    print("\n\n>>> BACKTEST 3: Last 60 days (full market cycle)")
    print(">>> Longest intraday lookback Yahoo allows.\n")
    backtest_ensemble(period="60d", interval="1h")
