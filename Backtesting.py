# pip install yfinance pandas numpy

import yfinance as yf
import numpy as np
import pandas as pd


def backtest_breakout(period="7d", interval="1m", capital=50000.0, min_vol=0.0001,
                      fee_rate=0.001, price_data_override=None):
    """
    Volatility Breakout + Trailing Stop strategy.

    Why this works:
      - In flat markets, price NEVER breaks the channel -> 0 trades -> 0 loss
      - In trending markets, breakouts catch the move early and ride it
      - Trailing stop lets winners run, cuts losers mechanically
      - Volatility filter adds a second gate: even if price pokes above
        the channel in low vol, we don't enter (it's noise, not signal)
      - Risk-based sizing: every trade risks the same % of equity

    Signal:
      1. Donchian channel: ENTRY_LB-bar high/low
      2. BUY when price > channel high AND vol filter passes
      3. EXIT when price < trailing stop (peak - STOP_MULT * ATR)
         OR price < EXIT_LB-bar low (channel exit)
      4. Cross-sectional: only hold TOP_K positions, prefer strongest breakouts

    Args:
      fee_rate: transaction fee per trade (0.001 = 0.1%, applied to both buys and sells)
      price_data_override: dict of {coin: np.array} to skip download (used by walk-forward)
    """
    UNIVERSE = ["BTC", "ETH", "SOL", "BNB"]

    if price_data_override is not None:
        price_data = price_data_override
        UNIVERSE = list(price_data.keys())
        min_len = min(len(v) for v in price_data.values())
        print(f"  Using provided data: {min_len} bars across {len(UNIVERSE)} coins\n")
    else:
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
            return None

        min_len = min(len(v) for v in price_data.values())
        for coin in list(price_data.keys()):
            price_data[coin] = price_data[coin][-min_len:]
        UNIVERSE = list(price_data.keys())
        print(f"  Aligned to {min_len} bars across {len(UNIVERSE)} coins\n")

    # ── Parameters ──
    ENTRY_LB       = 120    # Donchian entry channel lookback — optimized via sweep
    EXIT_LB        = 48     # Donchian exit channel lookback (tighter)
    ATR_WIN        = 20     # ATR window
    TOP_K          = 2      # max simultaneous positions
    RISK_PER_TRADE = 0.01   # risk 1% of equity per trade
    MAX_ALLOC      = 0.25   # never put more than 25% in one coin
    STOP_MULT      = 2.0    # trailing stop = peak - STOP_MULT * ATR
    MIN_VOL        = min_vol # minimum ATR/price to allow entry
    PRICE_OFFSET   = 0.0002 # slippage
    COOLDOWN       = 20     # bars after a stop-out before re-entry for same coin
    REGIME_VOL_THRESH = 0.0002  # avg vol must exceed this to allow entries
    REGIME_WINDOW  = 10     # ticks to smooth vol for regime detection

    # ── State ──
    pos = {c: 0.0 for c in UNIVERSE}
    peak_price = {c: 0.0 for c in UNIVERSE}
    entry_price = {}
    cash = capital
    current_longs = set()
    last_exit = {c: -COOLDOWN for c in UNIVERSE}  # cooldown per coin
    vol_history = []

    equity_curve = []
    trade_log = []
    signals_log = []

    warmup = max(ENTRY_LB, EXIT_LB, ATR_WIN) + 1

    for i in range(min_len):
        prices_now = {c: price_data[c][i] for c in UNIVERSE}

        # Update trailing peaks
        for c in current_longs:
            if prices_now[c] > peak_price[c]:
                peak_price[c] = prices_now[c]

        # Mark equity
        coin_val = sum(pos[c] * prices_now[c] for c in UNIVERSE)
        equity = cash + coin_val
        equity_curve.append(equity)

        if i < warmup:
            continue

        # ── Compute indicators ──
        atr = {}
        chan_high = {}
        chan_low = {}
        vol_pct = {}

        for c in UNIVERSE:
            ph = price_data[c]

            # ATR (average absolute move)
            diffs = [abs(ph[j] - ph[j-1]) for j in range(i - ATR_WIN + 1, i + 1)]
            atr[c] = sum(diffs) / len(diffs) if diffs else 1e-10

            # Donchian channels
            chan_high[c] = max(ph[i - ENTRY_LB:i])  # exclude current bar
            chan_low[c] = min(ph[i - EXIT_LB:i])

            # Volatility as % of price
            vol_pct[c] = atr[c] / prices_now[c] if prices_now[c] > 0 else 0

        # ── Check exits first ──
        for c in list(current_longs):
            if pos[c] <= 0:
                current_longs.discard(c)
                continue

            stop_level = peak_price[c] - STOP_MULT * atr[c]
            exit_channel = chan_low[c]
            exit_level = max(stop_level, exit_channel)

            if prices_now[c] <= exit_level:
                reason = "STOP" if prices_now[c] <= stop_level else "CHAN-EXIT"
                fill = prices_now[c] * (1 - PRICE_OFFSET)
                fee = pos[c] * fill * fee_rate
                pnl_trade = (fill - entry_price.get(c, fill)) / entry_price.get(c, fill)
                cash += pos[c] * fill - fee
                trade_log.append({
                    "bar": i, "coin": c, "side": f"SELL-{reason}",
                    "qty": pos[c], "price": prices_now[c], "fill": fill,
                    "pnl": pnl_trade, "equity": equity,
                })
                pos[c] = 0.0
                peak_price[c] = 0.0
                entry_price.pop(c, None)
                current_longs.discard(c)
                last_exit[c] = i

        # ── Regime detection ──
        avg_vol = sum(vol_pct[c] for c in UNIVERSE) / len(UNIVERSE)
        vol_history.append(avg_vol)
        if len(vol_history) > REGIME_WINDOW:
            vol_history.pop(0)
        smoothed_vol = sum(vol_history) / len(vol_history)
        regime = "ACTIVE" if smoothed_vol >= REGIME_VOL_THRESH else "FLAT"

        # ── Check entries (only in ACTIVE regime) ──
        breakout_candidates = []
        if regime == "ACTIVE":
            for c in UNIVERSE:
                if c in current_longs:
                    continue
                if (i - last_exit.get(c, -COOLDOWN)) < COOLDOWN:
                    continue
                if vol_pct[c] < MIN_VOL:
                    continue
                if prices_now[c] > chan_high[c]:
                    # Breakout strength = how far above channel
                    strength = (prices_now[c] - chan_high[c]) / atr[c] if atr[c] > 0 else 0
                    breakout_candidates.append((c, strength))

        # Rank by breakout strength, take top slots available
        breakout_candidates.sort(key=lambda x: x[1], reverse=True)
        slots = TOP_K - len(current_longs)

        for c, strength in breakout_candidates[:slots]:
            # Risk-based sizing: risk RISK_PER_TRADE of equity
            stop_distance = STOP_MULT * atr[c]
            if stop_distance <= 0:
                continue
            qty_by_risk = (equity * RISK_PER_TRADE) / stop_distance
            # Cap by max allocation
            qty_by_alloc = (equity * MAX_ALLOC) / prices_now[c]
            qty = min(qty_by_risk, qty_by_alloc)
            # Cap by cash
            fill = prices_now[c] * (1 + PRICE_OFFSET)
            max_qty = (cash * 0.95) / fill
            qty = min(qty, max_qty)

            if qty * prices_now[c] < 50:  # min notional
                continue

            fee = qty * fill * fee_rate
            cash -= qty * fill + fee
            pos[c] = qty
            entry_price[c] = prices_now[c]
            peak_price[c] = prices_now[c]
            current_longs.add(c)

            trade_log.append({
                "bar": i, "coin": c, "side": "BUY-BREAKOUT",
                "qty": qty, "price": prices_now[c], "fill": fill,
                "pnl": 0, "equity": equity,
            })

    # ── Final equity ──
    final_prices = {c: price_data[c][-1] for c in UNIVERSE}
    final_equity = cash + sum(pos[c] * final_prices[c] for c in UNIVERSE)

    # ── Metrics ──
    eq = pd.Series(equity_curve)
    returns = eq.pct_change().dropna()

    ann_map = {"1m": 525_600, "5m": 105_120, "1h": 8_760, "1d": 365}
    ann_factor = ann_map.get(interval, 525_600)

    if returns.std() == 0 or len(returns) < 2:
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(ann_factor)

    total_ret = (final_equity / capital - 1) * 100
    peak_eq = eq.expanding().max()
    max_dd = ((eq - peak_eq) / peak_eq).min() * 100

    # Trade stats
    sells = [t for t in trade_log if t["side"].startswith("SELL")]
    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] <= 0]
    win_rate = (len(wins) / len(sells) * 100) if sells else 0.0
    avg_win = np.mean([t["pnl"] for t in wins]) * 100 if wins else 0.0
    avg_loss = np.mean([t["pnl"] for t in losses]) * 100 if losses else 0.0
    profit_factor = (abs(sum(t["pnl"] for t in wins)) /
                     abs(sum(t["pnl"] for t in losses))) if losses and sum(t["pnl"] for t in losses) != 0 else float('inf')

    # Type breakdown
    type_counts = {}
    for t in trade_log:
        s = t["side"]
        type_counts[s] = type_counts.get(s, 0) + 1

    # ── Report ──
    print("=" * 62)
    print("  VOLATILITY BREAKOUT -- BACKTEST RESULTS")
    print(f"  Period: {period} | Interval: {interval} | Bars: {min_len:,}")
    print(f"  Params: ENTRY={ENTRY_LB} EXIT={EXIT_LB} STOP={STOP_MULT}xATR "
          f"MIN_VOL={MIN_VOL:.2%} FEE={fee_rate:.2%}")
    print("=" * 62)

    print(f"\n  PORTFOLIO METRICS")
    print(f"  {'-'*50}")
    print(f"  Starting Capital:   ${capital:>12,.2f}")
    print(f"  Final Equity:       ${final_equity:>12,.2f}")
    print(f"  Total Return:       {total_ret:>+11.2f}%")
    print(f"  Sharpe Ratio:       {sharpe:>12.2f}  (annualized)")
    print(f"  Max Drawdown:       {max_dd:>+11.2f}%")
    print(f"  Total Trades:       {len(trade_log):>12,}")

    print(f"\n  TRADE QUALITY")
    print(f"  {'-'*50}")
    print(f"  Win Rate:           {win_rate:>11.1f}%")
    print(f"  Avg Win:            {avg_win:>+11.2f}%")
    print(f"  Avg Loss:           {avg_loss:>+11.2f}%")
    print(f"  Profit Factor:      {profit_factor:>12.2f}")

    if type_counts:
        print(f"\n  TRADE BREAKDOWN")
        print(f"  {'-'*50}")
        for s, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"  {s:<20} {cnt:>6}")

    print(f"\n  PER-COIN FINAL POSITION")
    print(f"  {'-'*50}")
    for c in UNIVERSE:
        val = pos[c] * final_prices[c]
        held = "LONG" if c in current_longs else "FLAT"
        print(f"  {c:<6} {pos[c]:>12.6f}  ${val:>10,.2f}  {held}")
    print(f"  {'Cash':<6} {'':>12}  ${cash:>10,.2f}")

    print(f"\n{'=' * 62}")

    return {
        "sharpe": sharpe,
        "total_return_pct": total_ret,
        "max_drawdown_pct": max_dd,
        "total_trades": len(trade_log),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "equity_curve": equity_curve,
    }


def walk_forward_validation(period="60d", interval="1h", n_folds=3,
                            train_pct=0.7, capital=50000.0, fee_rate=0.001):
    """
    Walk-forward out-of-sample validation.

    Splits data into n_folds chronological windows. For each fold:
      - Train (in-sample): first train_pct of the fold
      - Test (out-of-sample): remaining (1 - train_pct) of the fold

    The test Sharpe is the honest estimate — it uses data the strategy
    has never seen during parameter selection.
    """
    UNIVERSE = ["BTC", "ETH", "SOL", "BNB"]

    print("=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print(f"  Period: {period} | Interval: {interval} | Folds: {n_folds}")
    print(f"  Train/Test split: {train_pct:.0%} / {1-train_pct:.0%}")
    print(f"  Fee rate: {fee_rate:.2%}")
    print("=" * 70)

    # Download all data once
    print("\nDownloading price data...")
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

    min_len = min(len(v) for v in price_data.values())
    for coin in list(price_data.keys()):
        price_data[coin] = price_data[coin][-min_len:]
    UNIVERSE = list(price_data.keys())
    print(f"  Aligned to {min_len} bars across {len(UNIVERSE)} coins\n")

    # Split into folds
    fold_size = min_len // n_folds
    train_results = []
    test_results = []

    for fold in range(n_folds):
        fold_start = fold * fold_size
        fold_end = fold_start + fold_size if fold < n_folds - 1 else min_len
        split_point = fold_start + int((fold_end - fold_start) * train_pct)

        print(f"\n{'─' * 70}")
        print(f"  FOLD {fold+1}/{n_folds}: bars [{fold_start}..{fold_end}]")
        print(f"  Train: [{fold_start}..{split_point}] ({split_point - fold_start} bars)")
        print(f"  Test:  [{split_point}..{fold_end}] ({fold_end - split_point} bars)")
        print(f"{'─' * 70}")

        # Train slice
        train_data = {c: price_data[c][fold_start:split_point] for c in UNIVERSE}
        print("\n  >>> IN-SAMPLE (train):")
        train_r = backtest_breakout(
            period=period, interval=interval, capital=capital,
            min_vol=0.00005, fee_rate=fee_rate, price_data_override=train_data)
        if train_r:
            train_results.append(train_r)

        # Test slice
        test_data = {c: price_data[c][split_point:fold_end] for c in UNIVERSE}
        print("\n  >>> OUT-OF-SAMPLE (test):")
        test_r = backtest_breakout(
            period=period, interval=interval, capital=capital,
            min_vol=0.00005, fee_rate=fee_rate, price_data_override=test_data)
        if test_r:
            test_results.append(test_r)

    # Summary
    sep = "=" * 70
    print(f"\n\n{sep}")
    print("  WALK-FORWARD SUMMARY")
    print(sep)

    if train_results:
        avg_train_sharpe = np.mean([r["sharpe"] for r in train_results])
        avg_train_ret = np.mean([r["total_return_pct"] for r in train_results])
        avg_train_dd = np.mean([r["max_drawdown_pct"] for r in train_results])
        print(f"\n  IN-SAMPLE (train) averages:")
        print(f"    Avg Sharpe:     {avg_train_sharpe:+.2f}")
        print(f"    Avg Return:     {avg_train_ret:+.2f}%")
        print(f"    Avg Max DD:     {avg_train_dd:+.2f}%")

    if test_results:
        avg_test_sharpe = np.mean([r["sharpe"] for r in test_results])
        avg_test_ret = np.mean([r["total_return_pct"] for r in test_results])
        avg_test_dd = np.mean([r["max_drawdown_pct"] for r in test_results])
        print(f"\n  OUT-OF-SAMPLE (test) averages — THIS IS YOUR HONEST ESTIMATE:")
        print(f"    Avg Sharpe:     {avg_test_sharpe:+.2f}")
        print(f"    Avg Return:     {avg_test_ret:+.2f}%")
        print(f"    Avg Max DD:     {avg_test_dd:+.2f}%")
        print(f"    Folds profitable: {sum(1 for r in test_results if r['total_return_pct'] > 0)}/{len(test_results)}")

    if train_results and test_results:
        decay = avg_test_sharpe / avg_train_sharpe if avg_train_sharpe != 0 else 0
        print(f"\n  Sharpe decay (test/train): {decay:.1%}")
        if decay > 0.5:
            print("  -> Strategy generalizes well (>50% retention)")
        elif decay > 0.2:
            print("  -> Moderate overfitting — some signal survives")
        else:
            print("  -> Heavy overfitting — in-sample results are misleading")

    print(f"\n{sep}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    # ── Fee rate for realistic backtests ──
    FEE_RATE = 0.001  # 0.1% per trade (typical exchange fee)

    if mode in ("all", "sweep"):
        # ── MIN_VOL Parameter Sweep (with fees) ──
        MIN_VOL_VALUES = [0.0, 0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.001]
        PERIODS = [
            ("7d/1m",  "7d",  "1m"),
            ("1mo/5m", "1mo", "5m"),
            ("60d/1h", "60d", "1h"),
        ]

        results = {}

        for mv in MIN_VOL_VALUES:
            results[mv] = {}
            print(f"\n{'#' * 62}")
            print(f"  TESTING MIN_VOL = {mv}")
            print(f"{'#' * 62}\n")
            for label, period, interval in PERIODS:
                print(f">>> {label} with MIN_VOL={mv}")
                r = backtest_breakout(period=period, interval=interval,
                                      min_vol=mv, fee_rate=FEE_RATE)
                results[mv][label] = r

        # ── Comparison Table ──
        print("\n\n" + "=" * 90)
        print("  MIN_VOL PARAMETER SWEEP -- COMPARISON (with fees)")
        print("=" * 90)

        for label, _, _ in PERIODS:
            print(f"\n  --- {label} ---")
            print(f"  {'MIN_VOL':>10} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Trades':>8} {'WinRate':>8} {'PF':>8}")
            print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
            for mv in MIN_VOL_VALUES:
                r = results[mv].get(label)
                if r:
                    print(f"  {mv:>10.5f} {r['total_return_pct']:>+9.2f}% {r['sharpe']:>10.2f} "
                          f"{r['max_drawdown_pct']:>+9.2f}% {r['total_trades']:>8} "
                          f"{r['win_rate']:>7.1f}% {r['profit_factor']:>8.2f}")
                else:
                    print(f"  {mv:>10.5f}   NO DATA")

        # ── Best MIN_VOL by average Sharpe ──
        print(f"\n  --- AVERAGE SHARPE ACROSS ALL PERIODS ---")
        print(f"  {'MIN_VOL':>10} {'Avg Sharpe':>12} {'Avg Return':>12} {'Avg MaxDD':>12}")
        print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
        best_mv, best_avg = None, -999
        for mv in MIN_VOL_VALUES:
            sharpes = [results[mv][l]["sharpe"] for l, _, _ in PERIODS if results[mv].get(l)]
            rets = [results[mv][l]["total_return_pct"] for l, _, _ in PERIODS if results[mv].get(l)]
            dds = [results[mv][l]["max_drawdown_pct"] for l, _, _ in PERIODS if results[mv].get(l)]
            if sharpes:
                avg_s = sum(sharpes) / len(sharpes)
                avg_r = sum(rets) / len(rets)
                avg_d = sum(dds) / len(dds)
                if avg_s > best_avg:
                    best_avg = avg_s
                    best_mv = mv
                print(f"  {mv:>10.5f} {avg_s:>+11.2f} {avg_r:>+11.2f}% {avg_d:>+11.2f}%")

        print(f"\n  >>> BEST MIN_VOL = {best_mv} (avg Sharpe = {best_avg:+.2f})")
        print("=" * 90)

    if mode in ("all", "walkforward"):
        # ── Walk-Forward Validation ──
        print("\n\n")
        walk_forward_validation(
            period="60d", interval="1h", n_folds=3,
            train_pct=0.7, fee_rate=FEE_RATE)
