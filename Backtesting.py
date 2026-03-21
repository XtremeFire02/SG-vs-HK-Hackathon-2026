# pip install yfinance pandas numpy

import yfinance as yf
import numpy as np
import pandas as pd


def backtest(coin, fast=20, slow=60, vol_window=20, target_vol=0.0002,
             max_scale=1.5, alloc=0.20, capital=50000):
    """
    Backtest: EMA crossover + volatility scaling.
    Returns (sharpe, total_return_pct, max_drawdown_pct).
    """
    df = yf.download(f"{coin}-USD", period="7d", interval="1m", progress=False)
    if df.empty:
        return 0.0, 0.0, 0.0

    prices = df['Close'].values.flatten().astype(float)
    cash = capital
    position = 0.0
    equity_curve = []

    k_f = 2.0 / (fast + 1)
    k_s = 2.0 / (slow + 1)
    ema_f = prices[0]
    ema_s = prices[0]

    for i, price in enumerate(prices):
        ema_f = price * k_f + ema_f * (1 - k_f)
        ema_s = price * k_s + ema_s * (1 - k_s)

        if i >= slow:
            total_eq = cash + position * price
            momentum = ema_f > ema_s

            # Vol scaling
            vol_scalar = 1.0
            if i >= vol_window + 1:
                rets = [(prices[j] - prices[j-1]) / prices[j-1]
                        for j in range(i - vol_window, i)]
                avg = sum(rets) / len(rets)
                vol = (sum((r - avg)**2 for r in rets) / (len(rets) - 1)) ** 0.5
                vol_scalar = min(target_vol / max(vol, 1e-10), max_scale)

            base_qty = total_eq * alloc / price
            target_qty = base_qty * vol_scalar if momentum else 0.0

            trade_qty = target_qty - position
            if abs(trade_qty) * price > 1.0:
                cash -= trade_qty * price
                position += trade_qty

        equity_curve.append(cash + position * price)

    eq = pd.Series(equity_curve)
    returns = eq.pct_change().dropna()

    if returns.std() == 0:
        return 0.0, 0.0, 0.0

    sharpe = (returns.mean() / returns.std()) * np.sqrt(525_600)
    total_ret = (equity_curve[-1] / capital - 1) * 100

    peak = eq.expanding().max()
    max_dd = ((eq - peak) / peak).min() * 100

    return sharpe, total_ret, max_dd


if __name__ == '__main__':
    universe = ["BTC", "ETH", "SOL", "BNB"]

    print("=" * 62)
    print("  VOLATILITY-TARGETED MOMENTUM — 7-DAY BACKTEST")
    print("  EMA(20,60) + Vol Scaling (target ~15% annual, cap 1.5x)")
    print("=" * 62)
    print()
    print(f"  {'Coin':<6} {'Sharpe':>8} {'Return':>10} {'Max DD':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10}")

    for coin in universe:
        s, r, d = backtest(coin)
        print(f"  {coin:<6} {s:>8.2f} {r:>+9.2f}% {d:>+9.2f}%")

    print()
    print("  Note: Vol scaling reduces position size during high-volatility")
    print("  periods and increases it during calm trending markets.")
