import time
import os

from dotenv import load_dotenv
load_dotenv()

from roostoo_client import RoostooClient

API_KEY = os.getenv("API_KEY")
SECRET = os.getenv("SECRET")

client = RoostooClient(api_key=API_KEY, secret_key=SECRET)

# ─── STRATEGY: VOLATILITY-TARGETED CROSS-ASSET MOMENTUM ──────
#
#  Three factors, each one line of logic:
#    1. EMA crossover  → identifies trend direction per coin
#    2. BTC regime     → only go long when the market leader is bullish
#    3. Vol scaling    → size up in calm markets, size down in volatile ones
#

def run_portfolio_bot():
    print("=" * 60)
    print("  VOLATILITY-TARGETED CROSS-ASSET MOMENTUM BOT")
    print("=" * 60)

    # ── Auth check ──
    print("\nChecking API credentials...")
    try:
        bal = client.get_balance()
        print("  API credentials OK")
    except Exception as e:
        print(f"  !! AUTH FAILED: {e}")
        print("  !! Check your API_KEY and SECRET in .env")
        return

    # ── Load exchange precision rules ──
    ex_info = client.get_exchange_info()
    amt_precision = {}
    if ex_info and "TradePairs" in ex_info:
        for pname, pinfo in ex_info["TradePairs"].items():
            amt_precision[pinfo["Coin"]] = pinfo["AmountPrecision"]

    # ── Parameters ──
    UNIVERSE   = ["BTC", "ETH", "SOL", "BNB"]
    FAST       = 5        # fast EMA period (in ticks)
    SLOW       = 20       # slow EMA period (in ticks)
    VOL_WIN    = 20       # realized vol lookback
    TARGET_VOL = 0.0002   # per-tick vol target (~15% annualized)
    MAX_SCALE  = 1.5      # cap on vol scalar to prevent overleverage
    ALLOC      = 0.20     # base allocation per coin (fraction of starting capital)

    k_f = 2.0 / (FAST + 1)
    k_s = 2.0 / (SLOW + 1)

    # ── Per-coin state ──
    ema_f  = {}   # fast EMA
    ema_s  = {}   # slow EMA
    hist   = {}   # price history
    ticks  = {}   # tick count

    # ── Sync with exchange ──
    print("\nSyncing with exchange...")
    pos = {c: 0.0 for c in UNIVERSE}
    start_usd = 50000.0

    wallet_key = "SpotWallet" if "SpotWallet" in bal else "Wallet"
    if wallet_key in bal:
        for c in UNIVERSE:
            pos[c] = bal[wallet_key].get(c, {}).get("Free", 0.0)
        start_usd = bal[wallet_key].get("USD", {}).get("Free", 50000.0)

    print(f"  Starting USD:       ${start_usd:,.2f}")
    print(f"  Starting positions: {pos}")
    print(f"  Strategy:           EMA({FAST},{SLOW}) + VolTarget + BTC Filter")
    print("-" * 60)

    # ── Main loop ──
    while True:
        for coin in UNIVERSE:
            try:
                pair = f"{coin}/USD"
                td = client.get_ticker(pair)
                if not (td and "Data" in td and pair in td["Data"]):
                    continue

                price = float(td["Data"][pair]["LastPrice"])

                # First tick initialization
                if coin not in ema_f:
                    ema_f[coin] = price
                    ema_s[coin] = price
                    hist[coin]  = []
                    ticks[coin] = 0

                # Update EMAs
                ema_f[coin] = price * k_f + ema_f[coin] * (1 - k_f)
                ema_s[coin] = price * k_s + ema_s[coin] * (1 - k_s)
                hist[coin].append(price)
                ticks[coin] += 1
                if len(hist[coin]) > SLOW * 5:
                    hist[coin] = hist[coin][-(VOL_WIN + 5):]

                signal = "HOLD"

                if ticks[coin] >= SLOW:
                    # Factor 1: Momentum — is this coin trending up?
                    momentum = ema_f[coin] > ema_s[coin]

                    # Factor 2: Regime — is BTC (market leader) trending up?
                    btc_ok = ema_f.get("BTC", 1) >= ema_s.get("BTC", 0)

                    # Factor 3: Vol scaling — size inversely with recent volatility
                    vol_scalar = 1.0
                    ph = hist[coin]
                    if len(ph) >= VOL_WIN + 1:
                        rets = [(ph[j] - ph[j-1]) / ph[j-1]
                                for j in range(len(ph) - VOL_WIN, len(ph))]
                        avg = sum(rets) / len(rets)
                        vol = (sum((r - avg)**2 for r in rets) / (len(rets) - 1)) ** 0.5
                        vol_scalar = min(TARGET_VOL / max(vol, 1e-10), MAX_SCALE)

                    # Position target
                    go_long = momentum and btc_ok
                    base_qty = start_usd * ALLOC / price
                    target_qty = base_qty * vol_scalar if go_long else 0.0
                    trade_qty = target_qty - pos[coin]

                    tag = "LONG" if go_long else "FLAT"
                    print(f"[{coin}] ${price:,.2f} | {tag} | vol_scale={vol_scalar:.2f}x | "
                          f"pos={pos[coin]:.6f} -> {target_qty:.6f}")

                    # Dead-band: only rebalance if position is off by >2% of target
                    # (prevents constant small rebalances from vol scalar noise)
                    min_rebalance = max(target_qty * 0.02, 0.0001)
                    if trade_qty > min_rebalance:
                        signal = "BUY"
                    elif trade_qty < -min_rebalance:
                        signal = "SELL"
                        trade_qty = abs(trade_qty)
                else:
                    print(f"[{coin}] ${price:,.2f} | warming up {ticks[coin]}/{SLOW}")

                # Execute LIMIT order (round to exchange precision)
                if signal in ("BUY", "SELL"):
                    prec = amt_precision.get(coin, 4)
                    trade_qty = round(trade_qty, prec)
                    if trade_qty > 0:
                        qty_str = f"{trade_qty:.{prec}f}"
                        price_str = str(price)
                        try:
                            result = client.place_order(
                                pair=pair,
                                side=signal,
                                quantity=qty_str,
                                order_type="LIMIT",
                                price=price_str,
                            )
                            print(f"    >> {signal} {qty_str} {coin} @ ${price} (LIMIT) — OK")
                            if signal == "BUY":
                                pos[coin] += trade_qty
                            else:
                                pos[coin] -= trade_qty
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
                    pos[c] = bal[wallet_key].get(c, {}).get("Free", 0.0)
                usd = bal[wallet_key].get("USD", {}).get("Free", 0)

                tk = client.get_ticker()
                if tk and "Data" in tk:
                    coin_val = sum(
                        pos[c] * tk["Data"].get(f"{c}/USD", {}).get("LastPrice", 0)
                        for c in UNIVERSE
                    )
                    equity = usd + coin_val
                    pnl = (equity / start_usd - 1) * 100

                    print(f"\n{'=' * 60}")
                    print(f"  EQUITY: ${equity:,.2f}  |  P&L: {pnl:+.2f}%")
                    print(f"  Cash: ${usd:,.2f}  |  Coins: ${coin_val:,.2f}")
                    print(f"{'=' * 60}\n")
        except Exception:
            pass

        time.sleep(30)


if __name__ == '__main__':
    run_portfolio_bot()
