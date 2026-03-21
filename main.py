import requests
import hashlib
import hmac
import time
import os

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
SECRET = os.getenv("SECRET")
BASE_URL = "https://mock-api.roostoo.com"

# ─── API LAYER ────────────────────────────────────────────────

def _sign(params):
    query = '&'.join(f"{k}={params[k]}" for k in sorted(params))
    return hmac.new(SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

def _auth(params):
    return {"RST-API-KEY": API_KEY, "MSG-SIGNATURE": _sign(params)}

def get_server_time():
    try:
        return requests.get(f"{BASE_URL}/v3/serverTime").json()
    except Exception:
        return None

def get_ex_info():
    try:
        return requests.get(f"{BASE_URL}/v3/exchangeInfo").json()
    except Exception:
        return None

def get_ticker(pair=None):
    try:
        p = {"timestamp": int(time.time() * 1000)}
        if pair:
            p["pair"] = pair
        return requests.get(f"{BASE_URL}/v3/ticker", params=p).json()
    except Exception:
        return None

def get_balance():
    try:
        p = {"timestamp": int(time.time() * 1000)}
        return requests.get(f"{BASE_URL}/v3/balance", params=p, headers=_auth(p)).json()
    except Exception:
        return None

def place_order(coin, side, qty, price=None):
    """Place an order. LIMIT if price given (half the fee), else MARKET."""
    try:
        p = {
            "timestamp": int(time.time() * 1000),
            "pair": f"{coin}/USD",
            "side": side,
            "quantity": round(qty, 6),
        }
        if price:
            p["type"] = "LIMIT"
            p["price"] = price
        else:
            p["type"] = "MARKET"
        r = requests.post(
            f"{BASE_URL}/v3/place_order",
            data=p,
            headers={"RST-API-KEY": API_KEY, "MSG-SIGNATURE": _sign({"timestamp": p["timestamp"]})}
        )
        if r.status_code == 200:
            order_type = "LIMIT" if price else "MARKET"
            print(f"    >> {side} {qty:.6f} {coin} @ ${price} ({order_type}) — OK")
            return r.json()
        print(f"    >> {side} {qty:.6f} {coin} — FAILED ({r.status_code}): {r.text}")
        return None
    except Exception as e:
        print(f"    >> {side} {coin} — ERROR: {e}")
        return None

def cancel_orders(pair):
    try:
        p = {"timestamp": int(time.time() * 1000), "pair": pair}
        return requests.post(f"{BASE_URL}/v3/cancel_order", data=p, headers=_auth(p)).json()
    except Exception:
        return None

def query_order():
    try:
        p = {"timestamp": int(time.time() * 1000)}
        return requests.post(f"{BASE_URL}/v3/query_order", data=p, headers=_auth(p)).json()
    except Exception:
        return None

def pending_count():
    try:
        p = {"timestamp": int(time.time() * 1000)}
        return requests.get(f"{BASE_URL}/v3/pending_count", params=p, headers=_auth(p)).json()
    except Exception:
        return None

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

    # ── Load exchange precision rules ──
    ex_info = get_ex_info()
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
    bal = get_balance()
    pos = {c: 0.0 for c in UNIVERSE}
    start_usd = 50000.0

    if bal and "SpotWallet" in bal:
        for c in UNIVERSE:
            pos[c] = bal["SpotWallet"].get(c, {}).get("Free", 0.0)
        start_usd = bal["SpotWallet"].get("USD", {}).get("Free", 50000.0)

    print(f"  Starting USD:       ${start_usd:,.2f}")
    print(f"  Starting positions: {pos}")
    print(f"  Strategy:           EMA({FAST},{SLOW}) + VolTarget + BTC Filter")
    print("-" * 60)

    # ── Main loop ──
    while True:
        for coin in UNIVERSE:
            try:
                pair = f"{coin}/USD"
                td = get_ticker(pair)
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
                if len(hist[coin]) > SLOW * 2:
                    hist[coin] = hist[coin][-SLOW:]

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
                        if place_order(coin, signal, trade_qty, price=price):
                            if signal == "BUY":
                                pos[coin] += trade_qty
                            else:
                                pos[coin] -= trade_qty

                time.sleep(1)

            except Exception as e:
                print(f"[{coin}] ERROR: {e}")

        # ── Re-sync positions & report P&L ──
        try:
            bal = get_balance()
            if bal and "SpotWallet" in bal:
                for c in UNIVERSE:
                    pos[c] = bal["SpotWallet"].get(c, {}).get("Free", 0.0)
                usd = bal["SpotWallet"].get("USD", {}).get("Free", 0)

                tk = get_ticker()
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
