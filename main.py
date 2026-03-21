import requests
import hashlib
import hmac
import time
import statistics
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
SECRET = os.getenv("SECRET")

BASE_URL = "https://mock-api.roostoo.com"

def generate_signature(params):
    query_string = '&'.join(["{}={}".format(k, params[k]) for k in sorted(params.keys())])
    us = SECRET.encode('utf-8')
    m = hmac.new(us, query_string.encode('utf-8'), hashlib.sha256)
    return m.hexdigest()

def get_server_time():
    r = requests.get(BASE_URL + "/v3/serverTime")
    print(r.status_code, r.text)
    return r.json()

def get_ex_info():
    r = requests.get(BASE_URL + "/v3/exchangeInfo")
    print(r.status_code, r.text)
    return r.json()

def get_ticker(pair=None):
    payload = {"timestamp": int(time.time() * 1000)}
    if pair:
        payload["pair"] = pair
    r = requests.get(BASE_URL + "/v3/ticker", params=payload)
    print(r.status_code, r.text)
    return r.json()

def get_balance():
    payload = {"timestamp": int(time.time() * 1000)}
    r = requests.get(
        BASE_URL + "/v3/balance",
        params=payload,
        headers={"RST-API-KEY": API_KEY, "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)
    return r.json()

def place_order(coin, side, qty, price=None):
    payload = {
        "timestamp": int(time.time() * 1000),
        "pair": coin + "/USD",
        "side": side,
        "quantity": qty,
    }
    if not price:
        payload['type'] = "MARKET"
    else:
        payload['type'] = "LIMIT"
        payload['price'] = price

    r = requests.post(
        BASE_URL + "/v3/place_order",
        data=payload,
        headers={"RST-API-KEY": API_KEY, "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)

def cancel_order():
    payload = {
        "timestamp": int(time.time() * 1000),
        "pair": "BTC/USD",
    }
    r = requests.post(
        BASE_URL + "/v3/cancel_order",
        data=payload,
        headers={"RST-API-KEY": API_KEY, "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)

def query_order():
    payload = {
        "timestamp": int(time.time() * 1000),
    }
    r = requests.post(
        BASE_URL + "/v3/query_order",
        data=payload,
        headers={"RST-API-KEY": API_KEY, "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)

def pending_count():
    payload = {
        "timestamp": int(time.time() * 1000),
    }
    r = requests.get(
        BASE_URL + "/v3/pending_count",
        params=payload,
        headers={"RST-API-KEY": API_KEY, "MSG-SIGNATURE": generate_signature(payload)}
    )
    print(r.status_code, r.text)
    return r.json()

if __name__ == '__main__':
    get_server_time()
    # get_ex_info()
    # get_ticker()
    get_balance()

def run_portfolio_bot():
    print("Starting Portfolio Bot. Press Ctrl+C to stop.")
    
    # 1. Define your portfolio universe
    universe = ["BTC", "ETH", "SOL", "BNB"]
    
    price_history = {}
    LOOKBACK_PERIOD = 60 # every 30mins we get rolling mean
    positions = {coin: 0.0 for coin in universe}
    while True:
        for coin in universe:
            try:
                pair = f"{coin}/USD"
                ticker_data = get_ticker(pair)
                
                if ticker_data and "Data" in ticker_data and pair in ticker_data["Data"]:
                    price = ticker_data["Data"][pair]["LastPrice"]
                    print(f"[{coin}] Current Price: ${price}")
                    
                    # 2. Strategy Logic: Mean Reversion
                    if coin not in price_history:
                        price_history[coin] = []
                    price_history[coin].append(price)

                    # Remove oldest price if we exceed the lookback window
                    if len(price_history[coin]) > LOOKBACK_PERIOD:
                        price_history[coin].pop(0)

                    signal = "HOLD" 

                    # Only calculate if we have a full dataset
                    if len(price_history[coin]) == LOOKBACK_PERIOD:
                        n = LOOKBACK_PERIOD
                        x = list(range(n))
                        y = price_history[coin]
                        
                        # 1. Calculate Linear Regression Line (y = mx + c)
                        sum_x = sum(x)
                        sum_y = sum(y)
                        sum_xy = sum(x[i] * y[i] for i in range(n))
                        sum_x2 = sum(x[i] ** 2 for i in range(n))
                        
                        m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                        c = (sum_y - m * sum_x) / n
                        
                        # 2. Calculate residuals (Actual Price - Trendline Price)
                        residuals = [y[i] - (m * x[i] + c) for i in range(n)]
                        
                        # 3. Z-Score of the current residual
                        current_residual = residuals[-1] # the most recent one
                        res_sigma = statistics.stdev(residuals)
                        
                        z_score = current_residual / res_sigma if res_sigma > 0 else 0
                        
                        print(f"[{coin}] Z-Score (Detrended): {z_score:.2f}")

                        # Determine Target Exposure based on Z-Score
                        target_qty = 0.0
                        if z_score < -3.0:
                            target_qty = 0.05  # Max position
                        elif z_score < -2.0:
                            target_qty = 0.03  # Medium position
                        elif z_score < -1.5:
                            target_qty = 0.01  # Entry position
                        elif z_score > 1.5:
                            target_qty = 0.0   # Sell everything

                        # Calculate difference between target and current inventory
                        trade_qty = target_qty - positions[coin]
                        
                        if trade_qty > 0.001:  # Buffer to avoid tiny floating point trades
                            signal = "BUY"
                        elif trade_qty < -0.001:
                            signal = "SELL"
                            trade_qty = abs(trade_qty) # Convert to positive number for the API
                    else:
                        print(f"[{coin}] Warming up: {len(price_history[coin])}/{LOOKBACK_PERIOD}")
                    
                    # 3. Execute Orders
                    if signal == "BUY":
                        print(f"Executing BUY for {trade_qty:.2f} {coin}...")
                        place_order(coin, "BUY", trade_qty) 
                        positions[coin] += trade_qty  
                        
                    elif signal == "SELL":
                        print(f"Executing SELL for {trade_qty:.2f} {coin}...")
                        place_order(coin, "SELL", trade_qty) 
                        positions[coin] -= trade_qty
                
                # Sleep briefly to respect API rate limits
                time.sleep(1) 
                
            except Exception as e:
                print(f"Error analyzing {coin}: {e}")

        # 4. Wait before the next full portfolio scan
        print("Portfolio scan complete. Sleeping for 30 seconds...\n")
        time.sleep(30)

if __name__ == '__main__':
    run_portfolio_bot()