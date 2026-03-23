# Strategy 2 live bot: fixed USD per active asset only

This package is a **separate live-trading implementation** of your Strategy 2 HFM momentum setup, stripped down to the only path you asked to keep:

- long-only spot execution
- `fixed_usd_per_active_asset`
- active coins fixed to `FLOKI/USD`, `NEAR/USD`, `PEPE/USD`, `SHIB/USD` by default
- Binance 1h public kline warm-up for signal generation
- Roostoo test/live mode selection from environment variables

## Files

- `config.py`
  - environment loading
  - live bot settings
  - warm-up calculation from signal parameters
- `roostoo_client.py`
  - minimal Roostoo REST client
  - balance / ticker / order endpoints
- `binance_client.py`
  - Binance spot symbol mapping
  - closed 1h kline retrieval
- `signal_engine.py`
  - indexed-price HFM signal
  - EMA crossover + double volatility normalization + response function
- `bot.py`
  - startup reconciliation
  - session threshold calculation
  - hourly signal check and trade decisions
- `run_strategy2_hfm_live.py`
  - command-line entry point
- `env.example`
  - placeholder environment file

## Live behavior

### Startup

On startup the bot does the following:

1. checks Roostoo connectivity
2. loads Roostoo `exchangeInfo`
3. maps each active Roostoo pair to a Binance spot symbol
4. fetches current Roostoo balances and tickers
5. sells any **non-active** coin position that has free balance and can clear `MiniOrder`
6. refreshes balances
7. computes:

```text
effective_cash_reserve = min(input_cash_reserve, available_usd_free)
current_market_value   = usd_total + market value of all remaining crypto holdings
max_position_value_per_asset = (current_market_value - effective_cash_reserve) / 4
```

That per-asset cap is then **fixed for the entire process run**.

### Signal warm-up

The signal warm-up is derived only from the signal parameters:

```text
warmup_hours_lost = (short_norm_hours - 1) + (long_norm_hours - 1)
required_history_bars = warmup_hours_lost + 1 + extra_warmup_hours
```

So for your current baseline:

```text
short_norm_hours = 12
long_norm_hours  = 168
extra_warmup_hours = 0
```

the bot uses:

```text
warmup_hours_lost = 178
required_history_bars = 179
```

No extra warm-up is added unless you explicitly set `EXTRA_WARMUP_HOURS`.

### Hourly decision rule

For each active coin, every hour after the bar-close delay:

- compute the latest signal from Binance 1h closes
- fetch current Roostoo balances and tickers
- if `signal <= SIGNAL_THRESHOLD`:
  - sell the full **free** position if it is above `MiniOrder`
- if `signal > SIGNAL_THRESHOLD`:
  - **buy exactly one more `BET_SIZE_USD` block** only if:
    - current position market value + `BET_SIZE_USD` does **not** exceed the fixed per-asset cap
    - free USD after keeping `effective_cash_reserve` is enough
    - the resulting order clears `MiniOrder`

If a position drifts above the cap because the price rises, the bot **does not trim** it. It only sells on `signal <= SIGNAL_THRESHOLD`.

## Run

```bash
cp strategy2_hfm_live_bot/env.example .env
# edit .env
python -m strategy2_hfm_live_bot.run_strategy2_hfm_live --env-file .env
```

Or:

```bash
python /path/to/strategy2_hfm_live_bot/run_strategy2_hfm_live.py --env-file .env
```

## Notes

- The live bot keeps the strategy implementation separate from the backtest files.
- It is intentionally **not** trying to reproduce the paper’s short side.
- Market buys/sells use Roostoo market orders and Roostoo `AmountPrecision` / `MiniOrder` checks.
- The bot prints detailed startup, mapping, signal, decision, and order logs, and also writes JSONL logs.
