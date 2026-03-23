import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from dotenv import load_dotenv

from roostoo_client import RoostooClient


EPSILON = 1e-12
LINE = "=" * 100
SUBLINE = "-" * 100


@dataclass
class BalanceRow:
    asset: str
    free: float
    locked: float

    @property
    def total(self) -> float:
        return float(self.free + self.locked)


@dataclass
class HoldingSummary:
    asset: str
    free: float
    locked: float
    total: float
    pair: Optional[str]
    last_price_usd: Optional[float]
    estimated_value_usd: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Roostoo wallet balances and show USD balance plus all held coins."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file containing Roostoo credentials.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also print the raw /v3/balance response.",
    )
    parser.add_argument(
        "--show-zero",
        action="store_true",
        help="Include zero-balance assets in the holdings table.",
    )
    return parser.parse_args()


def pick_credentials() -> tuple[str, str, str, str]:
    mode = os.getenv("ROOSTOO_MODE", "TEST").upper()

    if mode == "LIVE":
        base_url = os.getenv("ROOSTOO_LIVE_BASE_URL")
        api_key = os.getenv("ROOSTOO_LIVE_API_KEY")
        secret_key = os.getenv("ROOSTOO_LIVE_SECRET_KEY")
    else:
        base_url = os.getenv("ROOSTOO_TEST_BASE_URL", "https://mock-api.roostoo.com")
        api_key = os.getenv("ROOSTOO_TEST_API_KEY")
        secret_key = os.getenv("ROOSTOO_TEST_SECRET_KEY")

    if not api_key or not secret_key or not base_url:
        raise ValueError(
            f"Missing Roostoo credentials for mode={mode}. Check your .env file."
        )

    return mode, base_url, api_key, secret_key


def format_number(value: Optional[float]) -> str:
    if value is None:
        return "N/A"

    number = float(value)
    abs_number = abs(number)

    if abs_number == 0:
        return "0"
    if abs_number >= 1_000_000:
        rendered = f"{number:,.2f}"
    elif abs_number >= 1:
        rendered = f"{number:,.6f}"
    elif abs_number >= 1e-4:
        rendered = f"{number:.8f}"
    else:
        rendered = f"{number:.6g}"

    if "e" not in rendered.lower():
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def get_first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def extract_wallet(payload: Mapping[str, Any]) -> Dict[str, BalanceRow]:
    wallet = get_first_present(payload, "SpotWallet", "Wallet", "balances", "balance", "assets", default=None)

    if wallet is None and isinstance(payload.get("data"), dict):
        wallet = payload["data"]

    balances: Dict[str, BalanceRow] = {}

    if isinstance(wallet, dict):
        for asset, amounts in wallet.items():
            if isinstance(amounts, Mapping):
                free = float(get_first_present(amounts, "Free", "free", "available", "balance", "amount", default=0.0) or 0.0)
                locked = float(get_first_present(amounts, "Lock", "locked", "hold", "frozen", default=0.0) or 0.0)
            else:
                free = float(amounts or 0.0)
                locked = 0.0
            balances[str(asset)] = BalanceRow(asset=str(asset), free=free, locked=locked)
        return balances

    if isinstance(wallet, list):
        for row in wallet:
            if not isinstance(row, Mapping):
                continue
            asset = str(get_first_present(row, "asset", "Asset", "currency", "coin", "symbol", default="UNKNOWN"))
            free = float(get_first_present(row, "Free", "free", "available", "balance", "amount", default=0.0) or 0.0)
            locked = float(get_first_present(row, "Lock", "locked", "hold", "frozen", default=0.0) or 0.0)
            balances[asset] = BalanceRow(asset=asset, free=free, locked=locked)
        return balances

    raise RuntimeError(
        "Could not parse balances from the response. Expected SpotWallet/Wallet/balances/assets."
    )


def resolve_usd_asset(balances: Mapping[str, BalanceRow]) -> str:
    for candidate in ("USD", "USDT", "USDC"):
        if candidate in balances:
            return candidate
    return "USD"


def extract_last_prices(payload: Mapping[str, Any]) -> Dict[str, float]:
    data = get_first_present(payload, "Data", "data", default={})
    prices: Dict[str, float] = {}

    if isinstance(data, Mapping):
        for pair, row in data.items():
            if not isinstance(row, Mapping):
                continue
            last_price = get_first_present(row, "LastPrice", "last_price", "last", default=None)
            if last_price is None:
                continue
            try:
                prices[str(pair)] = float(last_price)
            except (TypeError, ValueError):
                continue

    return prices


def safe_get_tickers(client: RoostooClient) -> Dict[str, float]:
    try:
        payload = client.get_ticker()
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        print(f"\nWarning: could not fetch tickers for USD valuation: {exc}")
        return {}

    try:
        return extract_last_prices(payload)
    except Exception as exc:  # pragma: no cover - defensive runtime fallback
        print(f"\nWarning: could not parse ticker payload for USD valuation: {exc}")
        return {}


def build_holding_summaries(
    balances: Mapping[str, BalanceRow],
    prices_by_pair: Mapping[str, float],
    *,
    usd_asset: str,
    show_zero: bool,
) -> list[HoldingSummary]:
    rows: list[HoldingSummary] = []

    for asset, balance in balances.items():
        if not show_zero and abs(balance.total) <= EPSILON:
            continue

        if asset == usd_asset:
            pair = None
            last_price_usd = 1.0
            estimated_value_usd = balance.total
        else:
            pair = f"{asset}/{usd_asset}"
            last_price_usd = prices_by_pair.get(pair)
            estimated_value_usd = None if last_price_usd is None else balance.total * last_price_usd

        rows.append(
            HoldingSummary(
                asset=asset,
                free=balance.free,
                locked=balance.locked,
                total=balance.total,
                pair=pair,
                last_price_usd=last_price_usd,
                estimated_value_usd=estimated_value_usd,
            )
        )

    def sort_key(row: HoldingSummary) -> tuple[int, float, str]:
        if row.asset == usd_asset:
            return (0, 0.0, row.asset)
        value_sort = row.estimated_value_usd if row.estimated_value_usd is not None else -1.0
        return (1, -value_sort, row.asset)

    return sorted(rows, key=sort_key)


def print_section_title(title: str) -> None:
    print("\n" + LINE)
    print(title)
    print(SUBLINE)


def print_metadata(mode: str, base_url: str, usd_asset: str) -> None:
    print_section_title("ROOSTOO ACCOUNT BALANCE SNAPSHOT")
    print(f"Mode      : {mode}")
    print(f"Base URL  : {base_url}")
    print(f"USD Asset : {usd_asset}")


def print_cash_summary(cash_balance: Optional[BalanceRow], usd_asset: str) -> None:
    print("\nCash Balance")
    print(f"  Asset   : {usd_asset}")
    print(f"  Free    : {format_number(cash_balance.free if cash_balance else 0.0)}")
    print(f"  Locked  : {format_number(cash_balance.locked if cash_balance else 0.0)}")
    print(f"  Total   : {format_number(cash_balance.total if cash_balance else 0.0)}")


def render_table(headers: list[str], rows: Iterable[list[str]]) -> None:
    rows = list(rows)
    widths = [len(header) for header in headers]

    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))


def print_holdings_table(holdings: list[HoldingSummary], usd_asset: str) -> None:
    coin_rows = [row for row in holdings if row.asset != usd_asset]

    print("\nCoin Holdings")
    if not coin_rows:
        print("  No non-USD coin holdings found.")
        return

    table_rows = [
        [
            row.asset,
            format_number(row.free),
            format_number(row.locked),
            format_number(row.total),
            format_number(row.last_price_usd),
            format_number(row.estimated_value_usd),
            row.pair or "N/A",
        ]
        for row in coin_rows
    ]

    render_table(
        ["Asset", "Free", "Locked", "Total", "Last Price (USD)", "Est. Value (USD)", "Pair"],
        table_rows,
    )


def print_total_value(holdings: list[HoldingSummary]) -> None:
    estimated_total = 0.0
    missing_values: list[str] = []

    for row in holdings:
        if row.estimated_value_usd is None:
            missing_values.append(row.asset)
            continue
        estimated_total += row.estimated_value_usd

    print("\nPortfolio Summary")
    print(f"  Estimated total account value (USD): {format_number(estimated_total)}")
    if missing_values:
        print(
            "  Assets missing USD valuation        : "
            + ", ".join(sorted(missing_values))
        )

    print(LINE)


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)

    mode, base_url, api_key, secret_key = pick_credentials()

    client = RoostooClient(
        api_key=api_key,
        secret_key=secret_key,
        base_url=base_url,
        timeout=int(os.getenv("ROOSTOO_TIMEOUT_SECONDS", "10")),
        max_retries=int(os.getenv("ROOSTOO_MAX_RETRIES", "3")),
        retry_backoff_seconds=float(os.getenv("ROOSTOO_RETRY_BACKOFF_SECONDS", "0.5")),
    )

    balance_payload = client.get_balance()
    balances = extract_wallet(balance_payload)
    usd_asset = resolve_usd_asset(balances)
    prices_by_pair = safe_get_tickers(client)
    holdings = build_holding_summaries(
        balances,
        prices_by_pair,
        usd_asset=usd_asset,
        show_zero=args.show_zero,
    )

    print_metadata(mode, base_url, usd_asset)
    print_cash_summary(balances.get(usd_asset), usd_asset)
    print_holdings_table(holdings, usd_asset)
    print_total_value(holdings)

    if args.show_raw:
        print("\nRaw /v3/balance response")
        print(SUBLINE)
        print(json.dumps(balance_payload, indent=2))


if __name__ == "__main__":
    main()
