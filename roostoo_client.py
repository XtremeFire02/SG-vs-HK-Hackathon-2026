import hashlib
import hmac
import time
from typing import Any, Dict, Optional

import requests


class RoostooClient:
    """Minimal Roostoo REST client with light retry/backoff support.

    This keeps the signing logic from the current pipeline, but adds a shared
    request helper so the strategy code can rely on cleaner error handling.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str = "https://mock-api.roostoo.com",
        timeout: int = 10,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.session = requests.Session()

    @staticmethod
    def _timestamp_ms() -> str:
        return str(int(time.time() * 1000))

    @staticmethod
    def _serialize_params(params: Dict[str, Any]) -> str:
        clean_params = {k: v for k, v in params.items() if v is not None}
        sorted_keys = sorted(clean_params.keys())
        return "&".join(f"{k}={clean_params[k]}" for k in sorted_keys)

    def _signature(self, total_params: str) -> str:
        return hmac.new(
            self.secret_key.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _signed_headers(self, total_params: str, is_post: bool = False) -> Dict[str, str]:
        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": self._signature(total_params),
        }
        if is_post:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        return headers

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Dict[str, Any]:
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=self.timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                sleep_seconds = self.retry_backoff_seconds * (2 ** attempt)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        raise RuntimeError(f"Roostoo request failed after {self.max_retries} attempts: {last_error}")

    def get_server_time(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/serverTime"
        return self._request_json("GET", url)

    def get_exchange_info(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/exchangeInfo"
        return self._request_json("GET", url)

    def get_ticker(self, pair: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/ticker"
        params: Dict[str, Any] = {"timestamp": self._timestamp_ms()}
        if pair:
            params["pair"] = pair
        return self._request_json("GET", url, params=params)

    def get_balance(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/balance"
        params = {"timestamp": self._timestamp_ms()}
        total_params = self._serialize_params(params)
        headers = self._signed_headers(total_params, is_post=False)
        return self._request_json("GET", url, headers=headers, params=params)

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: str,
        order_type: str = "MARKET",
        price: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/place_order"
        payload: Dict[str, Any] = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
            "timestamp": self._timestamp_ms(),
        }

        if payload["type"] == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require a price.")
            payload["price"] = str(price)

        total_params = self._serialize_params(payload)
        headers = self._signed_headers(total_params, is_post=True)
        return self._request_json("POST", url, headers=headers, data=total_params)

    def query_order(
        self,
        order_id: Optional[str] = None,
        pair: Optional[str] = None,
        pending_only: Optional[bool] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/query_order"
        payload: Dict[str, Any] = {"timestamp": self._timestamp_ms()}

        if order_id is not None:
            payload["order_id"] = str(order_id)
        else:
            if pair is not None:
                payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if offset is not None:
                payload["offset"] = str(offset)
            if limit is not None:
                payload["limit"] = str(limit)

        total_params = self._serialize_params(payload)
        headers = self._signed_headers(total_params, is_post=True)
        return self._request_json("POST", url, headers=headers, data=total_params)

    def cancel_order(self, pair: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/cancel_order"
        payload: Dict[str, Any] = {
            "timestamp": self._timestamp_ms(),
            "pair": pair,
        }
        total_params = self._serialize_params(payload)
        headers = self._signed_headers(total_params, is_post=True)
        return self._request_json("POST", url, headers=headers, data=total_params)

    def pending_count(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v3/pending_count"
        params = {"timestamp": self._timestamp_ms()}
        total_params = self._serialize_params(params)
        headers = self._signed_headers(total_params, is_post=False)
        return self._request_json("GET", url, headers=headers, params=params)

    def get_trade_pair_rules(self, pair: str, exchange_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = exchange_info if exchange_info is not None else self.get_exchange_info()
        trade_pairs = info.get("TradePairs", {})
        if pair not in trade_pairs:
            available = ", ".join(sorted(trade_pairs.keys())[:10])
            raise KeyError(f"{pair} not found in exchangeInfo. Sample available pairs: {available}")
        return trade_pairs[pair]
