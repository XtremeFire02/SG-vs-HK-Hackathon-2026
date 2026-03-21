from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy2ExecutionConfig:
    """Backtest execution assumptions for the long/cash spot adaptation."""

    initial_cash: float = 50_000.0
    fee_rate: float = 0.0010
    slippage_bps: float = 5.0
    mini_order_nav_fraction: float = 0.005
    rebalance_buffer: float = 0.0

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        if self.fee_rate < 0:
            raise ValueError("fee_rate cannot be negative.")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative.")
        if self.mini_order_nav_fraction < 0:
            raise ValueError("mini_order_nav_fraction cannot be negative.")
        if self.rebalance_buffer < 0:
            raise ValueError("rebalance_buffer cannot be negative.")
