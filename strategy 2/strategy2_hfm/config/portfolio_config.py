from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Strategy2PortfolioConfig:
    """Portfolio rules for the direct spot-usable long/cash adaptation."""

    rebalance_hours: int = 1
    long_only: bool = True
    position_rule: str = "signal_above_threshold_div_universe"
    long_signal_threshold: float = 0.0
    bet_size: float | None = None

    def validate(self) -> None:
        if self.rebalance_hours <= 0:
            raise ValueError("rebalance_hours must be positive.")
        if not self.long_only:
            raise ValueError("This implementation supports long_only=True only.")
        if not math.isfinite(self.long_signal_threshold):
            raise ValueError("long_signal_threshold must be finite.")

        valid_rules = {
            "signal_above_threshold_div_universe",
            "fixed_pct_per_active_asset",
            "fixed_usd_per_active_asset",
        }
        if self.position_rule not in valid_rules:
            raise ValueError(f"Unknown position_rule: {self.position_rule}")

        if self.position_rule.startswith("fixed_"):
            if self.bet_size is None or not math.isfinite(self.bet_size) or self.bet_size <= 0.0:
                raise ValueError("bet_size must be a positive finite value for fixed bet sizing.")

        if self.position_rule == "fixed_pct_per_active_asset" and self.bet_size > 1.0:
            raise ValueError("For fixed_pct_per_active_asset, bet_size must be in (0, 1].")
