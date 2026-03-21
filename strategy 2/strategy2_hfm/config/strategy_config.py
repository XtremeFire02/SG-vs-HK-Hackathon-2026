from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

from .data_config import Strategy2DataConfig
from .execution_config import Strategy2ExecutionConfig
from .portfolio_config import Strategy2PortfolioConfig
from .signal_config import Strategy2SignalConfig


@dataclass(frozen=True)
class Strategy2Config:
    """Composite configuration object composed of nested config sections."""

    data: Strategy2DataConfig = field(default_factory=Strategy2DataConfig)
    signal: Strategy2SignalConfig = field(default_factory=Strategy2SignalConfig)
    portfolio: Strategy2PortfolioConfig = field(default_factory=Strategy2PortfolioConfig)
    execution: Strategy2ExecutionConfig = field(default_factory=Strategy2ExecutionConfig)

    def validate(self) -> None:
        self.data.validate()
        self.signal.validate()
        self.portfolio.validate()
        self.execution.validate()


def with_strategy2_updates(
    config: Strategy2Config,
    *,
    data: Optional[Mapping[str, Any]] = None,
    signal: Optional[Mapping[str, Any]] = None,
    portfolio: Optional[Mapping[str, Any]] = None,
    execution: Optional[Mapping[str, Any]] = None,
) -> Strategy2Config:
    """Convenience helper for notebook-side config tweaks."""

    return Strategy2Config(
        data=replace(config.data, **dict(data or {})),
        signal=replace(config.signal, **dict(signal or {})),
        portfolio=replace(config.portfolio, **dict(portfolio or {})),
        execution=replace(config.execution, **dict(execution or {})),
    )
