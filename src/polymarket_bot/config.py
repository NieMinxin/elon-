from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


class ConfigError(ValueError):
    """Raised when bot configuration is missing or unsafe."""


def _decimal(env: Mapping[str, str], name: str, default: str) -> Decimal:
    raw = env.get(name, default).strip()
    try:
        return Decimal(raw)
    except InvalidOperation as error:
        raise ConfigError(f"{name} must be a decimal, got {raw!r}") from error


def _integer(env: Mapping[str, str], name: str, default: str) -> int:
    raw = env.get(name, default).strip()
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from error


def _boolean(env: Mapping[str, str], name: str, default: str = "false") -> bool:
    raw = env.get(name, default).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class BotConfig:
    market_slug: str
    outcome: str
    buy_below: Decimal
    take_profit_at: Decimal
    stop_loss_at: Decimal
    order_amount_usd: Decimal
    max_position_shares: Decimal
    max_daily_buy_usd: Decimal
    max_slippage: Decimal
    poll_interval_seconds: int
    trade_cooldown_seconds: int
    state_path: Path
    enable_live: bool
    private_key: str | None
    wallet_address: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> BotConfig:
        values = os.environ if env is None else env
        config = cls(
            market_slug=values.get("POLYMARKET_MARKET_SLUG", "").strip(),
            outcome=values.get("POLYMARKET_OUTCOME", "YES").strip().upper(),
            buy_below=_decimal(values, "POLYMARKET_BUY_BELOW", "0.40"),
            take_profit_at=_decimal(values, "POLYMARKET_TAKE_PROFIT_AT", "0.60"),
            stop_loss_at=_decimal(values, "POLYMARKET_STOP_LOSS_AT", "0.25"),
            order_amount_usd=_decimal(values, "POLYMARKET_ORDER_AMOUNT_USD", "10"),
            max_position_shares=_decimal(values, "POLYMARKET_MAX_POSITION_SHARES", "50"),
            max_daily_buy_usd=_decimal(values, "POLYMARKET_MAX_DAILY_BUY_USD", "25"),
            max_slippage=_decimal(values, "POLYMARKET_MAX_SLIPPAGE", "0.02"),
            poll_interval_seconds=_integer(values, "POLYMARKET_POLL_INTERVAL_SECONDS", "30"),
            trade_cooldown_seconds=_integer(values, "POLYMARKET_TRADE_COOLDOWN_SECONDS", "300"),
            state_path=Path(values.get("POLYMARKET_STATE_PATH", "bot-state.json")),
            enable_live=_boolean(values, "POLYMARKET_ENABLE_LIVE"),
            private_key=values.get("POLYMARKET_PRIVATE_KEY") or None,
            wallet_address=values.get("POLYMARKET_WALLET_ADDRESS") or None,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.market_slug:
            raise ConfigError("POLYMARKET_MARKET_SLUG is required")
        if self.outcome not in {"YES", "NO"}:
            raise ConfigError("POLYMARKET_OUTCOME must be YES or NO")
        for name, value in (
            ("POLYMARKET_BUY_BELOW", self.buy_below),
            ("POLYMARKET_TAKE_PROFIT_AT", self.take_profit_at),
            ("POLYMARKET_STOP_LOSS_AT", self.stop_loss_at),
        ):
            if not Decimal("0") < value < Decimal("1"):
                raise ConfigError(f"{name} must be greater than 0 and less than 1")
        if self.stop_loss_at >= self.buy_below:
            raise ConfigError("POLYMARKET_STOP_LOSS_AT must be below POLYMARKET_BUY_BELOW")
        if self.take_profit_at <= self.buy_below:
            raise ConfigError("POLYMARKET_TAKE_PROFIT_AT must be above POLYMARKET_BUY_BELOW")
        for name, value in (
            ("POLYMARKET_ORDER_AMOUNT_USD", self.order_amount_usd),
            ("POLYMARKET_MAX_POSITION_SHARES", self.max_position_shares),
            ("POLYMARKET_MAX_DAILY_BUY_USD", self.max_daily_buy_usd),
        ):
            if value <= 0:
                raise ConfigError(f"{name} must be greater than 0")
        if not Decimal("0") <= self.max_slippage <= Decimal("0.20"):
            raise ConfigError("POLYMARKET_MAX_SLIPPAGE must be between 0 and 0.20")
        if self.poll_interval_seconds < 5:
            raise ConfigError("POLYMARKET_POLL_INTERVAL_SECONDS must be at least 5")
        if self.trade_cooldown_seconds < 0:
            raise ConfigError("POLYMARKET_TRADE_COOLDOWN_SECONDS cannot be negative")

    def require_live_credentials(self) -> tuple[str, str]:
        if not self.private_key:
            raise ConfigError("POLYMARKET_PRIVATE_KEY is required in live mode")
        if not self.wallet_address:
            raise ConfigError("POLYMARKET_WALLET_ADDRESS is required in live mode")
        return self.private_key, self.wallet_address
