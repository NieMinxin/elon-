from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from polymarket_bot.config import BotConfig


class ActionKind(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Decision:
    action: ActionKind
    reason: str
    amount_usd: Decimal = Decimal("0")
    shares: Decimal = Decimal("0")


def decide(
    *,
    config: BotConfig,
    buy_price: Decimal,
    sell_price: Decimal,
    position_shares: Decimal,
    daily_buy_usd: Decimal,
    last_trade_at: datetime | None,
    now: datetime,
) -> Decision:
    if last_trade_at is not None:
        elapsed = (now - last_trade_at).total_seconds()
        if elapsed < config.trade_cooldown_seconds:
            return Decision(ActionKind.HOLD, "trade cooldown is active")

    if position_shares > 0:
        if sell_price >= config.take_profit_at:
            return Decision(
                ActionKind.SELL,
                f"sell price {sell_price} reached take-profit {config.take_profit_at}",
                shares=position_shares,
            )
        if sell_price <= config.stop_loss_at:
            return Decision(
                ActionKind.SELL,
                f"sell price {sell_price} reached stop-loss {config.stop_loss_at}",
                shares=position_shares,
            )

    if buy_price > config.buy_below:
        return Decision(
            ActionKind.HOLD,
            f"buy price {buy_price} is above entry threshold {config.buy_below}",
        )

    remaining_daily = config.max_daily_buy_usd - daily_buy_usd
    if remaining_daily <= 0:
        return Decision(ActionKind.HOLD, "daily buy limit reached")

    remaining_shares = config.max_position_shares - position_shares
    if remaining_shares <= 0:
        return Decision(ActionKind.HOLD, "position limit reached")

    amount = min(config.order_amount_usd, remaining_daily, remaining_shares * buy_price)
    if amount <= 0:
        return Decision(ActionKind.HOLD, "no buy capacity remains")
    return Decision(
        ActionKind.BUY,
        f"buy price {buy_price} is at or below entry threshold {config.buy_below}",
        amount_usd=amount,
    )


def buy_price_cap(price: Decimal, slippage: Decimal) -> Decimal:
    return min(Decimal("0.9999"), price * (Decimal("1") + slippage))


def sell_price_floor(price: Decimal, slippage: Decimal) -> Decimal:
    return max(Decimal("0.0001"), price * (Decimal("1") - slippage))
