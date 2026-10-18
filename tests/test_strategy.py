from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from polymarket_bot.config import BotConfig
from polymarket_bot.strategy import (
    ActionKind,
    buy_price_cap,
    decide,
    sell_price_floor,
)

NOW = datetime(2026, 10, 17, tzinfo=UTC)


def config() -> BotConfig:
    return BotConfig.from_env(
        {
            "POLYMARKET_MARKET_SLUG": "example-market",
            "POLYMARKET_BUY_BELOW": "0.40",
            "POLYMARKET_TAKE_PROFIT_AT": "0.60",
            "POLYMARKET_STOP_LOSS_AT": "0.25",
            "POLYMARKET_ORDER_AMOUNT_USD": "10",
            "POLYMARKET_MAX_POSITION_SHARES": "50",
            "POLYMARKET_MAX_DAILY_BUY_USD": "25",
            "POLYMARKET_TRADE_COOLDOWN_SECONDS": "300",
        }
    )


def evaluate(**overrides: object):
    params = {
        "config": config(),
        "buy_price": Decimal("0.39"),
        "sell_price": Decimal("0.38"),
        "position_shares": Decimal("0"),
        "daily_buy_usd": Decimal("0"),
        "last_trade_at": None,
        "now": NOW,
    }
    params.update(overrides)
    return decide(**params)  # type: ignore[arg-type]


def test_buys_below_threshold() -> None:
    decision = evaluate()
    assert decision.action is ActionKind.BUY
    assert decision.amount_usd == Decimal("10")


def test_take_profit_sells_entire_position() -> None:
    decision = evaluate(position_shares=Decimal("12.5"), sell_price=Decimal("0.61"))
    assert decision.action is ActionKind.SELL
    assert decision.shares == Decimal("12.5")


def test_stop_loss_sells_entire_position() -> None:
    decision = evaluate(position_shares=Decimal("12.5"), sell_price=Decimal("0.24"))
    assert decision.action is ActionKind.SELL


def test_position_limit_caps_buy_amount() -> None:
    decision = evaluate(position_shares=Decimal("49"))
    assert decision.action is ActionKind.BUY
    assert decision.amount_usd == Decimal("0.39")


def test_daily_limit_caps_buy_amount() -> None:
    decision = evaluate(daily_buy_usd=Decimal("23"))
    assert decision.amount_usd == Decimal("2")


def test_cooldown_holds() -> None:
    decision = evaluate(last_trade_at=NOW - timedelta(seconds=10))
    assert decision.action is ActionKind.HOLD


def test_price_protection_bounds() -> None:
    cfg = replace(config(), max_slippage=Decimal("0.02"))
    assert buy_price_cap(Decimal("0.40"), cfg.max_slippage) == Decimal("0.4080")
    assert sell_price_floor(Decimal("0.40"), cfg.max_slippage) == Decimal("0.3920")
