from decimal import Decimal

import pytest

from polymarket_bot.elon_config import ElonArbConfig
from polymarket_bot.elon_strategy import (
    ArbActionKind,
    Interval,
    MarketQuote,
    PositionView,
    decide_action,
    parse_interval,
    taker_fee_per_share,
)


def config() -> ElonArbConfig:
    return ElonArbConfig.from_env({})


def quote(*, fair: str, bid: str = "0.30", ask: str = "0.32") -> MarketQuote:
    return MarketQuote(
        slug="market-340-359",
        asset_id="asset",
        condition_id="condition",
        interval=Interval("340-359", 340, 359),
        bid=Decimal(bid),
        ask=Decimal(ask),
        fee_rate=Decimal("0.05"),
        minimum_order_size=Decimal("5"),
        fair_probability=Decimal(fair),
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("<20", (0, 19)),
        ("20-39", (20, 39)),
        ("500+", (500, None)),
    ],
)
def test_parse_interval(label: str, expected: tuple[int, int | None]) -> None:
    interval = parse_interval(label)
    assert (interval.lower, interval.upper) == expected


def test_fee_formula() -> None:
    assert taker_fee_per_share(Decimal("0.50"), Decimal("0.05")) == Decimal("0.0125")


def test_enters_only_after_fee_adjusted_edge() -> None:
    decision = decide_action(
        config=config(),
        quotes=[quote(fair="0.40")],
        positions={},
        daily_buy_usd=Decimal("0"),
        minutes_remaining=Decimal("100"),
    )
    assert decision.action is ArbActionKind.BUY


def test_sells_when_market_net_bid_exceeds_model_value() -> None:
    item = quote(fair="0.20", bid="0.30")
    decision = decide_action(
        config=config(),
        quotes=[item],
        positions={"asset": PositionView(Decimal("10"), Decimal("3"))},
        daily_buy_usd=Decimal("0"),
        minutes_remaining=Decimal("100"),
    )
    assert decision.action is ArbActionKind.SELL


def test_holds_when_edge_is_too_small() -> None:
    decision = decide_action(
        config=config(),
        quotes=[quote(fair="0.34")],
        positions={},
        daily_buy_usd=Decimal("0"),
        minutes_remaining=Decimal("100"),
    )
    assert decision.action is ArbActionKind.HOLD


def test_market_order_estimates_walk_order_book_depth() -> None:
    item = MarketQuote(
        slug="market",
        asset_id="asset",
        condition_id="condition",
        interval=Interval("340-359", 340, 359),
        bid=Decimal("0.30"),
        ask=Decimal("0.32"),
        fee_rate=Decimal("0"),
        minimum_order_size=Decimal("1"),
        fair_probability=Decimal("0.40"),
        bids=((Decimal("0.30"), Decimal("5")), (Decimal("0.25"), Decimal("10"))),
        asks=((Decimal("0.32"), Decimal("5")), (Decimal("0.35"), Decimal("10"))),
    )

    proceeds, sold = item.estimate_sell(Decimal("12"))
    bought, spent = item.estimate_buy(Decimal("3.35"))

    assert sold == Decimal("12")
    assert proceeds == Decimal("3.25")
    assert bought == Decimal("10")
    assert spent == Decimal("3.35")
