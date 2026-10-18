from decimal import Decimal

from polymarket_bot.elon_monitor import assess_portfolio, select_best_band
from polymarket_bot.elon_strategy import Interval, MarketQuote, PositionView


def quote(lower: int, probability: str, *, bid: str = "0.20", ask: str = "0.22") -> MarketQuote:
    upper = lower + 19
    label = f"{lower}-{upper}"
    return MarketQuote(
        slug=label,
        asset_id=label,
        condition_id=label,
        interval=Interval(label, lower, upper),
        bid=Decimal(bid),
        ask=Decimal(ask),
        fee_rate=Decimal("0.05"),
        minimum_order_size=Decimal("1"),
        fair_probability=Decimal(probability),
    )


def test_selects_highest_probability_contiguous_three() -> None:
    quotes = [
        quote(300, "0.05"),
        quote(320, "0.15"),
        quote(340, "0.25"),
        quote(360, "0.20"),
        quote(380, "0.10"),
    ]

    band = select_best_band(quotes)

    assert band.labels == ("320-339", "340-359", "360-379")
    assert band.probability == Decimal("0.60")


def test_assessment_reports_two_leg_roll_and_transaction_costs() -> None:
    quotes = [
        quote(300, "0.02", bid="0.10", ask="0.12"),
        quote(320, "0.03", bid="0.10", ask="0.12"),
        quote(340, "0.10", bid="0.16", ask="0.18"),
        quote(360, "0.20", bid="0.18", ask="0.20"),
        quote(380, "0.28", bid="0.20", ask="0.22"),
        quote(400, "0.24", bid="0.17", ask="0.19"),
    ]
    positions = {
        "320-339": PositionView(Decimal("100"), Decimal("12")),
        "340-359": PositionView(Decimal("80"), Decimal("12")),
        "360-379": PositionView(Decimal("60"), Decimal("12")),
    }

    result = assess_portfolio(quotes, positions)

    assert result.target_band.labels == ("360-379", "380-399", "400-419")
    assert result.shift_steps == 2
    assert result.roll_count == 2
    assert result.sell_labels == ("320-339", "340-359")
    assert result.buy_labels == ("380-399", "400-419")
    assert result.rotation_proceeds_usd > 0
    assert result.rotation_value_gain_pct is not None


def test_assessment_does_not_treat_missing_quotes_as_valid_pnl() -> None:
    quotes = [quote(320, "0.2"), quote(340, "0.3"), quote(360, "0.2")]
    positions = {"unknown": PositionView(Decimal("10"), Decimal("2"))}

    result = assess_portfolio(quotes, positions)

    assert result.quotes_complete is False
    assert result.unrealized_return is None
