from datetime import UTC, date, datetime
from decimal import Decimal

from polymarket_bot.state import BotState


def test_state_round_trip(tmp_path) -> None:
    path = tmp_path / "state.json"
    today = datetime.now(UTC).date()
    state = BotState.empty(today)
    state.select_paper_market("example-market", "YES")
    state.record_buy(Decimal("10"), Decimal("25"), datetime.now(UTC))
    state.save(path)

    loaded = BotState.load(path)

    assert loaded.daily_buy_usd == Decimal("10")
    assert loaded.paper_market_key == "example-market:YES"
    assert loaded.paper_shares == Decimal("25")
    assert loaded.paper_cost_usd == Decimal("10")


def test_partial_sell_reduces_cost_proportionally() -> None:
    state = BotState.empty(date(2026, 10, 17))
    now = datetime(2026, 10, 17, tzinfo=UTC)
    state.record_buy(Decimal("10"), Decimal("20"), now)
    state.record_sell(Decimal("5"), Decimal("3"), now)

    assert state.paper_shares == Decimal("15")
    assert state.paper_cost_usd == Decimal("7.5")


def test_changing_paper_market_resets_only_paper_position() -> None:
    state = BotState.empty(date(2026, 10, 17))
    now = datetime(2026, 10, 17, tzinfo=UTC)
    state.select_paper_market("first", "YES")
    state.record_buy(Decimal("10"), Decimal("20"), now)

    state.select_paper_market("second", "NO")

    assert state.paper_market_key == "second:NO"
    assert state.paper_shares == 0
    assert state.paper_cost_usd == 0
    assert state.daily_buy_usd == Decimal("10")
