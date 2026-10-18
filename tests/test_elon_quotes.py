import asyncio
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from polymarket_bot.elon_config import ElonArbConfig
from polymarket_bot.elon_trader import ElonArbSystem, EventMarket
from polymarket_bot.forecast import CountForecast


class FakePriceClient:
    def __init__(self):
        self.sides = []

    async def get_price(self, *, asset_id, side):
        self.sides.append(side)
        return Decimal("0.30") if side == "BUY" else Decimal("0.34")


class FakeBookClient:
    async def get_order_books(self, *, asset_ids):
        assert asset_ids == ["asset"]

        def level(price, size):
            return SimpleNamespace(price=Decimal(price), size=Decimal(size))

        return (
            SimpleNamespace(
                asset_id="asset",
                bids=(level("0.20", "100"), level("0.30", "5")),
                asks=(level("0.45", "100"), level("0.34", "6")),
            ),
        )


def test_quote_maps_resting_buy_to_bid_and_sell_to_ask(tmp_path: Path) -> None:
    config = ElonArbConfig.from_env(
        {
            "ELON_STATE_PATH": str(tmp_path / "state.json"),
            "ELON_AUDIT_PATH": str(tmp_path / "audit.jsonl"),
            "ELON_HISTORY_CACHE_PATH": str(tmp_path / "history.json"),
        }
    )
    client = FakePriceClient()
    system = ElonArbSystem(client, object(), config, live=False)  # type: ignore[arg-type]
    market = EventMarket(
        slug="market",
        asset_id="asset",
        condition_id="condition",
        interval_label="300-319",
        minimum_order_size=Decimal("5"),
        fee_rate=Decimal("0.05"),
    )
    forecast = CountForecast(250, 310, 10, 10, 10, 10)

    result = asyncio.run(system._quote_market(market, forecast))

    assert result is not None
    assert result.bid == Decimal("0.30")
    assert result.ask == Decimal("0.34")
    assert client.sides == ["BUY", "SELL"]


def test_bulk_quotes_preserve_best_first_order_book_depth(tmp_path: Path) -> None:
    config = ElonArbConfig.from_env(
        {
            "ELON_STATE_PATH": str(tmp_path / "state.json"),
            "ELON_AUDIT_PATH": str(tmp_path / "audit.jsonl"),
        }
    )
    system = ElonArbSystem(FakeBookClient(), object(), config, live=False)  # type: ignore[arg-type]
    system.markets = (
        EventMarket(
            slug="market",
            asset_id="asset",
            condition_id="condition",
            interval_label="300-319",
            minimum_order_size=Decimal("5"),
            fee_rate=Decimal("0.05"),
        ),
    )
    forecast = CountForecast(250, 310, 10, 10, 10, 10)

    quotes = asyncio.run(system._quote_markets(forecast))

    assert quotes[0].bid == Decimal("0.30")
    assert quotes[0].ask == Decimal("0.34")
    assert quotes[0].bids[0] == (Decimal("0.30"), Decimal("5"))
    assert quotes[0].asks[0] == (Decimal("0.34"), Decimal("6"))
