import asyncio
from decimal import Decimal
from types import SimpleNamespace

from polymarket_bot.config import BotConfig
from polymarket_bot.trader import TradingBot


class FakePage:
    def __init__(self, items):
        self.items = items


class FakePaginator:
    def __init__(self, items):
        self.items = items

    async def first_page(self):
        return FakePage(self.items)


class FakeClient:
    def __init__(self):
        self.position_kwargs = None
        self.price_sides = []

    async def get_market(self, *, slug):
        return SimpleNamespace(
            state=SimpleNamespace(
                closed=False,
                archived=False,
                accepting_orders=True,
                enable_order_book=True,
            ),
            outcomes=SimpleNamespace(
                yes=SimpleNamespace(token_id="yes-token", position_id=None),
                no=SimpleNamespace(token_id="no-token", position_id=None),
            ),
            version="v1",
            condition_id="condition-1",
            question="Example?",
            trading=SimpleNamespace(minimum_order_size=Decimal("5")),
        )

    def list_positions(self, **kwargs):
        self.position_kwargs = kwargs
        return FakePaginator([SimpleNamespace(asset_id="yes-token", current_size=Decimal("7"))])

    async def get_price(self, *, asset_id, side):
        self.price_sides.append(side)
        return Decimal("0.40") if side == "SELL" else Decimal("0.38")


def test_live_position_uses_current_sdk_condition_id_filter(tmp_path) -> None:
    config = BotConfig.from_env(
        {
            "POLYMARKET_MARKET_SLUG": "example-market",
            "POLYMARKET_BUY_BELOW": "0.40",
            "POLYMARKET_TAKE_PROFIT_AT": "0.60",
            "POLYMARKET_STOP_LOSS_AT": "0.25",
            "POLYMARKET_STATE_PATH": str(tmp_path / "state.json"),
        }
    )
    client = FakeClient()
    bot = TradingBot(client, config, live=True)  # type: ignore[arg-type]

    async def scenario() -> Decimal:
        await bot.initialize()
        return await bot.position_shares()

    shares = asyncio.run(scenario())

    assert shares == Decimal("7")
    assert client.position_kwargs == {"condition_id": "condition-1", "page_size": 100}


def test_taker_buy_uses_ask_and_taker_sell_uses_bid(tmp_path) -> None:
    config = BotConfig.from_env(
        {
            "POLYMARKET_MARKET_SLUG": "example-market",
            "POLYMARKET_BUY_BELOW": "0.41",
            "POLYMARKET_TAKE_PROFIT_AT": "0.60",
            "POLYMARKET_STOP_LOSS_AT": "0.25",
            "POLYMARKET_STATE_PATH": str(tmp_path / "state.json"),
        }
    )
    client = FakeClient()
    bot = TradingBot(client, config, live=False)  # type: ignore[arg-type]

    async def scenario():
        await bot.initialize()
        return await bot.evaluate()

    _, buy_price, sell_price = asyncio.run(scenario())

    assert buy_price == Decimal("0.40")
    assert sell_price == Decimal("0.38")
    assert client.price_sides == ["SELL", "BUY"]
