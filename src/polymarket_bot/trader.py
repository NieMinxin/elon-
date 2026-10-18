from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

import httpx

from polymarket_bot.config import BotConfig
from polymarket_bot.state import BotState
from polymarket_bot.strategy import (
    ActionKind,
    Decision,
    buy_price_cap,
    decide,
    sell_price_floor,
)

LOGGER = logging.getLogger(__name__)


class TradingClient(Protocol):
    async def get_market(self, *, slug: str) -> Any: ...

    async def get_price(self, *, asset_id: str, side: str) -> Decimal: ...

    def list_positions(self, **kwargs: Any) -> Any: ...

    async def place_market_order(self, **kwargs: Any) -> Any: ...

    async def wait_for_order_fill_settlement(self, order: Any) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class MarketContext:
    asset_id: str
    condition_id: str
    question: str
    minimum_order_size: Decimal


async def check_geoblock() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get("https://polymarket.com/api/geoblock")
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict) or "blocked" not in payload:
        raise RuntimeError("Unexpected response from Polymarket geoblock endpoint")
    return payload


class TradingBot:
    def __init__(self, client: TradingClient, config: BotConfig, *, live: bool) -> None:
        self.client = client
        self.config = config
        self.live = live
        self.state = BotState.load(config.state_path)
        if not live:
            self.state.select_paper_market(config.market_slug, config.outcome)
        self.market: MarketContext | None = None

    async def initialize(self) -> MarketContext:
        market = await self.client.get_market(slug=self.config.market_slug)
        if market.state.closed or market.state.archived:
            raise RuntimeError("Market is closed or archived")
        if market.state.accepting_orders is False or market.state.enable_order_book is False:
            raise RuntimeError("Market is not accepting CLOB orders")
        outcome = market.outcomes.yes if self.config.outcome == "YES" else market.outcomes.no
        asset = outcome.position_id if market.version == "v2" else outcome.token_id
        if asset is None:
            raise RuntimeError(f"{self.config.outcome} outcome has no tradable asset ID")
        if market.condition_id is None:
            raise RuntimeError("Market has no condition ID")
        self.market = MarketContext(
            asset_id=str(asset),
            condition_id=str(market.condition_id),
            question=market.question or self.config.market_slug,
            minimum_order_size=market.trading.minimum_order_size or Decimal("0"),
        )
        return self.market

    async def _live_position_shares(self) -> Decimal:
        assert self.market is not None
        page = await self.client.list_positions(
            condition_id=self.market.condition_id, page_size=100
        ).first_page()
        position = next(
            (item for item in page.items if str(item.asset_id) == self.market.asset_id),
            None,
        )
        return Decimal("0") if position is None else position.current_size

    async def position_shares(self) -> Decimal:
        if self.live:
            return await self._live_position_shares()
        return self.state.paper_shares

    async def evaluate(self, now: datetime | None = None) -> tuple[Decision, Decimal, Decimal]:
        if self.market is None:
            await self.initialize()
        assert self.market is not None
        current_time = now or datetime.now(UTC)
        self.state.roll_day(current_time.date())
        # CLOB side names describe resting orders: BUY is the best bid and
        # SELL is the best ask. A taker buys at SELL and sells at BUY.
        buy_price = await self.client.get_price(asset_id=self.market.asset_id, side="SELL")
        sell_price = await self.client.get_price(asset_id=self.market.asset_id, side="BUY")
        shares = await self.position_shares()
        decision = decide(
            config=self.config,
            buy_price=buy_price,
            sell_price=sell_price,
            position_shares=shares,
            daily_buy_usd=self.state.daily_buy_usd,
            last_trade_at=self.state.last_trade_at,
            now=current_time,
        )
        return decision, buy_price, sell_price

    async def run_once(self, now: datetime | None = None) -> Decision:
        current_time = now or datetime.now(UTC)
        decision, buy_price, sell_price = await self.evaluate(current_time)
        assert self.market is not None
        LOGGER.info(
            "%s | %s %s | buy=%s sell=%s | %s",
            "LIVE" if self.live else "PAPER",
            self.config.outcome,
            self.market.question,
            buy_price,
            sell_price,
            decision.reason,
        )
        if decision.action is ActionKind.HOLD:
            self.state.save(self.config.state_path)
            return decision
        if self.live:
            await self._execute_live(decision, buy_price, sell_price, current_time)
        else:
            self._execute_paper(decision, buy_price, sell_price, current_time)
        self.state.save(self.config.state_path)
        return decision

    def _execute_paper(
        self,
        decision: Decision,
        buy_price: Decimal,
        sell_price: Decimal,
        now: datetime,
    ) -> None:
        if decision.action is ActionKind.BUY:
            shares = decision.amount_usd / buy_price
            self.state.record_buy(decision.amount_usd, shares, now)
            LOGGER.warning(
                "PAPER BUY: spend=%s estimated_shares=%s price=%s",
                decision.amount_usd,
                shares,
                buy_price,
            )
        elif decision.action is ActionKind.SELL:
            proceeds = decision.shares * sell_price
            self.state.record_sell(decision.shares, proceeds, now)
            LOGGER.warning(
                "PAPER SELL: shares=%s estimated_proceeds=%s price=%s",
                decision.shares,
                proceeds,
                sell_price,
            )

    async def _execute_live(
        self,
        decision: Decision,
        buy_price: Decimal,
        sell_price: Decimal,
        now: datetime,
    ) -> None:
        assert self.market is not None
        geo = await check_geoblock()
        if geo.get("blocked"):
            country = geo.get("country", "unknown")
            region = geo.get("region", "")
            raise RuntimeError(f"Polymarket trading is blocked for {country}-{region}")

        if decision.action is ActionKind.BUY:
            estimated_shares = decision.amount_usd / buy_price
            if estimated_shares < self.market.minimum_order_size:
                LOGGER.warning(
                    "Skipping buy: estimated %s shares is below market minimum %s",
                    estimated_shares,
                    self.market.minimum_order_size,
                )
                return
            response = await self.client.place_market_order(
                asset_id=self.market.asset_id,
                side="BUY",
                amount=decision.amount_usd,
                max_spend=decision.amount_usd,
                max_price=buy_price_cap(buy_price, self.config.max_slippage),
                order_type="FAK",
            )
            if not response.ok:
                raise RuntimeError(f"Buy rejected ({response.code}): {response.message}")
            self.state.daily_buy_usd += response.making_amount
        else:
            response = await self.client.place_market_order(
                asset_id=self.market.asset_id,
                side="SELL",
                shares=decision.shares,
                min_price=sell_price_floor(sell_price, self.config.max_slippage),
                order_type="FAK",
            )
            if not response.ok:
                raise RuntimeError(f"Sell rejected ({response.code}): {response.message}")

        self.state.last_trade_at = now
        # Persist acceptance before waiting for asynchronous settlement. This
        # prevents a restart during that wait from immediately repeating an order.
        self.state.save(self.config.state_path)
        LOGGER.warning(
            "LIVE %s accepted: order_id=%s status=%s",
            decision.action.value,
            response.order_id,
            response.status,
        )
        if response.trade_ids:
            try:
                hashes = await self.client.wait_for_order_fill_settlement(response)
                LOGGER.info("Settlement confirmed: %s", ", ".join(map(str, hashes)))
            except TimeoutError:
                LOGGER.warning("Settlement is still pending; the accepted order was not canceled")
