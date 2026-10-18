from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from polymarket_bot.elon_config import ElonArbConfig


class ArbActionKind(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Interval:
    label: str
    lower: int
    upper: int | None


@dataclass(frozen=True)
class MarketQuote:
    slug: str
    asset_id: str
    condition_id: str
    interval: Interval
    bid: Decimal
    ask: Decimal
    fee_rate: Decimal
    minimum_order_size: Decimal
    fair_probability: Decimal
    bids: tuple[tuple[Decimal, Decimal], ...] = ()
    asks: tuple[tuple[Decimal, Decimal], ...] = ()
    has_full_book: bool = False

    @property
    def entry_fee_per_share(self) -> Decimal:
        return taker_fee_per_share(self.ask, self.fee_rate)

    @property
    def exit_fee_per_share(self) -> Decimal:
        return taker_fee_per_share(self.bid, self.fee_rate)

    @property
    def all_in_ask(self) -> Decimal:
        return self.ask + self.entry_fee_per_share

    @property
    def net_bid(self) -> Decimal:
        return max(Decimal("0"), self.bid - self.exit_fee_per_share)

    @property
    def entry_edge(self) -> Decimal:
        return self.fair_probability - self.all_in_ask

    @property
    def exit_advantage(self) -> Decimal:
        return self.net_bid - self.fair_probability

    def estimate_buy(self, amount_usd: Decimal) -> tuple[Decimal, Decimal]:
        """Return (shares, spend) using ask depth, including taker fees."""

        if amount_usd <= 0:
            return Decimal("0"), Decimal("0")
        if self.has_full_book and not self.asks:
            return Decimal("0"), Decimal("0")
        levels = self.asks or ((self.ask, Decimal("Infinity")),)
        shares = Decimal("0")
        spent = Decimal("0")
        remaining = amount_usd
        for price, size in levels:
            all_in_price = price + taker_fee_per_share(price, self.fee_rate)
            fill = min(size, remaining / all_in_price)
            shares += fill
            level_spend = fill * all_in_price
            spent += level_spend
            remaining -= level_spend
            if remaining <= Decimal("0.00000001"):
                break
        return shares, spent

    def estimate_sell(self, shares: Decimal) -> tuple[Decimal, Decimal]:
        """Return (net proceeds, filled shares) using bid depth and taker fees."""

        if shares <= 0:
            return Decimal("0"), Decimal("0")
        if self.has_full_book and not self.bids:
            return Decimal("0"), Decimal("0")
        levels = self.bids or ((self.bid, Decimal("Infinity")),)
        proceeds = Decimal("0")
        filled = Decimal("0")
        remaining = shares
        for price, size in levels:
            fill = min(size, remaining)
            proceeds += fill * max(
                Decimal("0"),
                price - taker_fee_per_share(price, self.fee_rate),
            )
            filled += fill
            remaining -= fill
            if remaining <= Decimal("0.00000001"):
                break
        return proceeds, filled


@dataclass(frozen=True)
class PositionView:
    shares: Decimal
    cost_usd: Decimal


@dataclass(frozen=True)
class ArbDecision:
    action: ArbActionKind
    reason: str
    quote: MarketQuote | None = None
    amount_usd: Decimal = Decimal("0")
    shares: Decimal = Decimal("0")


def parse_interval(label: str) -> Interval:
    normalized = label.strip().replace("–", "-")
    match = re.fullmatch(r"<?\s*(\d+)\s*-\s*(\d+)", normalized)
    if match:
        return Interval(normalized, int(match.group(1)), int(match.group(2)))
    match = re.fullmatch(r"<\s*(\d+)", normalized)
    if match:
        return Interval(normalized, 0, int(match.group(1)) - 1)
    match = re.fullmatch(r"(\d+)\s*\+", normalized)
    if match:
        return Interval(normalized, int(match.group(1)), None)
    raise ValueError(f"Cannot parse market interval {label!r}")


def taker_fee_per_share(price: Decimal, fee_rate: Decimal) -> Decimal:
    return fee_rate * price * (Decimal("1") - price)


def decide_action(
    *,
    config: ElonArbConfig,
    quotes: list[MarketQuote],
    positions: dict[str, PositionView],
    daily_buy_usd: Decimal,
    minutes_remaining: Decimal,
) -> ArbDecision:
    for quote in quotes:
        position = positions.get(quote.asset_id)
        if position is None or position.shares <= 0:
            continue
        impossible = quote.interval.upper is not None and quote.fair_probability == 0
        if impossible or quote.exit_advantage >= config.min_exit_advantage:
            reason = (
                "interval is no longer reachable"
                if impossible
                else f"net bid exceeds model value by {quote.exit_advantage:.4f}"
            )
            return ArbDecision(
                ArbActionKind.SELL,
                reason,
                quote=quote,
                shares=position.shares,
            )

    if minutes_remaining <= config.no_new_position_minutes:
        return ArbDecision(ArbActionKind.HOLD, "new-position cutoff is active")
    if len(positions) >= config.max_positions:
        return ArbDecision(ArbActionKind.HOLD, "position-count limit reached")
    total_cost = sum((item.cost_usd for item in positions.values()), Decimal("0"))
    remaining_total = config.max_total_cost_usd - total_cost
    remaining_daily = config.max_daily_buy_usd - daily_buy_usd
    capacity = min(config.order_amount_usd, remaining_total, remaining_daily)
    if capacity <= 0:
        return ArbDecision(ArbActionKind.HOLD, "capital limit reached")

    candidates = [
        quote
        for quote in quotes
        if quote.asset_id not in positions
        and quote.fair_probability >= config.min_fair_probability
        and quote.entry_edge >= config.min_entry_edge
        and capacity / quote.all_in_ask >= quote.minimum_order_size
    ]
    if not candidates:
        return ArbDecision(ArbActionKind.HOLD, "no fee-adjusted edge meets entry threshold")
    selected = max(candidates, key=lambda quote: quote.entry_edge)
    return ArbDecision(
        ArbActionKind.BUY,
        f"model edge {selected.entry_edge:.4f} after taker fee",
        quote=selected,
        amount_usd=capacity,
    )
