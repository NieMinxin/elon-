from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from polymarket_bot.elon_strategy import MarketQuote, PositionView


@dataclass(frozen=True)
class IntervalBand:
    quotes: tuple[MarketQuote, ...]
    probability: Decimal

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(item.interval.label for item in self.quotes)

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(item.asset_id for item in self.quotes)


@dataclass(frozen=True)
class PortfolioAssessment:
    target_band: IntervalBand
    held_labels: tuple[str, ...]
    held_probability: Decimal
    probability_gain: Decimal
    shift_steps: int | None
    roll_count: int
    sell_labels: tuple[str, ...]
    buy_labels: tuple[str, ...]
    invested_usd: Decimal
    liquidation_usd: Decimal
    model_value_usd: Decimal
    unrealized_return: Decimal | None
    rotation_proceeds_usd: Decimal
    rotated_model_value_usd: Decimal
    rotation_value_gain: Decimal
    rotation_value_gain_pct: Decimal | None
    quotes_complete: bool
    liquidity_complete: bool


def _are_adjacent(left: MarketQuote, right: MarketQuote) -> bool:
    return left.interval.upper is not None and left.interval.upper + 1 == right.interval.lower


def select_best_band(quotes: list[MarketQuote], *, size: int = 3) -> IntervalBand:
    if size < 1:
        raise ValueError("Band size must be positive")
    ordered = sorted(quotes, key=lambda item: item.interval.lower)
    candidates = []
    for index in range(len(ordered) - size + 1):
        group = tuple(ordered[index : index + size])
        if all(_are_adjacent(group[offset], group[offset + 1]) for offset in range(size - 1)):
            probability = sum(
                (item.fair_probability for item in group),
                Decimal("0"),
            )
            entry_edge = sum((item.entry_edge for item in group), Decimal("0"))
            candidates.append((probability, entry_edge, group))
    if not candidates:
        raise ValueError(f"No contiguous {size}-interval band is available")
    probability, _, group = max(candidates, key=lambda item: (item[0], item[1]))
    return IntervalBand(group, probability)


def _shift_steps(
    ordered_quotes: list[MarketQuote],
    held_assets: tuple[str, ...],
    target_assets: tuple[str, ...],
) -> int | None:
    if len(held_assets) != len(target_assets):
        return None
    indexes = {quote.asset_id: index for index, quote in enumerate(ordered_quotes)}
    if any(asset not in indexes for asset in (*held_assets, *target_assets)):
        return None
    held_indexes = sorted(indexes[asset] for asset in held_assets)
    target_indexes = sorted(indexes[asset] for asset in target_assets)
    if any(right != left + 1 for left, right in zip(held_indexes, held_indexes[1:], strict=False)):
        return None
    return target_indexes[0] - held_indexes[0]


def assess_portfolio(
    quotes: list[MarketQuote],
    positions: dict[str, PositionView],
    *,
    band_size: int = 3,
) -> PortfolioAssessment:
    target = select_best_band(quotes, size=band_size)
    quote_by_asset = {item.asset_id: item for item in quotes}
    held_assets = tuple(
        item.asset_id
        for item in sorted(quotes, key=lambda quote: quote.interval.lower)
        if item.asset_id in positions and positions[item.asset_id].shares > 0
    )
    quotes_complete = all(asset in quote_by_asset for asset in positions)
    held_quotes = [quote_by_asset[asset] for asset in held_assets]
    held_labels = tuple(item.interval.label for item in held_quotes)
    held_probability = sum(
        (item.fair_probability for item in held_quotes),
        Decimal("0"),
    )
    invested = sum((item.cost_usd for item in positions.values()), Decimal("0"))
    liquidation = Decimal("0")
    liquidity_complete = quotes_complete
    for asset, position in positions.items():
        if asset not in quote_by_asset:
            continue
        proceeds, filled = quote_by_asset[asset].estimate_sell(position.shares)
        liquidation += proceeds
        liquidity_complete = liquidity_complete and filled >= position.shares
    model_value = sum(
        (
            positions[asset].shares * quote_by_asset[asset].fair_probability
            for asset in positions
            if asset in quote_by_asset
        ),
        Decimal("0"),
    )
    unrealized_return = (
        liquidation / invested - 1
        if invested > 0 and quotes_complete and liquidity_complete
        else None
    )

    target_assets = target.asset_ids
    sell_assets = tuple(asset for asset in held_assets if asset not in target_assets)
    buy_assets = tuple(asset for asset in target_assets if asset not in held_assets)
    rotation_proceeds = Decimal("0")
    for asset in sell_assets:
        proceeds, filled = quote_by_asset[asset].estimate_sell(positions[asset].shares)
        rotation_proceeds += proceeds
        liquidity_complete = liquidity_complete and filled >= positions[asset].shares
    removed_model_value = sum(
        (positions[asset].shares * quote_by_asset[asset].fair_probability for asset in sell_assets),
        Decimal("0"),
    )
    rotated_model_value = Decimal("0")
    if buy_assets and rotation_proceeds > 0:
        allocation = rotation_proceeds / len(buy_assets)
        for asset in buy_assets:
            shares, spent = quote_by_asset[asset].estimate_buy(allocation)
            rotated_model_value += shares * quote_by_asset[asset].fair_probability
            liquidity_complete = liquidity_complete and spent >= allocation - Decimal("0.00000001")
    rotation_gain = rotated_model_value - removed_model_value
    rotation_gain_pct = rotation_gain / removed_model_value if removed_model_value > 0 else None
    shift = _shift_steps(
        sorted(quotes, key=lambda item: item.interval.lower),
        held_assets,
        target_assets,
    )
    return PortfolioAssessment(
        target_band=target,
        held_labels=held_labels,
        held_probability=held_probability,
        probability_gain=target.probability - held_probability,
        shift_steps=shift,
        roll_count=max(len(sell_assets), len(buy_assets)),
        sell_labels=tuple(quote_by_asset[asset].interval.label for asset in sell_assets),
        buy_labels=tuple(quote_by_asset[asset].interval.label for asset in buy_assets),
        invested_usd=invested,
        liquidation_usd=liquidation,
        model_value_usd=model_value,
        unrealized_return=unrealized_return,
        rotation_proceeds_usd=rotation_proceeds,
        rotated_model_value_usd=rotated_model_value,
        rotation_value_gain=rotation_gain,
        rotation_value_gain_pct=rotation_gain_pct,
        quotes_complete=quotes_complete,
        liquidity_complete=liquidity_complete,
    )
