from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarketInterval:
    label: str
    lower: int
    upper: int | None

    def contains(self, value: int) -> bool:
        return value >= self.lower and (self.upper is None or value <= self.upper)


def load_market_intervals(path: Path) -> tuple[str, tuple[MarketInterval, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    intervals = tuple(
        MarketInterval(
            label=str(item["label"]),
            lower=int(item["lower"]),
            upper=int(item["upper"]) if item.get("upper") is not None else None,
        )
        for item in payload["intervals"]
    )
    validate_market_intervals(intervals)
    return str(payload["event_slug"]), intervals


def validate_market_intervals(intervals: tuple[MarketInterval, ...]) -> None:
    if not intervals:
        raise ValueError("At least one market interval is required")
    if intervals[0].lower != 0:
        raise ValueError("Market intervals must start at zero")
    for left, right in zip(intervals, intervals[1:], strict=False):
        if left.upper is None or left.upper + 1 != right.lower:
            raise ValueError("Market intervals must be contiguous and ordered")
    if intervals[-1].upper is not None:
        raise ValueError("The final market interval must be open ended")


def actual_interval_index(intervals: tuple[MarketInterval, ...], value: int) -> int:
    for index, interval in enumerate(intervals):
        if interval.contains(value):
            return index
    raise ValueError(f"Value {value} is outside the configured market intervals")
