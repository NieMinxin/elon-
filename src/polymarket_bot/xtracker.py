from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TrackingSnapshot:
    tracking_id: str
    title: str
    start: datetime
    end: datetime
    total: int
    fetched_at: datetime
    latest_bucket_at: datetime | None


@dataclass(frozen=True)
class XTrackerPost:
    tracker_id: str
    user_id: str
    platform_id: str
    content: str | None
    created_at: datetime
    imported_at: datetime | None
    metrics: Any
    raw_payload: dict[str, Any]


class XTrackerClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = "https://xtracker.polymarket.com/api",
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        response = await self.client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise RuntimeError(f"Unexpected xtracker response for {path}")
        return payload.get("data")

    async def find_tracking(
        self,
        handle: str,
        *,
        start: datetime,
        end: datetime,
        active_only: bool = True,
    ) -> str:
        items = await self._get(
            f"/users/{handle}/trackings",
            {"platform": "X", "activeOnly": "true" if active_only else "false"},
        )
        if not isinstance(items, list):
            raise RuntimeError("xtracker tracking list has an unexpected shape")
        best: tuple[float, str] | None = None
        for item in items:
            item_start = parse_datetime(str(item["startDate"]))
            item_end = parse_datetime(str(item["endDate"]))
            distance = abs((item_start - start).total_seconds()) + abs(
                (item_end - end).total_seconds()
            )
            candidate = (distance, str(item["id"]))
            if best is None or candidate < best:
                best = candidate
        if best is None or best[0] > 300:
            raise RuntimeError("No xtracker period matches the Polymarket event window")
        return best[1]

    async def get_tracking(self, tracking_id: str) -> TrackingSnapshot:
        item = await self._get(f"/trackings/{tracking_id}", {"includeStats": "true"})
        if not isinstance(item, dict) or not isinstance(item.get("stats"), dict):
            raise RuntimeError("xtracker tracking response has an unexpected shape")
        buckets = item["stats"].get("daily") or []
        latest = max(
            (parse_datetime(str(bucket["date"])) for bucket in buckets),
            default=None,
        )
        return TrackingSnapshot(
            tracking_id=str(item["id"]),
            title=str(item["title"]),
            start=parse_datetime(str(item["startDate"])),
            end=parse_datetime(str(item["endDate"])),
            total=int(item["stats"]["total"]),
            fetched_at=datetime.now(UTC),
            latest_bucket_at=latest,
        )

    async def get_posts(
        self,
        handle: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[XTrackerPost]:
        items = await self._get(
            f"/users/{handle}/posts",
            {
                "platform": "X",
                "startDate": isoformat(start),
                "endDate": isoformat(end),
            },
        )
        if not isinstance(items, list):
            raise RuntimeError("xtracker posts response has an unexpected shape")
        posts = []
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError("xtracker post has an unexpected shape")
            imported_at = item.get("importedAt")
            posts.append(
                XTrackerPost(
                    tracker_id=str(item["id"]),
                    user_id=str(item["userId"]),
                    platform_id=str(item["platformId"]),
                    content=str(item["content"]) if item.get("content") is not None else None,
                    created_at=parse_datetime(str(item["createdAt"])),
                    imported_at=parse_datetime(str(imported_at)) if imported_at else None,
                    metrics=item.get("metrics"),
                    raw_payload=dict(item),
                )
            )
        return sorted(posts, key=lambda item: item.created_at)

    async def get_post_times(
        self,
        handle: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[datetime]:
        return [item.created_at for item in await self.get_posts(handle, start=start, end=end)]


def load_history_cache(path: Path, *, start: datetime, end: datetime) -> list[datetime] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("start") != isoformat(start) or payload.get("end") != isoformat(end):
            return None
        return [parse_datetime(item) for item in payload["post_times"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_history_cache(
    path: Path, *, start: datetime, end: datetime, post_times: list[datetime]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "start": isoformat(start),
        "end": isoformat(end),
        "fetched_at": isoformat(datetime.now(UTC)),
        "post_times": [isoformat(item) for item in post_times],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
