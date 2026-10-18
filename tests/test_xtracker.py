import asyncio
from datetime import UTC, datetime

import httpx

from polymarket_bot.xtracker import XTrackerClient


def test_get_posts_preserves_content_and_raw_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["platform"] == "X"
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": [
                    {
                        "id": "tracker-1",
                        "userId": "user-1",
                        "platformId": "123456",
                        "content": "Tesla x SpaceX",
                        "createdAt": "2026-10-16T00:44:34Z",
                        "importedAt": "2026-10-16T00:45:30.646Z",
                        "metrics": None,
                    }
                ],
            },
        )

    async def get_posts():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await XTrackerClient(client).get_posts(
                "elonmusk",
                start=datetime(2026, 10, 16, tzinfo=UTC),
                end=datetime(2026, 10, 17, tzinfo=UTC),
            )

    posts = asyncio.run(get_posts())

    assert posts[0].platform_id == "123456"
    assert posts[0].content == "Tesla x SpaceX"
    assert posts[0].imported_at == datetime(2026, 10, 16, 0, 45, 30, 646000, tzinfo=UTC)
    assert posts[0].raw_payload["id"] == "tracker-1"


def test_find_tracking_can_include_inactive_periods() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["activeOnly"] == "false"
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": [
                    {
                        "id": "tracking-1",
                        "startDate": "2026-10-13T16:00:00Z",
                        "endDate": "2026-10-20T16:00:00Z",
                    }
                ],
            },
        )

    async def find_tracking() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await XTrackerClient(client).find_tracking(
                "elonmusk",
                start=datetime(2026, 10, 13, 16, tzinfo=UTC),
                end=datetime(2026, 10, 20, 16, tzinfo=UTC),
                active_only=False,
            )

    assert asyncio.run(find_tracking()) == "tracking-1"
