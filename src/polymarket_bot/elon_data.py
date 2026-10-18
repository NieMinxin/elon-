from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from polymarket_bot.config import ConfigError
from polymarket_bot.database import MySQLConfig, PostRepository
from polymarket_bot.market_intervals import actual_interval_index, load_market_intervals
from polymarket_bot.xtracker import XTrackerClient, parse_datetime

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Store xTracker posts in local MySQL")
    parser.add_argument("--handle", default="elonmusk", help="xTracker username")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create the database and tables")
    backfill = subparsers.add_parser("backfill", help="Import a historical range")
    backfill.add_argument("--days", type=int, default=365, help="Days before --end")
    backfill.add_argument("--start", help="Inclusive ISO-8601 start time")
    backfill.add_argument("--end", help="Exclusive ISO-8601 end time; defaults to now")
    backfill.add_argument("--chunk-days", type=int, default=31, help="Days per API request")

    sync = subparsers.add_parser("sync", help="Import posts since the latest stored post")
    sync.add_argument("--start", help="Inclusive ISO-8601 start; defaults to DB maximum date")
    sync.add_argument("--overlap-hours", type=int, default=0)
    sync.add_argument("--initial-days", type=int, default=365)
    sync.add_argument("--chunk-days", type=int, default=31)

    reconcile = subparsers.add_parser(
        "reconcile", help="Compare official tracking, posts endpoint, and MySQL counts"
    )
    reconcile.add_argument("--start", required=True, help="Tracking period start")
    reconcile.add_argument("--end", required=True, help="Tracking period end")
    reconcile.add_argument("--tracking-id", help="Defaults to the closest xTracker period")

    finalize = subparsers.add_parser(
        "finalize-event", help="Store an ended event's official count and winning interval label"
    )
    finalize.add_argument("--start", required=True, help="Tracking period start")
    finalize.add_argument("--end", required=True, help="Tracking period end")
    finalize.add_argument("--tracking-id", help="Defaults to the closest xTracker period")
    finalize.add_argument("--interval-config", type=Path, required=True)

    subparsers.add_parser("status", help="Show local data coverage and last sync")
    return parser


async def sync_range(
    repository: PostRepository,
    *,
    handle: str,
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> int:
    if start >= end:
        raise ConfigError("Sync start must be before end")
    if not 1 <= chunk_days <= 90:
        raise ConfigError("--chunk-days must be between 1 and 90")
    total = 0
    async with httpx.AsyncClient(timeout=60) as http_client:
        xtracker = XTrackerClient(http_client)
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + timedelta(days=chunk_days))
            started_at = datetime.now(UTC)
            try:
                posts = await xtracker.get_posts(handle, start=cursor, end=chunk_end)
                stored = repository.upsert_posts(handle, posts)
                repository.record_sync_run(
                    handle=handle,
                    start=cursor,
                    end=chunk_end,
                    started_at=started_at,
                    status="success",
                    fetched_count=len(posts),
                )
            except Exception as error:
                repository.record_sync_run(
                    handle=handle,
                    start=cursor,
                    end=chunk_end,
                    started_at=started_at,
                    status="failed",
                    fetched_count=0,
                    error_message=str(error)[:2000],
                )
                raise
            total += stored
            LOGGER.info(
                "Stored %s posts for %s to %s",
                stored,
                cursor.isoformat(),
                chunk_end.isoformat(),
            )
            cursor = chunk_end
    return total


async def reconcile_range(
    repository: PostRepository,
    *,
    handle: str,
    start: datetime,
    end: datetime,
    tracking_id: str | None,
) -> dict[str, object]:
    async with httpx.AsyncClient(timeout=60) as http_client:
        xtracker = XTrackerClient(http_client)
        resolved_tracking_id = tracking_id or await xtracker.find_tracking(
            handle,
            start=start,
            end=end,
            active_only=False,
        )
        tracking = await xtracker.get_tracking(resolved_tracking_id)
        posts = await xtracker.get_posts(handle, start=start, end=end)
    endpoint_ids = {item.platform_id for item in posts}
    mysql_ids_before = repository.post_ids(handle, start=start, end=end)
    suspected_deleted = sorted(mysql_ids_before - endpoint_ids)
    missing_from_mysql = sorted(endpoint_ids - mysql_ids_before)
    repository.upsert_posts(handle, posts)
    mysql_ids_after = repository.post_ids(handle, start=start, end=end)
    payload: dict[str, object] = {
        "handle": handle,
        "tracking_id": resolved_tracking_id,
        "range_start": start,
        "range_end": end,
        "observed_at": datetime.now(UTC),
        "official_total": tracking.total,
        "endpoint_total": len(endpoint_ids),
        "mysql_total": len(mysql_ids_after),
        "tracking_posts_delta": tracking.total - len(endpoint_ids),
        "endpoint_mysql_delta": len(endpoint_ids) - len(mysql_ids_after),
        "suspected_deleted_count": len(suspected_deleted),
        "missing_from_mysql_count": len(missing_from_mysql),
        "suspected_deleted_ids": suspected_deleted,
        "missing_from_mysql_ids": missing_from_mysql,
    }
    repository.record_reconciliation(payload)
    return payload


async def finalize_event(
    repository: PostRepository,
    *,
    handle: str,
    start: datetime,
    end: datetime,
    tracking_id: str | None,
    interval_config: Path,
) -> tuple[str, dict[str, object]]:
    if datetime.now(UTC) < end:
        raise ConfigError("Cannot finalize an event before its end time")
    event_slug, intervals = load_market_intervals(interval_config)
    async with httpx.AsyncClient(timeout=60) as http_client:
        xtracker = XTrackerClient(http_client)
        resolved_tracking_id = tracking_id or await xtracker.find_tracking(
            handle,
            start=start,
            end=end,
            active_only=False,
        )
        tracking = await xtracker.get_tracking(resolved_tracking_id)
        posts = await xtracker.get_posts(handle, start=start, end=end)
    endpoint_ids = {item.platform_id for item in posts}
    repository.upsert_posts(handle, posts)
    mysql_ids = repository.post_ids(handle, start=start, end=end)
    interval = intervals[actual_interval_index(intervals, tracking.total)]
    payload: dict[str, object] = {
        "event_slug": event_slug,
        "tracking_id": resolved_tracking_id,
        "handle": handle,
        "event_start": start,
        "event_end": end,
        "final_count": tracking.total,
        "winning_interval_label": interval.label,
        "winning_lower": interval.lower,
        "winning_upper": interval.upper,
        "label_source": "xtracker_tracking",
        "endpoint_total": len(endpoint_ids),
        "mysql_total": len(mysql_ids),
        "tracking_posts_delta": tracking.total - len(endpoint_ids),
        "endpoint_mysql_delta": len(endpoint_ids) - len(mysql_ids),
        "observed_at": datetime.now(UTC),
    }
    return repository.record_event_outcome(payload), payload


def _show_status(repository: PostRepository, handle: str) -> None:
    stats = repository.stats(handle)
    LOGGER.info(
        "MySQL posts=%s with_content=%s earliest=%s latest=%s",
        stats["count"],
        stats["with_content"],
        stats["earliest"],
        stats["latest"],
    )
    if stats["last_sync"]:
        status, started, finished, fetched, error = stats["last_sync"]
        LOGGER.info(
            "Last sync status=%s started=%s finished=%s fetched=%s error=%s",
            status,
            started,
            finished,
            fetched,
            error or "none",
        )


async def run(args: argparse.Namespace) -> None:
    repository = PostRepository(MySQLConfig.from_env())
    repository.initialize()
    if args.command == "init-db":
        LOGGER.info("MySQL database and tables are ready")
        return
    if args.command == "status":
        _show_status(repository, args.handle)
        return

    if args.command == "reconcile":
        result = await reconcile_range(
            repository,
            handle=args.handle,
            start=parse_datetime(args.start),
            end=parse_datetime(args.end),
            tracking_id=args.tracking_id,
        )
        LOGGER.info(
            "Reconciliation tracking=%s official=%s endpoint=%s mysql=%s "
            "tracking_delta=%s endpoint_delta=%s suspected_deleted=%s newly_stored=%s",
            result["tracking_id"],
            result["official_total"],
            result["endpoint_total"],
            result["mysql_total"],
            result["tracking_posts_delta"],
            result["endpoint_mysql_delta"],
            result["suspected_deleted_count"],
            result["missing_from_mysql_count"],
        )
        return

    if args.command == "finalize-event":
        status, result = await finalize_event(
            repository,
            handle=args.handle,
            start=parse_datetime(args.start),
            end=parse_datetime(args.end),
            tracking_id=args.tracking_id,
            interval_config=args.interval_config,
        )
        LOGGER.info(
            "Event label %s: event=%s final=%s interval=%s endpoint=%s mysql=%s",
            status,
            result["event_slug"],
            result["final_count"],
            result["winning_interval_label"],
            result["endpoint_total"],
            result["mysql_total"],
        )
        return

    now = datetime.now(UTC)
    if args.command == "backfill":
        end = parse_datetime(args.end) if args.end else now
        start = parse_datetime(args.start) if args.start else end - timedelta(days=args.days)
    else:
        if args.overlap_hours < 0:
            raise ConfigError("--overlap-hours cannot be negative")
        latest = repository.latest_created_at(args.handle)
        if args.start:
            start = parse_datetime(args.start)
            source = "command line"
        elif latest:
            start = latest - timedelta(hours=args.overlap_hours)
            source = "database maximum created_at"
        else:
            start = now - timedelta(days=args.initial_days)
            source = "empty-database fallback"
        end = now
        LOGGER.info("Incremental range start=%s (%s), end=%s", start, source, end)
    total = await sync_range(
        repository,
        handle=args.handle,
        start=start,
        end=end,
        chunk_days=args.chunk_days,
    )
    LOGGER.info("Sync completed: %s fetched rows were upserted", total)
    _show_status(repository, args.handle)


def cli() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        asyncio.run(run(args))
    except ConfigError as error:
        logging.error("Configuration error: %s", error)
        raise SystemExit(2) from error
    except KeyboardInterrupt:
        logging.info("Stopped")


if __name__ == "__main__":
    cli()
