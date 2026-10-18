from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pymysql
from pymysql.connections import Connection

from polymarket_bot.config import ConfigError
from polymarket_bot.xtracker import XTrackerPost

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MySQLConfig:
        values = os.environ if env is None else env
        try:
            port = int(values.get("MYSQL_PORT", "3306"))
        except ValueError as error:
            raise ConfigError("MYSQL_PORT must be an integer") from error
        config = cls(
            host=values.get("MYSQL_HOST", "127.0.0.1").strip(),
            port=port,
            user=values.get("MYSQL_USER", "root").strip(),
            password=values.get("MYSQL_PASSWORD", ""),
            database=values.get("MYSQL_DATABASE", "polymarket_bot").strip(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.host or not self.user:
            raise ConfigError("MYSQL_HOST and MYSQL_USER are required")
        if not 1 <= self.port <= 65535:
            raise ConfigError("MYSQL_PORT must be between 1 and 65535")
        if not _IDENTIFIER.fullmatch(self.database):
            raise ConfigError("MYSQL_DATABASE may contain only letters, digits, and underscores")


@dataclass(frozen=True)
class PostObservation:
    created_at: datetime
    imported_at: datetime | None


def _mysql_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(UTC).replace(tzinfo=None)


class PostRepository:
    def __init__(self, config: MySQLConfig) -> None:
        self.config = config

    def _connect(self, *, include_database: bool = True, autocommit: bool = False) -> Connection:
        kwargs: dict[str, Any] = {
            "host": self.config.host,
            "port": self.config.port,
            "user": self.config.user,
            "password": self.config.password,
            "charset": "utf8mb4",
            "autocommit": autocommit,
            "connect_timeout": 10,
        }
        if include_database:
            kwargs["database"] = self.config.database
        return pymysql.connect(**kwargs)

    def initialize(self) -> None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        except pymysql.OperationalError as error:
            if error.args[0] != 1049:
                raise
            with (
                self._connect(include_database=False, autocommit=True) as connection,
                connection.cursor() as cursor,
            ):
                cursor.execute(
                    f"CREATE DATABASE `{self.config.database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS xtracker_posts (
                        platform_post_id VARCHAR(32) NOT NULL,
                        xtracker_post_id VARCHAR(64) NOT NULL,
                        xtracker_user_id VARCHAR(64) NOT NULL,
                        handle VARCHAR(64) NOT NULL,
                        content MEDIUMTEXT NULL,
                        created_at DATETIME(6) NOT NULL,
                        imported_at DATETIME(6) NULL,
                        metrics_json JSON NULL,
                        raw_payload JSON NOT NULL,
                        first_seen_at DATETIME(6) NOT NULL,
                        last_seen_at DATETIME(6) NOT NULL,
                        PRIMARY KEY (platform_post_id),
                        UNIQUE KEY uq_xtracker_post_id (xtracker_post_id),
                        KEY ix_handle_created_at (handle, created_at),
                        KEY ix_imported_at (imported_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS xtracker_sync_runs (
                        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                        handle VARCHAR(64) NOT NULL,
                        range_start DATETIME(6) NOT NULL,
                        range_end DATETIME(6) NOT NULL,
                        started_at DATETIME(6) NOT NULL,
                        finished_at DATETIME(6) NOT NULL,
                        status ENUM('success', 'failed') NOT NULL,
                        fetched_count INT UNSIGNED NOT NULL DEFAULT 0,
                        error_message TEXT NULL,
                        PRIMARY KEY (id),
                        KEY ix_sync_handle_started (handle, started_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS xtracker_reconciliation_runs (
                        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                        handle VARCHAR(64) NOT NULL,
                        tracking_id VARCHAR(64) NOT NULL,
                        range_start DATETIME(6) NOT NULL,
                        range_end DATETIME(6) NOT NULL,
                        observed_at DATETIME(6) NOT NULL,
                        official_total INT UNSIGNED NOT NULL,
                        endpoint_total INT UNSIGNED NOT NULL,
                        mysql_total INT UNSIGNED NOT NULL,
                        tracking_posts_delta INT NOT NULL,
                        endpoint_mysql_delta INT NOT NULL,
                        suspected_deleted_count INT UNSIGNED NOT NULL,
                        missing_from_mysql_count INT UNSIGNED NOT NULL,
                        suspected_deleted_ids_json JSON NOT NULL,
                        missing_from_mysql_ids_json JSON NOT NULL,
                        PRIMARY KEY (id),
                        KEY ix_reconcile_handle_observed (handle, observed_at),
                        KEY ix_reconcile_tracking (tracking_id, observed_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS elon_monitor_snapshots (
                        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                        event_slug VARCHAR(255) NOT NULL,
                        observed_at DATETIME(6) NOT NULL,
                        post_count INT UNSIGNED NOT NULL,
                        expected_final DECIMAL(12,4) NOT NULL,
                        base_expected_final DECIMAL(12,4) NULL,
                        stddev_final DECIMAL(12,4) NOT NULL,
                        regime VARCHAR(32) NOT NULL,
                        realtime_multiplier DECIMAL(12,6) NOT NULL,
                        recent_counts_json JSON NOT NULL,
                        target_band_json JSON NULL,
                        target_probability DECIMAL(12,8) NULL,
                        held_band_json JSON NULL,
                        held_probability DECIMAL(12,8) NULL,
                        invested_usd DECIMAL(18,8) NULL,
                        liquidation_usd DECIMAL(18,8) NULL,
                        model_value_usd DECIMAL(18,8) NULL,
                        unrealized_return DECIMAL(12,8) NULL,
                        roll_count TINYINT UNSIGNED NULL,
                        rotation_value_gain_pct DECIMAL(12,8) NULL,
                        alerts_json JSON NOT NULL,
                        created_at DATETIME(6) NOT NULL,
                        PRIMARY KEY (id),
                        UNIQUE KEY uq_event_observed (event_slug, observed_at),
                        KEY ix_event_created (event_slug, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS polymarket_quote_snapshots (
                        monitor_snapshot_id BIGINT UNSIGNED NOT NULL,
                        asset_id VARCHAR(128) NOT NULL,
                        market_slug VARCHAR(255) NOT NULL,
                        interval_label VARCHAR(64) NOT NULL,
                        bid DECIMAL(12,8) NOT NULL,
                        ask DECIMAL(12,8) NOT NULL,
                        net_bid DECIMAL(12,8) NOT NULL,
                        all_in_ask DECIMAL(12,8) NOT NULL,
                        fair_probability DECIMAL(12,8) NOT NULL,
                        PRIMARY KEY (monitor_snapshot_id, asset_id),
                        KEY ix_quote_asset (asset_id),
                        CONSTRAINT fk_quote_monitor_snapshot
                            FOREIGN KEY (monitor_snapshot_id)
                            REFERENCES elon_monitor_snapshots (id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS polymarket_orderbook_levels (
                        monitor_snapshot_id BIGINT UNSIGNED NOT NULL,
                        asset_id VARCHAR(128) NOT NULL,
                        side ENUM('bid', 'ask') NOT NULL,
                        level_index TINYINT UNSIGNED NOT NULL,
                        price DECIMAL(12,8) NOT NULL,
                        size DECIMAL(24,8) NOT NULL,
                        PRIMARY KEY (monitor_snapshot_id, asset_id, side, level_index),
                        CONSTRAINT fk_level_monitor_snapshot
                            FOREIGN KEY (monitor_snapshot_id)
                            REFERENCES elon_monitor_snapshots (id)
                            ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS elon_event_outcomes (
                        event_slug VARCHAR(255) NOT NULL,
                        tracking_id VARCHAR(64) NOT NULL,
                        handle VARCHAR(64) NOT NULL,
                        event_start DATETIME(6) NOT NULL,
                        event_end DATETIME(6) NOT NULL,
                        final_count INT UNSIGNED NOT NULL,
                        winning_interval_label VARCHAR(64) NOT NULL,
                        winning_lower INT UNSIGNED NOT NULL,
                        winning_upper INT UNSIGNED NULL,
                        label_source VARCHAR(64) NOT NULL,
                        endpoint_total INT UNSIGNED NOT NULL,
                        mysql_total INT UNSIGNED NOT NULL,
                        tracking_posts_delta INT NOT NULL,
                        endpoint_mysql_delta INT NOT NULL,
                        observed_at DATETIME(6) NOT NULL,
                        label_version INT UNSIGNED NOT NULL DEFAULT 1,
                        created_at DATETIME(6) NOT NULL,
                        updated_at DATETIME(6) NOT NULL,
                        PRIMARY KEY (event_slug),
                        KEY ix_event_outcome_end (event_end),
                        KEY ix_event_outcome_tracking (tracking_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
            connection.commit()

    def upsert_posts(self, handle: str, posts: list[XTrackerPost]) -> int:
        if not posts:
            return 0
        now = datetime.now(UTC).replace(tzinfo=None)
        rows = [
            (
                post.platform_id,
                post.tracker_id,
                post.user_id,
                handle,
                post.content,
                _mysql_datetime(post.created_at),
                _mysql_datetime(post.imported_at),
                json.dumps(post.metrics, ensure_ascii=False) if post.metrics is not None else None,
                json.dumps(post.raw_payload, ensure_ascii=False),
                now,
                now,
            )
            for post in posts
        ]
        statement = """
            INSERT INTO xtracker_posts (
                platform_post_id, xtracker_post_id, xtracker_user_id, handle, content,
                created_at, imported_at, metrics_json, raw_payload, first_seen_at, last_seen_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                xtracker_post_id = VALUES(xtracker_post_id),
                xtracker_user_id = VALUES(xtracker_user_id),
                handle = VALUES(handle),
                content = COALESCE(VALUES(content), content),
                created_at = VALUES(created_at),
                imported_at = VALUES(imported_at),
                metrics_json = VALUES(metrics_json),
                raw_payload = VALUES(raw_payload),
                last_seen_at = VALUES(last_seen_at)
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(statement, rows)
            connection.commit()
        return len(rows)

    def record_sync_run(
        self,
        *,
        handle: str,
        start: datetime,
        end: datetime,
        started_at: datetime,
        status: str,
        fetched_count: int,
        error_message: str | None = None,
    ) -> None:
        statement = """
            INSERT INTO xtracker_sync_runs (
                handle, range_start, range_end, started_at, finished_at,
                status, fetched_count, error_message
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    statement,
                    (
                        handle,
                        _mysql_datetime(start),
                        _mysql_datetime(end),
                        _mysql_datetime(started_at),
                        datetime.now(UTC).replace(tzinfo=None),
                        status,
                        fetched_count,
                        error_message,
                    ),
                )
            connection.commit()

    def latest_created_at(self, handle: str) -> datetime | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT MAX(created_at) FROM xtracker_posts WHERE handle = %s", (handle,)
            )
            value = cursor.fetchone()[0]
        return value.replace(tzinfo=UTC) if value is not None else None

    def latest_successful_sync_end(self, handle: str) -> datetime | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                    SELECT MAX(range_end) FROM xtracker_sync_runs
                    WHERE handle = %s AND status = 'success'
                    """,
                (handle,),
            )
            value = cursor.fetchone()[0]
        return value.replace(tzinfo=UTC) if value is not None else None

    def stats(self, handle: str) -> dict[str, Any]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                    SELECT COUNT(*), MIN(created_at), MAX(created_at),
                           SUM(content IS NOT NULL AND content <> '')
                    FROM xtracker_posts WHERE handle = %s
                    """,
                (handle,),
            )
            count, earliest, latest, with_content = cursor.fetchone()
            cursor.execute(
                """
                    SELECT status, started_at, finished_at, fetched_count, error_message
                    FROM xtracker_sync_runs WHERE handle = %s ORDER BY id DESC LIMIT 1
                    """,
                (handle,),
            )
            last_sync = cursor.fetchone()
        return {
            "count": int(count),
            "earliest": earliest,
            "latest": latest,
            "with_content": int(with_content or 0),
            "last_sync": last_sync,
        }

    def post_times(self, handle: str, *, start: datetime, end: datetime) -> list[datetime]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                    SELECT created_at FROM xtracker_posts
                    WHERE handle = %s AND created_at >= %s AND created_at < %s
                    ORDER BY created_at
                    """,
                (handle, _mysql_datetime(start), _mysql_datetime(end)),
            )
            rows = cursor.fetchall()
        return [row[0].replace(tzinfo=UTC) for row in rows]

    def post_observations(
        self, handle: str, *, start: datetime, end: datetime
    ) -> list[PostObservation]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                    SELECT created_at, imported_at FROM xtracker_posts
                    WHERE handle = %s AND created_at >= %s AND created_at < %s
                    ORDER BY created_at
                    """,
                (handle, _mysql_datetime(start), _mysql_datetime(end)),
            )
            rows = cursor.fetchall()
        return [
            PostObservation(
                created_at=row[0].replace(tzinfo=UTC),
                imported_at=row[1].replace(tzinfo=UTC) if row[1] is not None else None,
            )
            for row in rows
        ]

    def post_ids(self, handle: str, *, start: datetime, end: datetime) -> set[str]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                    SELECT platform_post_id FROM xtracker_posts
                    WHERE handle = %s AND created_at >= %s AND created_at < %s
                    """,
                (handle, _mysql_datetime(start), _mysql_datetime(end)),
            )
            rows = cursor.fetchall()
        return {str(row[0]) for row in rows}

    def record_reconciliation(self, payload: dict[str, Any]) -> int:
        statement = """
            INSERT INTO xtracker_reconciliation_runs (
                handle, tracking_id, range_start, range_end, observed_at,
                official_total, endpoint_total, mysql_total, tracking_posts_delta,
                endpoint_mysql_delta, suspected_deleted_count, missing_from_mysql_count,
                suspected_deleted_ids_json, missing_from_mysql_ids_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    statement,
                    (
                        payload["handle"],
                        payload["tracking_id"],
                        _mysql_datetime(payload["range_start"]),
                        _mysql_datetime(payload["range_end"]),
                        _mysql_datetime(payload["observed_at"]),
                        payload["official_total"],
                        payload["endpoint_total"],
                        payload["mysql_total"],
                        payload["tracking_posts_delta"],
                        payload["endpoint_mysql_delta"],
                        payload["suspected_deleted_count"],
                        payload["missing_from_mysql_count"],
                        json.dumps(payload["suspected_deleted_ids"]),
                        json.dumps(payload["missing_from_mysql_ids"]),
                    ),
                )
                identifier = int(cursor.lastrowid)
            connection.commit()
        return identifier

    def latest_reconciliation(self, handle: str) -> dict[str, Any] | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                    SELECT tracking_id, range_start, range_end, observed_at,
                           official_total, endpoint_total, mysql_total,
                           tracking_posts_delta, endpoint_mysql_delta,
                           suspected_deleted_count, missing_from_mysql_count
                    FROM xtracker_reconciliation_runs
                    WHERE handle = %s ORDER BY observed_at DESC, id DESC LIMIT 1
                    """,
                (handle,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        keys = (
            "tracking_id",
            "range_start",
            "range_end",
            "observed_at",
            "official_total",
            "endpoint_total",
            "mysql_total",
            "tracking_posts_delta",
            "endpoint_mysql_delta",
            "suspected_deleted_count",
            "missing_from_mysql_count",
        )
        result = dict(zip(keys, row, strict=True))
        for key in ("range_start", "range_end", "observed_at"):
            result[key] = result[key].replace(tzinfo=UTC).isoformat()
        return result

    def record_monitor_snapshot(
        self,
        snapshot: dict[str, Any],
        quotes: list[dict[str, Any]],
    ) -> int:
        statement = """
            INSERT INTO elon_monitor_snapshots (
                event_slug, observed_at, post_count, expected_final, base_expected_final,
                stddev_final, regime, realtime_multiplier, recent_counts_json,
                target_band_json, target_probability, held_band_json, held_probability,
                invested_usd, liquidation_usd, model_value_usd, unrealized_return,
                roll_count, rotation_value_gain_pct, alerts_json, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id)
        """
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    statement,
                    (
                        snapshot["event_slug"],
                        _mysql_datetime(snapshot["observed_at"]),
                        snapshot["post_count"],
                        snapshot["expected_final"],
                        snapshot.get("base_expected_final"),
                        snapshot["stddev_final"],
                        snapshot["regime"],
                        snapshot["realtime_multiplier"],
                        json.dumps(snapshot["recent_counts"]),
                        json.dumps(snapshot.get("target_band")),
                        snapshot.get("target_probability"),
                        json.dumps(snapshot.get("held_band")),
                        snapshot.get("held_probability"),
                        snapshot.get("invested_usd"),
                        snapshot.get("liquidation_usd"),
                        snapshot.get("model_value_usd"),
                        snapshot.get("unrealized_return"),
                        snapshot.get("roll_count"),
                        snapshot.get("rotation_value_gain_pct"),
                        json.dumps(snapshot.get("alerts", [])),
                        now,
                    ),
                )
                snapshot_id = int(cursor.lastrowid)
                if quotes and snapshot_id:
                    cursor.executemany(
                        """
                        INSERT IGNORE INTO polymarket_quote_snapshots (
                            monitor_snapshot_id, asset_id, market_slug, interval_label,
                            bid, ask, net_bid, all_in_ask, fair_probability
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        [
                            (
                                snapshot_id,
                                item["asset_id"],
                                item["market_slug"],
                                item["interval_label"],
                                item["bid"],
                                item["ask"],
                                item["net_bid"],
                                item["all_in_ask"],
                                item["fair_probability"],
                            )
                            for item in quotes
                        ],
                    )
                    levels = []
                    for item in quotes:
                        for side in ("bid", "ask"):
                            for index, (price, size) in enumerate(item.get(f"{side}s", ())):
                                levels.append(
                                    (
                                        snapshot_id,
                                        item["asset_id"],
                                        side,
                                        index,
                                        price,
                                        size,
                                    )
                                )
                    if levels:
                        cursor.executemany(
                            """
                            INSERT IGNORE INTO polymarket_orderbook_levels (
                                monitor_snapshot_id, asset_id, side, level_index, price, size
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            levels,
                        )
            connection.commit()
        return snapshot_id

    def event_outcome(self, event_slug: str) -> dict[str, Any] | None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT tracking_id, handle, event_start, event_end, final_count,
                       winning_interval_label, winning_lower, winning_upper, label_source,
                       endpoint_total, mysql_total, tracking_posts_delta,
                       endpoint_mysql_delta, observed_at, label_version,
                       created_at, updated_at
                FROM elon_event_outcomes
                WHERE event_slug = %s
                """,
                (event_slug,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        names = (
            "tracking_id",
            "handle",
            "event_start",
            "event_end",
            "final_count",
            "winning_interval_label",
            "winning_lower",
            "winning_upper",
            "label_source",
            "endpoint_total",
            "mysql_total",
            "tracking_posts_delta",
            "endpoint_mysql_delta",
            "observed_at",
            "label_version",
            "created_at",
            "updated_at",
        )
        result = dict(zip(names, row, strict=True))
        result["event_slug"] = event_slug
        return result

    def record_event_outcome(self, payload: dict[str, Any]) -> str:
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT final_count, winning_interval_label, winning_lower,
                           winning_upper, label_version
                    FROM elon_event_outcomes
                    WHERE event_slug = %s
                    FOR UPDATE
                    """,
                    (payload["event_slug"],),
                )
                existing = cursor.fetchone()
                values = (
                    payload["tracking_id"],
                    payload["handle"],
                    _mysql_datetime(payload["event_start"]),
                    _mysql_datetime(payload["event_end"]),
                    payload["final_count"],
                    payload["winning_interval_label"],
                    payload["winning_lower"],
                    payload.get("winning_upper"),
                    payload.get("label_source", "xtracker_tracking"),
                    payload["endpoint_total"],
                    payload["mysql_total"],
                    payload["tracking_posts_delta"],
                    payload["endpoint_mysql_delta"],
                    _mysql_datetime(payload["observed_at"]),
                )
                if existing is None:
                    cursor.execute(
                        """
                        INSERT INTO elon_event_outcomes (
                            event_slug, tracking_id, handle, event_start, event_end,
                            final_count, winning_interval_label, winning_lower,
                            winning_upper, label_source, endpoint_total, mysql_total,
                            tracking_posts_delta, endpoint_mysql_delta, observed_at,
                            label_version, created_at, updated_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, 1, %s, %s
                        )
                        """,
                        (payload["event_slug"], *values, now, now),
                    )
                    status = "inserted"
                else:
                    label_values = (
                        payload["final_count"],
                        payload["winning_interval_label"],
                        payload["winning_lower"],
                        payload.get("winning_upper"),
                    )
                    changed = existing[:4] != label_values
                    version = int(existing[4]) + int(changed)
                    cursor.execute(
                        """
                        UPDATE elon_event_outcomes
                        SET tracking_id = %s, handle = %s, event_start = %s,
                            event_end = %s, final_count = %s,
                            winning_interval_label = %s, winning_lower = %s,
                            winning_upper = %s, label_source = %s,
                            endpoint_total = %s, mysql_total = %s,
                            tracking_posts_delta = %s, endpoint_mysql_delta = %s,
                            observed_at = %s, label_version = %s, updated_at = %s
                        WHERE event_slug = %s
                        """,
                        (*values, version, now, payload["event_slug"]),
                    )
                    status = "updated" if changed else "unchanged"
            connection.commit()
        return status
