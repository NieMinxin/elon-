from polymarket_bot.elon_data import build_parser


def test_sync_defaults_to_inclusive_database_watermark() -> None:
    args = build_parser().parse_args(["sync"])

    assert args.start is None
    assert args.overlap_hours == 0


def test_sync_accepts_explicit_start_and_overlap() -> None:
    args = build_parser().parse_args(
        ["--handle", "elonmusk", "sync", "--start", "2026-10-17T00:00:00Z", "--overlap-hours", "2"]
    )

    assert args.start == "2026-10-17T00:00:00Z"
    assert args.overlap_hours == 2


def test_reconcile_requires_explicit_tracking_period() -> None:
    args = build_parser().parse_args(
        [
            "reconcile",
            "--start",
            "2026-10-13T16:00:00Z",
            "--end",
            "2026-10-20T16:00:00Z",
        ]
    )

    assert args.tracking_id is None
    assert args.start == "2026-10-13T16:00:00Z"
