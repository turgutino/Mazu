from datetime import datetime, timezone
from pathlib import Path

from mazu.curator.areas import Area
from mazu.curator.store import CuratorStore


def _parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def is_eligible(
    curator_store: CuratorStore, area: Area, root: Path, forced: bool, full: bool
) -> bool:
    """forced=True means the caller explicitly asked for this area via --area, which
    always runs it regardless of watermark/signal. full=True (--full) ignores every
    watermark but still requires a real signal (no point spending money analyzing
    an area with nothing to look at, even on a forced full sweep) -- forced is the
    one flag that skips the signal check too, since an explicit --area request is
    the user directly asking, not an automated decision."""
    if forced:
        return True
    if not full:
        watermark = curator_store.get_watermark(area.name)
        if watermark is not None and watermark["last_run_at"] is not None:
            age_days = (datetime.now(timezone.utc) - _parse_iso(watermark["last_run_at"])).total_seconds() / 86400
            if age_days < area.min_interval_days:
                return False
    return area.signal_fn(root) > 0
