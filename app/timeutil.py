"""Time helpers. Everything internal is epoch seconds (UTC); ISO strings only at the edges."""
from datetime import datetime, timezone


def parse_time(value) -> float | None:
    """Accepts epoch (int/float/numeric string) or ISO-8601 ('Z' ok; naive = UTC). None passes through."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)
    except ValueError:
        pass
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def humanize(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 45:
        return "just now"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        h, m = divmod(s // 60, 60)
        return f"{h}h {m}m" if m else f"{h}h"
    d, rem = divmod(s, 86400)
    h = rem // 3600
    return f"{d}d {h}h" if h else f"{d}d"
