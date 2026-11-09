"""Progress lines in the output of a notebook cell.

Written to stdout at once: Databricks shows them while the cell runs and keeps the last ones when the process is killed,
so that the line ``started`` of the document being processed names it. The memory of the process tells a slow rise over
the documents from a jump on one of them.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _memory() -> str:
    """Never fails the step: it is only an indication."""
    try:
        import psutil

        return f" | memory {psutil.Process().memory_info().rss / 1024 ** 3:.1f} GB"
    except Exception:  # noqa: BLE001
        return ""


def report(message: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S} UTC] {message}{_memory()}", flush=True)
