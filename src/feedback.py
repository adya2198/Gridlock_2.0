"""
User feedback loop: capture new events and retrain (the learning system).

This closes the "no post-event learning system" gap in the problem statement.
When an operator logs a new event in the app, we append it to a feedback CSV
that uses the SAME schema as the raw file. On retrain, data_prep.load_raw()
automatically concatenates this file, so every new event makes the next
forecast smarter.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pandas as pd

from .config import FEEDBACK_CSV, STATUS_OVERRIDES_CSV


# Minimal schema the pipeline needs; everything else can be blank/NULL.
_SCHEMA = [
    "id",
    "event_type",
    "latitude",
    "longitude",
    "event_cause",
    "requires_road_closure",
    "start_datetime",
    "end_datetime",
    "modified_datetime",
    "resolved_datetime",
    "closed_datetime",
    "status",
    "veh_type",
    "corridor",
    "priority",
    "zone",
    "police_station",
    "junction",
]


def add_event(
    *,
    latitude: float,
    longitude: float,
    event_cause: str,
    event_type: str = "unplanned",
    requires_road_closure: bool = False,
    priority: str = "high",
    corridor: str = "non-corridor",
    zone: str = "unknown",
    junction: str = "unknown",
    veh_type: str = "unknown",
    start_datetime: dt.datetime | None = None,
    duration_min: float | None = None,
) -> str:
    """Append a new operator-logged event. Returns its generated id."""

    start = start_datetime or dt.datetime.now(dt.timezone.utc)

    end = (
        start + dt.timedelta(minutes=duration_min)
        if duration_min and duration_min > 0
        else None
    )

    new_id = f"USR{uuid.uuid4().hex[:8].upper()}"

    row = {
        "id": new_id,
        "event_type": event_type,
        "latitude": latitude,
        "longitude": longitude,
        "event_cause": event_cause,
        "requires_road_closure": "TRUE" if requires_road_closure else "FALSE",
        "start_datetime": start.isoformat(),
        "end_datetime": end.isoformat() if end else "NULL",
        "modified_datetime": end.isoformat() if end else "NULL",
        "resolved_datetime": "NULL",
        "closed_datetime": "NULL",
        "status": "active",
        "veh_type": veh_type,
        "corridor": corridor,
        "priority": priority,
        "zone": zone,
        "police_station": "unknown",
        "junction": junction,
    }

    df_new = pd.DataFrame([row])[_SCHEMA]

    if FEEDBACK_CSV.exists():
        existing = pd.read_csv(FEEDBACK_CSV, dtype=str)
        df_new = pd.concat([existing, df_new], ignore_index=True)

    df_new.to_csv(FEEDBACK_CSV, index=False)

    return new_id


def feedback_count() -> int:
    if not FEEDBACK_CSV.exists():
        return 0

    return len(pd.read_csv(FEEDBACK_CSV, dtype=str))


def resolve_event(
    event_id: str,
    status: str = "resolved",
) -> None:
    """
    Record that an incident 'event_id' is no longer active.

    Stored as an id->status override applied at load time, so it works
    for both base-CSV and operator-logged incidents without mutating
    the source files.
    """

    now = dt.datetime.now(dt.timezone.utc).isoformat()

    row = {
        "id": str(event_id),
        "status": status,
        "resolved_at": now,
    }

    df_new = pd.DataFrame([row])

    if STATUS_OVERRIDES_CSV.exists():
        existing = pd.read_csv(STATUS_OVERRIDES_CSV, dtype=str)
        existing = existing[existing["id"] != str(event_id)]
        df_new = pd.concat([existing, df_new], ignore_index=True)

    df_new.to_csv(STATUS_OVERRIDES_CSV, index=False)


def load_status_overrides() -> dict[str, str]:
    """Return {id: status} of operator status changes (empty if none)."""

    if not STATUS_OVERRIDES_CSV.exists():
        return {}

    ov = pd.read_csv(STATUS_OVERRIDES_CSV, dtype=str)

    if ov.empty or "id" not in ov or "status" not in ov:
        return {}

    return dict(zip(ov["id"], ov["status"]))


if __name__ == "__main__":
    eid = add_event(
        latitude=12.9716,
        longitude=77.5946,
        event_cause="procession",
        requires_road_closure=True,
        priority="high",
        corridor="cbd 2",
        duration_min=120,
    )

    print(
        f"Added event {eid}. "
        f"Total feedback events: {feedback_count()}"
    )