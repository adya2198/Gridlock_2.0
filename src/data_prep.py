"""
Load and clean the raw incident log into a tidy DataFrame.

The raw file is an incident/event log from Bengaluru traffic operations.
This module:
* parses datetimes (handling 'NULL' strings and tz suffixes),
* derives clear-time (minutes to resolve) as an objective impact signal,
* normalises noisy categoricals,
* keeps only rows usable for modelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEEDBACK_CSV, MAJOR_CORRIDOR_KEYWORDS, RAW_CSV
from .feedback import load_status_overrides


# Columns we actually use downstream (the raw file has 40+).
_KEEP = [
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


def _to_dt(series: pd.Series) -> pd.Series:
    """Parse a column of mixed datetime strings, mapping NULL/'' -> NaT."""
    cleaned = series.replace({"NULL": np.nan, "": np.nan, "None": np.nan})
    return pd.to_datetime(cleaned, errors="coerce", utc=True)


def _norm_text(series: pd.Series) -> pd.Series:
    out = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace({"null": pd.NA, "": pd.NA, "none": pd.NA})
    )
    return out


def _is_major_corridor(corridor: pd.Series) -> pd.Series:
    low = corridor.fillna("").str.lower()
    mask = pd.Series(False, index=corridor.index)

    for kw in MAJOR_CORRIDOR_KEYWORDS:
        mask = mask | low.str.contains(kw, na=False)

    return mask.astype(int)


def load_raw() -> pd.DataFrame:
    """Read the base CSV plus any user-added feedback events."""

    base = pd.read_csv(RAW_CSV, dtype=str, low_memory=False)
    frames = [base]

    if FEEDBACK_CSV.exists():
        fb = pd.read_csv(FEEDBACK_CSV, dtype=str, low_memory=False)

        if not fb.empty:
            frames.append(fb)

    df = pd.concat(frames, ignore_index=True)

    # Ensure all expected columns exist even if feedback file is sparse.
    for col in _KEEP:
        if col not in df.columns:
            df[col] = np.nan

    return df[_KEEP].copy()


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a cleaned, model-ready DataFrame."""

    if df is None:
        df = load_raw()

    df = df.copy()

    # ---- numeric coords ----
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # ---- datetimes ----
    for col in [
        "start_datetime",
        "end_datetime",
        "modified_datetime",
        "resolved_datetime",
        "closed_datetime",
    ]:
        df[col] = _to_dt(df[col])

    # ---- categorical normalisation ----
    df["event_cause"] = _norm_text(df["event_cause"]).fillna("others")
    df["event_type"] = _norm_text(df["event_type"]).fillna("unplanned")
    df["veh_type"] = _norm_text(df["veh_type"]).fillna("unknown")
    df["corridor"] = _norm_text(df["corridor"]).fillna("non-corridor")
    df["priority"] = _norm_text(df["priority"]).fillna("low")
    df["zone"] = _norm_text(df["zone"]).fillna("unknown")
    df["police_station"] = _norm_text(df["police_station"]).fillna("unknown")
    df["junction"] = _norm_text(df["junction"]).fillna("unknown")

    df["requires_road_closure"] = (
        _norm_text(df["requires_road_closure"])
        .eq("true")
        .astype("Int64")
    )

    df["is_major_corridor"] = _is_major_corridor(df["corridor"])

    # Apply operator status changes
    # (e.g. an active incident marked resolved).
    overrides = load_status_overrides()

    if overrides:
        mask = df["id"].isin(overrides)
        df.loc[mask, "status"] = (
            df.loc[mask, "id"]
            .map(overrides)
        )

    # ---- clear-time (impact proxy) ----
    # Best available end timestamp, in priority order.
    end_ts = (
        df["resolved_datetime"]
        .fillna(df["closed_datetime"])
        .fillna(df["end_datetime"])
        .fillna(df["modified_datetime"])
    )

    clear_min = (
        end_ts - df["start_datetime"]
    ).dt.total_seconds() / 60.0

    # Guard against negative / absurd values (data entry noise).
    clear_min = clear_min.where(
        (clear_min > 0) &
        (clear_min < 60 * 24 * 3)
    )

    df["clear_time_min"] = clear_min

    # ---- temporal features from start datetime ----
    # Convert to India local time for hour/day semantics.
    local = df["start_datetime"].dt.tz_convert("Asia/Kolkata")

    df["hour"] = local.dt.hour
    df["dow"] = local.dt.dayofweek  # 0=Mon
    df["month"] = local.dt.month
    df["date"] = local.dt.date

    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["is_peak"] = (
        df["hour"]
        .isin([8, 9, 10, 17, 18, 19, 20])
        .astype("Int64")
        .fillna(0)
        .astype(int)
    )

    # Drop rows with no usable location or no start time.
    df = df[df["start_datetime"].notna()].copy()

    df = df[
        (df["latitude"].between(12.6, 13.3)) &
        (df["longitude"].between(77.3, 77.9))
    ].copy()

    return df.reset_index(drop=True)


if __name__ == "__main__":
    d = clean()

    print(f"Cleaned rows: {len(d)}")
    print(
        f"With clear_time: "
        f"{d['clear_time_min'].notna().sum()}"
    )

    print(
        d[
            [
                "event_cause",
                "priority",
                "hour",
                "is_peak",
                "clear_time_min",
            ]
        ].head()
    )