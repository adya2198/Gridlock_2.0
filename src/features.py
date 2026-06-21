"""
Feature engineering + impact-target construction.

KEY INSIGHT: the raw data has no "impact" column. We construct an objective
'impact_score' (0-100) by blending:

* clear_time_min  -> how long it took to clear (strongest objective signal),
* priority        -> operator-assigned urgency,
* road closure    -> closures are high-impact by definition,
* cause weight    -> accidents/processions disrupt more than potholes,
* corridor + peak -> arterial road during peak hour amplifies impact,
* festival/holiday -> structural footfall amplification.

We also build a recurring-hotspot signal: historical incident density per
(junction, hour, dow) so the model knows chronically-congested spots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar_features import attach_calendar
from .config import CAUSE_DISRUPTION_WEIGHT

# Feature columns fed to the model.
CATEGORICAL = [
    "event_cause",
    "event_type",
    "veh_type",
    "corridor",
    "priority",
    "zone",
]

NUMERIC = [
    "requires_road_closure",
    "is_major_corridor",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_peak",
    "is_public_holiday",
    "is_festival",
    "festival_footfall",
    "cause_weight",
    "hotspot_density",
]

TARGET = "impact_score"


def _robust_minmax(s: pd.Series) -> pd.Series:
    """Scale to 0-1 using 5th-95th percentile to resist outliers."""
    lo, hi = s.quantile(0.05), s.quantile(0.95)

    if hi <= lo:
        return pd.Series(0.5, index=s.index)

    return ((s - lo) / (hi - lo)).clip(0, 1)


def add_cause_weight(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["cause_weight"] = (
        df["event_cause"]
        .map(CAUSE_DISRUPTION_WEIGHT)
        .fillna(0.35)
        .astype(float)
    )

    return df


def add_hotspot_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical incident count per (junction, hour-band, dow),
    normalised 0-1.

    Captures 'this spot is chronically busy at this time'
    independent of the current incident's cause.
    """

    df = df.copy()

    df["hour_band"] = (df["hour"] // 3).astype("Int64")

    grp = (
        df.groupby(["junction", "hour_band", "dow"])
        .size()
        .rename("cnt")
        .reset_index()
    )

    df = df.merge(
        grp,
        on=["junction", "hour_band", "dow"],
        how="left",
    )

    # Unknown junction shouldn't dominate; cap its influence.
    df.loc[df["junction"] == "unknown", "cnt"] = df["cnt"].median()

    df["hotspot_density"] = (
        _robust_minmax(df["cnt"].astype(float))
    )

    return df.drop(columns=["hour_band", "cnt"])


def build_impact_target(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the 0-100 impact_score label."""

    df = df.copy()

    clear_norm = _robust_minmax(
        df["clear_time_min"].fillna(
            df["clear_time_min"].median()
        )
    )

    priority_norm = (
        df["priority"]
        .eq("high")
        .astype(float)
    )

    closure_norm = df["requires_road_closure"].astype(float)
    cause_norm = df["cause_weight"]
    corridor_norm = df["is_major_corridor"].astype(float)
    peak_norm = df["is_peak"].astype(float)
    footfall_norm = df["festival_footfall"].astype(float)

    # Weighted blend (weights sum to 1.0).
    score01 = (
        0.30 * clear_norm
        + 0.18 * cause_norm
        + 0.16 * priority_norm
        + 0.14 * closure_norm
        + 0.10 * corridor_norm
        + 0.07 * peak_norm
        + 0.05 * footfall_norm
    )

    df[TARGET] = (score01 * 100).round(2)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: calendar -> cause weight -> hotspot -> target."""

    df = attach_calendar(df)
    df = add_cause_weight(df)
    df = add_hotspot_density(df)
    df = build_impact_target(df)

    return df


def feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) ready for the model.
    Categoricals as pandas 'category'.
    """

    X = df[CATEGORICAL + NUMERIC].copy()

    for c in CATEGORICAL:
        X[c] = X[c].astype("category")

    for c in NUMERIC:
        X[c] = pd.to_numeric(
            X[c],
            errors="coerce"
        ).fillna(0.0)

    y = df[TARGET].astype(float)

    return X, y


def impact_tier(score: float) -> str:
    if score >= 65:
        return "SEVERE"

    if score >= 45:
        return "HIGH"

    if score >= 25:
        return "MODERATE"

    return "LOW"


if __name__ == "__main__":
    from .data_prep import clean

    d = build_features(clean())

    X, y = feature_matrix(d)

    print(f"Rows: {len(d)} | features: {X.shape[1]}")
    print(y.describe().round(2))

    print("\nTier distribution:")
    print(y.apply(impact_tier).value_counts())