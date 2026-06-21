"""
Calendar enrichment: national holidays, festivals, and known rally days.

The problem statement explicitly calls out festivals, rallies and gatherings.
Incident-level data alone misses this, so we join an external calendar signal:
- India + Karnataka public holidays (via the 'holidays' package),
- a curated festival/major-event list (high-footfall days),
so the model learns that some dates are structurally higher-risk.
"""

from __future__ import annotations

import datetime as dt

import holidays
import pandas as pd

# Curated high-footfall festivals / event windows relevant to Bengaluru.
# Month-day based so they apply every year (approximate for lunar festivals).
# Each entry: (month, day, name, footfall_weight 0-1).

_RECURRING_FESTIVALS: list[tuple[int, int, str, float]] = [
    (1, 1, "New Year", 0.7),
    (1, 14, "Sankranti", 0.6),
    (1, 26, "Republic Day", 0.8),
    (3, 8, "Holi", 0.6),
    (4, 14, "Ambedkar Jayanti", 0.6),
    (8, 15, "Independence Day", 0.8),
    (9, 7, "Ganesh Chaturthi", 0.95),
    (10, 2, "Gandhi Jayanti", 0.6),
    (10, 31, "Kannada Rajyotsava window", 0.7),
    (11, 1, "Kannada Rajyotsava", 0.85),
    (12, 25, "Christmas", 0.5),
    (12, 31, "New Year Eve", 0.8),
]


def _festival_lookup() -> dict[tuple[int, int], tuple[str, float]]:
    return {(m, d): (name, w) for (m, d, name, w) in _RECURRING_FESTIVALS}


def build_calendar(years: list[int]) -> pd.DataFrame:
    """Return a per-date calendar DataFrame for the requested years."""

    in_holidays = holidays.India(years=years, subdiv="KA")
    fest = _festival_lookup()

    rows = []

    for year in years:
        d = dt.date(year, 1, 1)
        end = dt.date(year, 12, 31)

        while d <= end:
            is_holiday = d in in_holidays
            holiday_name = in_holidays.get(d, "")

            fest_name, fest_w = fest.get(
                (d.month, d.day),
                ("", 0.0)
            )

            rows.append(
                {
                    "date": d,
                    "is_public_holiday": int(is_holiday),
                    "holiday_name": holiday_name or fest_name,
                    "is_festival": int(fest_w > 0),
                    "festival_footfall": fest_w,
                }
            )

            d += dt.timedelta(days=1)

    return pd.DataFrame(rows)


def attach_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join calendar signals onto an incident DataFrame
    with a 'date' column.
    """

    df = df.copy()

    years = sorted(
        {d.year for d in pd.to_datetime(df["date"]).dt.date.unique()}
    )

    cal = build_calendar(years)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    cal["date"] = pd.to_datetime(cal["date"]).dt.date

    merged = df.merge(cal, on="date", how="left")

    for col, fill in [
        ("is_public_holiday", 0),
        ("is_festival", 0),
        ("festival_footfall", 0.0),
        ("holiday_name", ""),
    ]:
        merged[col] = merged[col].fillna(fill)

    return merged


if __name__ == "__main__":
    cal = build_calendar([2024])

    print(
        cal[cal["is_public_holiday"] == 1]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nFestival days 2024: {cal['is_festival'].sum()}")