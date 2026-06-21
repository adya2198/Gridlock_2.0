"""
Resource & mitigation recommender (rule + ML hybrid).

The predicted `impact_score` drives a transparent rule layer that outputs:
* manpower (traffic officers + wardens),
* barricade units,
* tow / crane requirement,
* diversion advice,
* citizen-facing mitigation tips,
* an alert level.

Rules are transparent on purpose: operators must be able to trust and audit
why a recommendation was made (a black-box officer count is a non-starter
in real deployments). The ML model decides "how big" the impact is; the
rules decide "what to do" about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .features import impact_tier


@dataclass
class Recommendation:
    impact_score: float
    tier: str
    alert_level: str
    officers: int
    wardens: int
    barricade_units: int
    tow_crane: bool
    diversion: str
    mitigations: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)


# Base manpower per tier (officers, wardens, barricades).
_TIER_BASE = {
    "SEVERE": (8, 6, 6),
    "HIGH": (5, 4, 3),
    "MODERATE": (2, 2, 1),
    "LOW": (1, 0, 0),
}

_ALERT = {
    "SEVERE": "RED",
    "HIGH": "ORANGE",
    "MODERATE": "YELLOW",
    "LOW": "GREEN",
}


def recommend(
    impact_score: float,
    *,
    event_cause: str = "others",
    requires_road_closure: bool = False,
    is_major_corridor: bool = False,
    is_peak: bool = False,
    is_festival: bool = False,
) -> Recommendation:

    tier = impact_tier(impact_score)

    officers, wardens, barricades = _TIER_BASE[tier]

    rationale: list[str] = [
        f"Predicted impact {impact_score:.0f}/100 -> {tier} tier."
    ]

    mitigations: list[str] = []
    tow = False

    # --- amplifiers -------------------------------------------------

    if is_peak:
        officers += 2
        wardens += 1
        rationale.append(
            "Peak-hour incident: +2 officers, +1 warden."
        )

    if is_major_corridor:
        officers += 1
        barricades += 1
        rationale.append(
            "Major arterial corridor: +1 officer, +1 barricade unit."
        )

    if is_festival:
        officers += 2
        wardens += 2
        rationale.append(
            "Festival/high-footfall day: +2 officers, +2 wardens."
        )

    if requires_road_closure:
        barricades += 2
        rationale.append(
            "Road closure required: +2 barricade units, diversion mandatory."
        )

    cause = event_cause.lower()

    if cause in {
        "accident",
        "vehicle_breakdown",
        "tree_fall",
        "debris",
    }:
        tow = True
        rationale.append(
            f"Cause '{cause}': tow/crane dispatched to clear the blockage."
        )

    if cause in {
        "procession",
        "protest",
        "vip_movement",
        "public_event",
    }:
        officers += 3
        barricades += 2

        rationale.append(
            f"Crowd-type event '{cause}': "
            "+3 officers, +2 barricades for crowd control."
        )

    if cause == "water_logging":
        mitigations.append(
            "Coordinate with BBMP for pumping; mark submerged stretch."
        )

    # --- diversion advice -------------------------------------------

    if requires_road_closure or tier in {"SEVERE", "HIGH"}:
        diversion = (
            "ACTIVATE diversion: route through-traffic to the nearest "
            "parallel arterial; pre-position signage 500 m upstream "
            "both directions."
        )

    elif tier == "MODERATE":
        diversion = (
            "STANDBY diversion: deploy cones in the affected lane only; "
            "keep one lane flowing."
        )

    else:
        diversion = (
            "No diversion needed; monitor and clear quickly."
        )

    # --- citizen-facing mitigations ---------------------------------

    if tier in {"SEVERE", "HIGH"}:
        mitigations.append(
            "Push real-time alert to navigation apps + variable message signs."
        )

        mitigations.append(
            "Advise commuters to delay non-essential travel by 30-45 min."
        )

        mitigations.append(
            "Hold/space-out signal cycles upstream to prevent gridlock spillback."
        )

    if is_peak:
        mitigations.append(
            "Stagger nearby school/office exit if possible."
        )

    if is_festival:
        mitigations.append(
            "Promote park-and-ride + extra public transport frequency."
        )

    if not mitigations:
        mitigations.append(
            "Routine monitoring; clear incident promptly to restore flow."
        )

    return Recommendation(
        impact_score=round(impact_score, 1),
        tier=tier,
        alert_level=_ALERT[tier],
        officers=officers,
        wardens=wardens,
        barricade_units=barricades,
        tow_crane=tow,
        diversion=diversion,
        mitigations=mitigations,
        rationale=rationale,
    )


if __name__ == "__main__":
    r = recommend(
        72,
        event_cause="procession",
        requires_road_closure=True,
        is_major_corridor=True,
        is_peak=True,
        is_festival=True,
    )

    from pprint import pprint

    pprint(r)