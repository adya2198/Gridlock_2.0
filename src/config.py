"""Central configuration: paths and domain constants."""

from __future__ import annotations

from pathlib import Path

# ---- Paths ---------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "train_model.csv"
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

# User-added events captured from the app feed back into training.
FEEDBACK_CSV = DATA_DIR / "user_events.csv"

# Operator status changes (e.g. marking an active incident resolved) keyed by id.
STATUS_OVERRIDES_CSV = DATA_DIR / "status_overrides.csv"

MODEL_PATH = ARTIFACTS_DIR / "impact_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---- Bengaluru geographic centre (for map defaults) ---------------------

BLR_CENTER = (12.9716, 77.5946)

# ---- Domain weights ------------------------------------------------------

# How much each cause typically disrupts traffic (0-1).
# Tuned from domain knowledge; used both as a model feature
# and inside the impact target.

CAUSE_DISRUPTION_WEIGHT: dict[str, float] = {
    "accident": 0.95,
    "vip_movement": 0.90,
    "procession": 0.88,
    "protest": 0.85,
    "public_event": 0.80,
    "water_logging": 0.75,
    "tree_fall": 0.70,
    "construction": 0.65,
    "congestion": 0.60,
    "vehicle_breakdown": 0.55,
    "road_conditions": 0.50,
    "debris": 0.45,
    "pot_holes": 0.40,
    "fog / low visibility": 0.50,
    "others": 0.35,
    "test_demo": 0.10,
}

# Corridors are major arterial roads -> higher base impact than side streets.

MAJOR_CORRIDOR_KEYWORDS = (
    "orr",
    "mysore",
    "bellary",
    "tumkur",
    "hosur",
    "magadi",
    "old madras",
    "bannerghata",
    "chord",
    "airport",
    "cbd",
)

# Event-like causes (the gathering/event family the problem statement targets).

EVENT_CAUSES = {
    "public_event",
    "procession",
    "vip_movement",
    "protest",
}