# 🚦 TrafficGuard — Event Traffic Impact Forecasting & Resource Planner

💡 **Forecasts the traffic impact of events** (rallies, festivals, sports, construction, breakdowns, gatherings) **in advance** and recommends optimal **manpower, barricading, and diversion** plans — and learns from every new event.

**Built for the operational challenge:**  
*"How can historical and real-time data be used to forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans?"*

---

## 🎯 The Core Insight (Why This Approach Wins)

The raw dataset (`train_model.csv`, ~8k Bengaluru incidents) has **no "impact" column** — so you cannot naively "predict impact". The breakthrough is to **engineer an objective impact target** from signals that *do* exist:

| Signal | Source column | Why it measures impact |
|---|---|---|
| Clear time (mins to resolve) | `start` -> `resolved/closed/modified_datetime` | Longer to clear = bigger disruption (strongest signal, **7,403 rows**) |
| Operator priority | `priority` | Human-assigned urgency |
| Road closure | `requires_road_closure` | Closures are high-impact by definition |
| Cause severity | `event_cause` | Accidents/processions disrupt more than potholes |
| Corridor + peak hour | `corridor`, hour | Arterial road at rush hour amplifies impact |
| Festival/holiday footfall | external calendar | Structural footfall on special days |

These blend into a **0-100 `impact_score`** that a LightGBM model learns to forecast for *any future event* — before it happens.

---

## 🔧 What It Does (4 Pillars)

1. **Forecast in advance** — enter an upcoming event's details -> predicted impact score + tier (LOW / MODERATE / HIGH / SEVERE).
2. **Recommend resources** — transparent rule+ML hybrid outputs officers, wardens, barricade units, tow/crane, a diversion plan, and citizen mitigation tips — *with a full rationale* operators can audit.
3. **Alert + map** — live alert feed for active incidents and a historical hotspot map to pre-position resources.
4. **Learn from every event** — operators log new events in-app; they feed straight back into the next retrain (closes the "no post-event learning" gap).

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -m src.model        # trains & saves the model (creates artifacts/)
streamlit run app.py       # launches the dashboard
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📐 Architecture

```plaintext
train_model.csv ┐
                ├─> data_prep.py ──> features.py ──> model.py (LightGBM)
user_events.csv ┘   (clean +         (impact_score   │
  (feedback)         clear-time)      + calendar)    ▼
                       │                             recommend.py ──> app.py
                       └───────── calendar_features.py ──┘        (Streamlit)
                                  (holidays + festivals)
```

## 📁 File Responsibilities

| File | Purpose |
|------|---------|
| `src/data_prep.py` | Clean CSV, parse datetimes, compute clear-time, normalise categories |
| `src/calendar_features.py` | India/Karnataka holidays + curated festival footfalls |
| `src/features.py` | Build the impact_score target + spatial/temporal/hotspot features |
| `src/model.py` | Train/persist/serve the LightGBM impact forecaster |
| `src/recommend.py` | Transparent rules → manpower, barricades, diversion, mitigations |
| `src/feedback.py` | Append operator-logged events to the learning store |
| `app.py` | Streamlit UI: dashboard, forecast, hotspot map, log event, model stats |

---

## 📊 Model Performance

*(Held-out 20% test set)*

| Metric | Value |
|--------|-------|
| MAE (0-100 scale) | ~4.6 |
| R² | ~0.67 |
| Tier accuracy | ~65% |

**Top drivers:** hour of day, day of week, month, cause severity, hotspot density, corridor — all operationally sensible.

---

## ✨ What Makes It Stand Out

1. **Target engineering** — solves the "no label" problem most teams miss.
2. **Calendar intelligence** — festivals/holidays/rallies baked in as features.
3. **Recurring-hotspot signal** — learns chronically-congested junction×time slots.
4. **Transparent recommendations** — every officer count comes with a rationale (auditable, deployable — not a black box).
5. **Closed learning loop** — the system genuinely improves with each new event.

---

## ⚠️ Notes & Honest Limitations

- **`impact_score` is a derived proxy**, not measured congestion. With real sensor/GPS-speed data it can be recalibrated to ground truth.
- **Festival dates for lunar calendars are approximate**; swap in an authoritative feed for production.
- **Manpower rules are domain heuristics**; tune the `_TIER_BASE` table with real deployment records as they accrue.