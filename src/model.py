"""
Train, persist, and serve the impact-forecasting model.

A LightGBM regressor predicts `impact_score` (0-100) from event + spatial +
temporal + calendar features. LightGBM is chosen because it handles mixed
categorical/numeric data natively, trains fast, and is robust on tabular data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .config import METRICS_PATH, MODEL_PATH
from .data_prep import clean
from .features import (
    CATEGORICAL,
    NUMERIC,
    build_features,
    feature_matrix,
    impact_tier,
)


@dataclass
class TrainedModel:
    model: LGBMRegressor
    feature_cols: list[str]
    categorical: list[str]
    metrics: dict


def train(save: bool = True) -> TrainedModel:
    df = build_features(clean())
    X, y = feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_samples=40,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL,
    )

    pred = model.predict(X_test)

    metrics = {
        "mae": round(
            float(mean_absolute_error(y_test, pred)),
            3,
        ),
        "r2": round(
            float(r2_score(y_test, pred)),
            3,
        ),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "tier_accuracy": _tier_accuracy(
            y_test.to_numpy(),
            pred,
        ),
    }

    trained = TrainedModel(
        model=model,
        feature_cols=CATEGORICAL + NUMERIC,
        categorical=CATEGORICAL,
        metrics=metrics,
    )

    if save:
        # Persist a plain dict, not the dataclass:
        # avoids pickling a class whose module path differs
        # between `python -m src.model` and the app.
        joblib.dump(
            {
                "model": trained.model,
                "feature_cols": trained.feature_cols,
                "categorical": trained.categorical,
                "metrics": trained.metrics,
            },
            MODEL_PATH,
        )

        METRICS_PATH.write_text(
            json.dumps(metrics, indent=2)
        )

    return trained


def _tier_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    t_true = [impact_tier(v) for v in y_true]
    t_pred = [impact_tier(v) for v in y_pred]

    correct = sum(
        a == b
        for a, b in zip(t_true, t_pred)
    )

    return round(correct / len(t_true), 3)


def load() -> TrainedModel:
    if not MODEL_PATH.exists():
        return train()

    try:
        data = joblib.load(MODEL_PATH)
    except (AttributeError, ModuleNotFoundError):
        return train()

    if isinstance(data, TrainedModel):
        return data

    return TrainedModel(
        model=data["model"],
        feature_cols=data["feature_cols"],
        categorical=data["categorical"],
        metrics=data["metrics"],
    )


def predict_one(
    trained: TrainedModel,
    row: dict,
) -> float:
    """Predict impact_score for a single feature dict."""

    X = pd.DataFrame([row])

    for c in CATEGORICAL:
        if c not in X:
            X[c] = "others"

        X[c] = (
            X[c]
            .astype("string")
            .fillna("others")
            .astype("category")
        )

    for c in NUMERIC:
        val = X[c] if c in X else 0
        X[c] = (
            pd.to_numeric(
                val,
                errors="coerce",
            )
            .fillna(0.0)
        )

    X = X[trained.feature_cols]

    return float(
        trained.model.predict(X)[0]
    )


def feature_importance(
    trained: TrainedModel,
    top: int = 12,
) -> pd.DataFrame:
    imp = pd.DataFrame(
        {
            "feature": trained.feature_cols,
            "importance": trained.model.feature_importances_,
        }
    )

    return (
        imp.sort_values(
            "importance",
            ascending=False,
        )
        .head(top)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    tm = train()

    print(
        "Metrics:",
        json.dumps(
            tm.metrics,
            indent=2,
        ),
    )

    print("\nTop features:")

    print(
        feature_importance(tm)
        .to_string(index=False)
    )