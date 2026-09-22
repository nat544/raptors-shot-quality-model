"""
Trains a gradient-boosted (XGBoost) shot-quality model predicting the
probability a given Raptors shot attempt is made, from spatio-temporal
features, and reports standard classification metrics.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import xgboost as xgb

FEATURES = ["distance_ft", "x", "y", "game_elapsed_s", "period", "score_margin", "is_clutch", "is_three"]


def train_and_evaluate(df: pd.DataFrame, random_state: int = 42):
    X = df[FEATURES].astype(float)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "log_loss": log_loss(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "n_test": len(y_test),
        "make_rate_test": y_test.mean(),
    }
    return clf, metrics, (X_test, y_test, proba)


if __name__ == "__main__":
    print("Import this module from the notebook; it's not meant to run standalone without data.")
