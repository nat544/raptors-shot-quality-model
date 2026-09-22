"""
Features: shot distance/location, game-clock time remaining,
and score margin at the moment of the shot.
"""

import pandas as pd

TOTAL_GAME_SECONDS = 48 * 60  # regulation; overtime periods handled separately below


def _game_elapsed_seconds(period: int, clock_remaining_s) -> float:
    if clock_remaining_s is None:
        return None
    period_length = 12 * 60 if period <= 4 else 5 * 60  # OT periods are 5 minutes
    periods_before = min(period - 1, 4) * 12 * 60 + max(0, period - 5) * 5 * 60
    return periods_before + (period_length - clock_remaining_s)


def build_features(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.dropna(subset=["x", "y"]).copy()
    # Shots with no "-foot" mention in the play text are near-rim attempts
    # (layups, dunks, tip shots) -- ESPN's play text only calls out distance
    # for jump shots. Impute a short distance rather than dropping these rows.
    df["distance_ft"] = df["distance_ft"].fillna(2)

    df["game_elapsed_s"] = df.apply(
        lambda r: _game_elapsed_seconds(r["period"], r["clock_remaining_s"]), axis=1
    )
    df["is_clutch"] = (df["period"] >= 4) & (df["clock_remaining_s"] <= 300)
    df["is_three"] = df["points_attempted"] == 3
    df["target"] = df["made"].astype(int)

    feature_cols = [
        "distance_ft", "x", "y", "game_elapsed_s", "period",
        "score_margin", "is_clutch", "is_three",
    ]
    return df[feature_cols + ["target", "game_id", "play_id", "text", "points_attempted"]]
