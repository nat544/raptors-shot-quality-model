"""
Two visualizations:
  1. A shot-location scatter, colored by predicted make probability.
  2. A cumulative expected-points-over-game-time chart for a single game.
"""

import matplotlib.pyplot as plt


def plot_shot_map(df, prob_col="make_prob", ax=None):
    """df needs columns: x, y, and a predicted-probability column."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    sc = ax.scatter(df["x"], df["y"], c=df[prob_col], cmap="RdYlGn", s=25, alpha=0.8, edgecolors="k", linewidths=0.3)
    ax.set_title("Raptors shot attempts colored by predicted make probability")
    ax.set_xlabel("x (ESPN shot-chart coordinate)")
    ax.set_ylabel("y (ESPN shot-chart coordinate)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="Predicted make probability")
    return ax


def plot_cumulative_expected_points(df_game, prob_col="make_prob", ax=None):
    """df_game: rows for ONE game, sorted by game_elapsed_s, with make_prob,
    points_attempted, and actual `target` (made/missed)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    df_sorted = df_game.sort_values("game_elapsed_s").copy()
    df_sorted["expected_points"] = df_sorted[prob_col] * df_sorted["points_attempted"]
    df_sorted["actual_points"] = df_sorted["target"] * df_sorted["points_attempted"]

    ax.step(df_sorted["game_elapsed_s"] / 60, df_sorted["expected_points"].cumsum(), where="post", label="Expected points")
    ax.step(df_sorted["game_elapsed_s"] / 60, df_sorted["actual_points"].cumsum(), where="post", label="Actual points", linestyle="--")

    ax.set_xlabel("Game time (minutes)")
    ax.set_ylabel("Cumulative points from field goals")
    ax.set_title("Expected vs. actual scoring over game time (Raptors)")
    ax.legend()
    return ax
