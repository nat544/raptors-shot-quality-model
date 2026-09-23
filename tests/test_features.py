"""
Unit tests for features.py -- game-clock math and the feature-engineering
logic (rim-shot distance imputation, clutch/three-point flags).
"""

import pandas as pd

from features import _game_elapsed_seconds, build_features


# ---- _game_elapsed_seconds -------------------------------------------------

def test_start_of_game_is_zero_elapsed():
    assert _game_elapsed_seconds(1, 720) == 0  # Q1, full 12:00 remaining


def test_partway_through_first_quarter():
    assert _game_elapsed_seconds(1, 360) == 360  # 6:00 played


def test_partway_through_fourth_quarter():
    # 3 full quarters (2160s) + 7 minutes into Q4 (420s)
    assert _game_elapsed_seconds(4, 300) == 2160 + 420


def test_start_of_first_overtime():
    # 4 full quarters = 2880s regulation, OT just starting (5:00 remaining)
    assert _game_elapsed_seconds(5, 300) == 2880


def test_start_of_second_overtime():
    # regulation (2880s) + one full 5-minute OT (300s)
    assert _game_elapsed_seconds(6, 300) == 3180


def test_missing_clock_returns_none():
    assert _game_elapsed_seconds(1, None) is None


# ---- build_features ---------------------------------------------------

def _base_row(**overrides):
    row = {
        "game_id": "1",
        "play_id": "1",
        "period": 1,
        "clock_remaining_s": 600,
        "x": 25,
        "y": 10,
        "distance_ft": 20,
        "points_attempted": 2,
        "made": True,
        "score_margin": 0,
        "text": "Some Player makes a shot",
    }
    row.update(overrides)
    return row


def test_rows_missing_coordinates_are_dropped():
    df = build_features([_base_row(x=None), _base_row(x=25)])
    assert len(df) == 1


def test_missing_distance_is_imputed_to_2ft_for_rim_shots():
    df = build_features([_base_row(distance_ft=None, text="Player makes layup")])
    assert df.iloc[0]["distance_ft"] == 2


def test_explicit_distance_is_preserved():
    df = build_features([_base_row(distance_ft=27)])
    assert df.iloc[0]["distance_ft"] == 27


def test_is_clutch_true_in_late_fourth_quarter():
    df = build_features([_base_row(period=4, clock_remaining_s=300)])
    assert bool(df.iloc[0]["is_clutch"]) is True


def test_is_clutch_false_earlier_in_fourth_quarter():
    df = build_features([_base_row(period=4, clock_remaining_s=301)])
    assert bool(df.iloc[0]["is_clutch"]) is False


def test_is_clutch_false_in_third_quarter_even_with_low_clock():
    df = build_features([_base_row(period=3, clock_remaining_s=10)])
    assert bool(df.iloc[0]["is_clutch"]) is False


def test_is_three_flag_matches_points_attempted():
    df = build_features([_base_row(points_attempted=3), _base_row(points_attempted=2)])
    assert df["is_three"].tolist() == [True, False]


def test_target_matches_made_flag():
    df = build_features([_base_row(made=True), _base_row(made=False)])
    assert df["target"].tolist() == [1, 0]


def test_build_features_on_empty_input_returns_empty_dataframe():
    assert build_features([]).empty
