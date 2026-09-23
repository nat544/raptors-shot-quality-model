"""
Unit tests for fetch_data.py: the distance-text parser and the play-by-play
shot extraction logic. Uses small hand-built ESPN-shaped fixtures -- no
network calls, no live API.
"""

from fetch_data import _parse_distance_ft, extract_shots, RAPTORS_TEAM_ID

OPPONENT_TEAM_ID = "7"


def test_parse_distance_ft_extracts_number_from_jump_shot_text():
    assert _parse_distance_ft("Scottie Barnes misses 25-foot three point jumper") == 25


def test_parse_distance_ft_handles_assist_suffix():
    assert _parse_distance_ft("Pascal Siakam makes 27-foot three point jumper (Dennis Schroder assists)") == 27


def test_parse_distance_ft_returns_none_for_rim_shots():
    assert _parse_distance_ft("Pascal Siakam misses driving layup") is None
    assert _parse_distance_ft("Precious Achiuwa makes dunk") is None


def test_parse_distance_ft_handles_empty_or_none_text():
    assert _parse_distance_ft("") is None
    assert _parse_distance_ft(None) is None


# ---- extract_shots ---------------------------------------------------

def _fake_summary(plays, raptors_home=True):
    return {
        "header": {
            "competitions": [{
                "competitors": [
                    {"team": {"id": RAPTORS_TEAM_ID}, "homeAway": "home" if raptors_home else "away"},
                    {"team": {"id": OPPONENT_TEAM_ID}, "homeAway": "away" if raptors_home else "home"},
                ]
            }]
        },
        "plays": plays,
    }


def _shot_play(team_id=RAPTORS_TEAM_ID, text="Player makes 20-foot jumper", made=True,
                x=25, y=10, clock="10:30", away_score=10, home_score=12, points=2):
    return {
        "shootingPlay": True,
        "team": {"id": team_id},
        "text": text,
        "scoringPlay": made,
        "coordinate": {"x": x, "y": y},
        "clock": {"displayValue": clock},
        "awayScore": away_score,
        "homeScore": home_score,
        "pointsAttempted": points,
        "period": {"number": 1},
    }


def test_only_raptors_shooting_plays_are_kept():
    plays = [
        _shot_play(team_id=RAPTORS_TEAM_ID),
        _shot_play(team_id=OPPONENT_TEAM_ID),
        {"shootingPlay": False, "team": {"id": RAPTORS_TEAM_ID}, "text": "Rebound"},
    ]
    rows = extract_shots(_fake_summary(plays), game_id="1")
    assert len(rows) == 1


def test_free_throws_are_excluded():
    plays = [_shot_play(text="Player makes free throw 1 of 2")]
    rows = extract_shots(_fake_summary(plays), game_id="1")
    assert rows == []


def test_score_margin_from_raptors_perspective_when_home():
    plays = [_shot_play(home_score=12, away_score=10)]
    rows = extract_shots(_fake_summary(plays, raptors_home=True), game_id="1")
    assert rows[0]["score_margin"] == 2  # 12 - 10


def test_score_margin_from_raptors_perspective_when_away():
    plays = [_shot_play(home_score=12, away_score=10)]
    rows = extract_shots(_fake_summary(plays, raptors_home=False), game_id="1")
    assert rows[0]["score_margin"] == -2  # 10 - 12


def test_clock_is_parsed_into_seconds_remaining():
    plays = [_shot_play(clock="9:45")]
    rows = extract_shots(_fake_summary(plays), game_id="1")
    assert rows[0]["clock_remaining_s"] == 9 * 60 + 45


def test_malformed_clock_falls_back_to_none():
    plays = [_shot_play(clock="--")]
    rows = extract_shots(_fake_summary(plays), game_id="1")
    assert rows[0]["clock_remaining_s"] is None


def test_made_flag_reflects_scoring_play():
    plays = [_shot_play(made=True), _shot_play(made=False)]
    rows = extract_shots(_fake_summary(plays), game_id="1")
    assert rows[0]["made"] is True
    assert rows[1]["made"] is False


def test_returns_empty_list_when_raptors_not_in_game():
    summary = {
        "header": {"competitions": [{"competitors": [
            {"team": {"id": "99"}, "homeAway": "home"},
            {"team": {"id": OPPONENT_TEAM_ID}, "homeAway": "away"},
        ]}]},
        "plays": [_shot_play()],
    }
    assert extract_shots(summary, game_id="1") == []
