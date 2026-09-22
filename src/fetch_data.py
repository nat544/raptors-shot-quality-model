"""
Pulls Toronto Raptors play-by-play data from ESPN's public API and extracts
shooting plays (shot attempts) into a flat list of dicts.

No API key needed. Endpoints used:
  - team schedule: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule?season={year}
  - game summary (play-by-play): https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}

Schema verified directly against the live API (Sept 2026). NBA's own stats.nba.com
API blocks most cloud/datacenter IPs at the network level, which is why this uses
ESPN's API instead.

RAPTORS_TEAM_ID = 28 (ESPN's internal id for the Toronto Raptors).
"""

import re
import requests

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
RAPTORS_TEAM_ID = "28"

DISTANCE_RE = re.compile(r"(\d+)-foot")


def get_team_schedule(team_id: str, season: int) -> list[dict]:
    """season is the ending year of the season, e.g. 2024 for the 2023-24 season."""
    resp = requests.get(f"{BASE_URL}/teams/{team_id}/schedule", params={"season": season}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    games = []
    for event in data.get("events", []):
        games.append({
            "game_id": event.get("id"),
            "date": event.get("date"),
            "name": event.get("name"),
        })
    return games


def get_game_summary(event_id: str) -> dict:
    resp = requests.get(f"{BASE_URL}/summary", params={"event": event_id}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _parse_distance_ft(text: str):
    m = DISTANCE_RE.search(text or "")
    return int(m.group(1)) if m else None


def extract_shots(summary: dict, game_id: str) -> list[dict]:
    """
    Flattens a game summary's play-by-play into one row per shot attempt by
    the Raptors, with the score margin (Raptors minus opponent) at the moment
    of the shot.
    """
    plays = summary.get("plays", [])

    # Figure out whether the Raptors were home or away this game, so we know
    # which of awayScore/homeScore is "theirs".
    header = summary.get("header", {})
    competitors = header.get("competitions", [{}])[0].get("competitors", [])
    raptors_is_home = None
    for c in competitors:
        if c.get("team", {}).get("id") == RAPTORS_TEAM_ID:
            raptors_is_home = (c.get("homeAway") == "home")
    if raptors_is_home is None:
        return []  # couldn't identify Raptors in this game -- skip

    rows = []
    for play in plays:
        if not play.get("shootingPlay"):
            continue
        if play.get("team", {}).get("id") != RAPTORS_TEAM_ID:
            continue  # only Raptors' own shots
        if "free throw" in (play.get("text") or "").lower():
            continue  # free throws aren't spatial shot attempts

        away_score = play.get("awayScore", 0)
        home_score = play.get("homeScore", 0)
        raptors_score, opp_score = (home_score, away_score) if raptors_is_home else (away_score, home_score)

        coord = play.get("coordinate", {})
        clock_display = play.get("clock", {}).get("displayValue", "0:00")
        try:
            minutes, seconds = clock_display.split(":")
            clock_remaining_s = int(minutes) * 60 + int(seconds)
        except ValueError:
            clock_remaining_s = None

        rows.append({
            "game_id": game_id,
            "play_id": play.get("id"),
            "period": play.get("period", {}).get("number"),
            "clock_remaining_s": clock_remaining_s,
            "x": coord.get("x"),
            "y": coord.get("y"),
            "distance_ft": _parse_distance_ft(play.get("text", "")),
            "points_attempted": play.get("pointsAttempted") or play.get("scoreValue") or 2,
            "made": bool(play.get("scoringPlay")),
            "score_margin": raptors_score - opp_score,
            "text": play.get("text"),
        })
    return rows


def fetch_season_shots(season: int, max_games: int = None) -> list[dict]:
    """Fetch every Raptors shot attempt for a given season (e.g. 2024 = 2023-24 season)."""
    games = get_team_schedule(RAPTORS_TEAM_ID, season)
    if max_games:
        games = games[:max_games]

    all_shots = []
    for g in games:
        try:
            summary = get_game_summary(g["game_id"])
            all_shots.extend(extract_shots(summary, g["game_id"]))
        except requests.exceptions.RequestException as e:
            print(f"skipping game {g['game_id']} ({g['name']}): {e}")
    return all_shots


if __name__ == "__main__":
    games = get_team_schedule(RAPTORS_TEAM_ID, 2024)
    print(f"Found {len(games)} games in the 2023-24 season")
    summary = get_game_summary(games[0]["game_id"])
    shots = extract_shots(summary, games[0]["game_id"])
    print(f"Extracted {len(shots)} Raptors shot attempts from game {games[0]['name']}")
    print(shots[0])
