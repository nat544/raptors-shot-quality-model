# Toronto Raptors Shot Quality Model — Spatio-Temporal Shot Model

A shot-quality model for Toronto Raptors field-goal attempts, built to
demonstrate **spatio-temporal modeling**: predicting an outcome (was this shot
made?) from *where* on the court a shot was taken combined with *when* in the
game it happened and the context at that moment (score margin, clutch time).

Data comes from ESPN's public NBA API — no API key or paid data source
required. Note: NBA's own stats.nba.com API blocks most cloud/datacenter
network ranges at the network level (confirmed while building this — direct
requests timed out even with full browser headers), which is why this project
uses ESPN's API instead. That's the same workaround the open sports-analytics
community commonly reaches for.

## Pipeline

1. **`src/fetch_data.py`** — pulls a season's worth of Raptors games from
   ESPN's schedule endpoint, then pulls each game's play-by-play and extracts
   every Raptors shot attempt (excluding free throws, which aren't spatial
   shots), tracking the score margin as it goes.
2. **`src/features.py`** — engineers the spatio-temporal features:
   - *Spatial:* shot location (x, y) and distance in feet, parsed directly
     from ESPN's play-by-play text (near-rim shots like layups/dunks are
     imputed at 2 ft since ESPN's text doesn't call out a distance for those)
   - *Temporal:* elapsed game time, period, score margin, and a "clutch" flag
     (4th quarter/OT, under 5 minutes remaining) **at the moment of the shot**
3. **`src/model.py`** — trains a gradient-boosted classifier (XGBoost) to
   predict make probability, with standard evaluation metrics (ROC-AUC, log
   loss, Brier score).
4. **`src/visualize.py`** — plots shot locations colored by predicted make
   probability, and an expected-vs-actual cumulative scoring chart over game
   time.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook raptors_shot_quality_model.ipynb
```

The notebook pulls a small sample (20 games) by default so it runs in under a
minute — widen `MAX_GAMES` (up to 82) for a full-season, more robust sample.

## Testing

```bash
pytest tests/
```

27 unit tests cover the pure logic that doesn't require live network calls:
game-clock math across regulation and overtime, the rim-shot distance
imputation and clutch/three-point flags, the shot-distance text parser, and
the play-by-play extraction logic (team/free-throw filtering, score-margin
calculation, clock parsing) — all using small hand-built ESPN-shaped
fixtures rather than hitting the live API.

## Results (sample run, ~1,800 shots from 20 games)

- ROC-AUC: ~0.67
- Make rate in sample: ~46% (consistent with real NBA field-goal percentages)

**On the ROC-AUC**: this is meaningfully lower than what the companion NHL xG
project achieves (~0.86), and that's expected, not a shortcoming of the
pipeline. Shot location and game time explain only part of whether an NBA
shot goes in — the biggest driver of make probability is defender proximity
and contest level, which isn't available from box-score-level play-by-play.
Capturing that requires real player-tracking data (the kind pro teams collect
internally, and the kind referenced in job postings that ask for
"spatio-temporal tracking data" analysis). This model is an honest baseline
built entirely on public data, with that limitation stated plainly rather
than papered over.

## Why this project

Built to demonstrate hands-on **spatio-temporal modeling** — combining
spatial (location) and temporal (game-time/context) features into a single
predictive model — for a team MLSE actually owns, using data that is
genuinely public and reproducible.

## Possible extensions

- Pull a full season (`MAX_GAMES=82`) for a more robust model.
- Parse more shot-type detail from ESPN's play text (catch-and-shoot vs.
  pull-up, transition vs. half-court).
- Compare against a simple logistic-regression baseline (distance + angle
  only) to quantify how much the temporal/context features actually add.
- Aggregate per-player actual-vs-expected make rate to find over/under-
  performing shooters relative to shot quality (a "shooting skill" analysis).
