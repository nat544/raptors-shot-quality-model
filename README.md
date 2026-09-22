# Toronto Raptors Shot Quality Model — Spatio-Temporal Shot Model

A shot-quality model for Toronto Raptors field-goal attempts, built to
demonstrate **spatio-temporal modeling**: predicting an outcome (was this shot
made?) from *where* on the court a shot was taken combined with *when* in the
game it happened and the context at that moment (score margin, clutch time).

Data comes from ESPN's public NBA API. Note: NBA's own stats.nba.com API blocks most cloud/datacenter
network ranges at the network level (confirmed while building this — direct
requests timed out even with full browser headers), which is why this project
uses ESPN's API instead. 

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

## Results (sample run, ~1,800 shots from 20 games)

- ROC-AUC: ~0.67
- Make rate in sample: ~46% (consistent with real NBA field-goal percentages)

**On the ROC-AUC**: this is meaningfully lower than what NHL xG
project achieves (~0.86), and that's expected. Shot location and game time explain only part of whether an NBA
shot goes in — the biggest driver of make probability is defender proximity
and contest level, which isn't available from box-score-level play-by-play.
Capturing that requires real player-tracking data.

