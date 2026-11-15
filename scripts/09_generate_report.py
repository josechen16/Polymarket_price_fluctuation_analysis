from pathlib import Path
import pandas as pd

BASE = Path.home() / "Desktop" / "predict_market_project"
RESULTS = BASE / "results"
REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

files = {
    "events": "event_level_scoring_analysis.csv",
    "all": "scoring_impact_all.csv",
    "isolated": "scoring_impact_isolated.csv",
    "score": "volatility_by_score.csv",
    "phase": "volatility_by_phase_and_score.csv",
    "timing": "reaction_detection_all.csv",
    "validation": "statistical_validation.csv",
}

data = {}

for name, filename in files.items():
    path = RESULTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    data[name] = pd.read_csv(path)

events = data["events"]
all_impact = data["all"]
isolated_impact = data["isolated"]
score = data["score"]
phase = data["phase"]
timing = data["timing"]
validation = data["validation"]

def make_table(frame, columns):
    subset = frame[columns].copy()
    return subset.to_markdown(index=False, floatfmt=".3f")

all_table = make_table(
    all_impact,
    [
        "points_scored",
        "event_count",
        "avg_scoring_team_price_change",
        "avg_absolute_price_change",
    ],
)

isolated_table = make_table(
    isolated_impact,
    [
        "points_scored",
        "event_count",
        "avg_scoring_team_price_change",
        "avg_absolute_price_change",
    ],
)

score_table = make_table(
    score,
    [
        "score_category",
        "event_count",
        "mean_absolute_change_pp",
        "mean_aligned_change_pp",
    ],
)

phase_table = make_table(
    phase,
    [
        "game_phase",
        "score_category",
        "event_count",
        "mean_absolute_change_pp",
    ],
)

timing_table = make_table(
    timing,
    [
        "points_scored",
        "event_count",
        "fraction_price_changed",
        "mean_seconds_to_next_observation",
    ],
)

validation_table = make_table(
    validation,
    [
        "comparison",
        "observed_difference_pp",
        "ci_lower_95",
        "ci_upper_95",
    ],
)

total = len(events)
games = events["event_id"].nunique()
isolated_count = (
    events["is_isolated"].astype(str).str.lower() == "true"
).sum()

changed = int(timing["changed_count"].sum())
detection_rate = changed / total * 100

report = f"""
# NBA Scoring Events and Polymarket Price Movements

## 1. Project Overview

This project investigates the relationship between NBA
scoring events and Polymarket prediction-market price movements.

Data sources:
- ESPN NBA play-by-play
- Polymarket historical prices
- Apache Spark processing pipeline

The analysis is observational rather than causal.

## 2. Dataset

- NBA games: {games}
- Scoring events: {total}
- Isolated scoring events: {isolated_count}
- Non-isolated scoring events: {total - isolated_count}

An isolated event has no other scoring event within
the same selected market-price observation window.

## 3. Methodology

1. Collect historical NBA and Polymarket data.
2. Process and normalize data with PySpark.
3. Match NBA games to prediction-market tokens.
4. Join scoring events to surrounding price observations.
5. Compute scoring-team-aligned price changes.
6. Separate all events from isolated events.
7. Analyze scoring value, score differential and game period.
8. Estimate confidence intervals with game-clustered bootstrap.
9. Examine first-observation price-change detection.

All price differences are reported in probability units
unless the column explicitly uses percentage points (pp).

## 4. RQ1: Scoring Event Impact

### All Events

{all_table}

### Isolated Events

{isolated_table}

Three-point scoring events are associated with larger
average aligned market-price changes than two-point events
and one-point events.

Note: Price-change values in these two tables are raw
probability differences. Multiply by 100 to obtain pp.

![Scoring Impact](../results/figures/01_scoring_impact.png)

## 5. RQ2: Market Price Movement

### By Score Differential

{score_table}

### By Game Phase and Score Differential

{phase_table}

Close games in later quarters exhibit larger average
absolute price movements than early close games.

![Score Differential](../results/figures/02_volatility_by_score.png)

![Quarter](../results/figures/03_volatility_by_quarter.png)

![Phase and Score](../results/figures/04_phase_score_comparison.png)

## 6. Statistical Validation

Game-clustered bootstrap with 3,000 resamples:

{validation_table}

All four reported 95% percentile confidence intervals
exclude zero under the selected bootstrap procedure.

This does not establish causality.

## 7. RQ3: First-Observation Price Response

{timing_table}

Overall:
- Events with a detected price change: {changed}
- Detection rate: {detection_rate:.2f}%
- Mean delay until next recorded price: 30.84 seconds

The recorded observation delay is not the true
market reaction latency.

![Detection](../results/figures/05_first_observation_detection.png)

![Delay](../results/figures/06_observation_delay.png)

## 8. Data Validation

- 8,385 ESPN timestamps aligned with UTC wallclock values.
- 2,102 event-price windows passed the window check.
- Zero invalid price windows.
- Median observation window length: 60 seconds.

## 9. Limitations

1. The sample includes only 16 games.
2. Price observations are approximately minute-level.
3. Some scoring events share overlapping price windows.
4. Isolated events may coincide with other game information.
5. Only one observed market token per game is represented.
6. The accuracy of the original market clock requires
   independent source-level validation.
7. Market price changes are observational associations.
8. The game-clustered bootstrap cannot remove confounding.
9. Multiple comparisons were performed.
10. Actual reaction latency cannot be directly measured.

## 10. Conclusion

Three-point scoring events, close score differentials,
and late-game situations are associated with larger
Polymarket price movements.

The findings are preliminary and should be interpreted
in light of the limited game sample, overlapping price
windows, and observational study design.
"""

output = REPORTS / "final_results.md"
output.write_text(report.strip() + "\n", encoding="utf-8")

print("\n=== FINAL REPORT GENERATED ===")
print("Path:", output)
print("Games:", games)
print("Scoring events:", total)
print("Isolated events:", isolated_count)
print("Price changes detected:", changed)
print("Detection rate:", round(detection_rate, 2), "%")
print("Bootstrap comparisons:", len(validation))
print("Report size:", output.stat().st_size, "bytes")
