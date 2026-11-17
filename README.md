# NBA Polymarket Analytics with Apache Spark

**Big Data Systems Final Project**

## Overview

This project investigates how NBA game events and scoring momentum relate to price movements in Polymarket prediction markets using Apache Spark.

We integrate ESPN NBA play-by-play data with minute-level Polymarket prices to examine scoring momentum, market volatility, and short-term price adjustments across the 2024–25 and 2025–26 NBA seasons.

## Research Questions

1. **Scoring Momentum:** How does the magnitude of consecutive scoring runs relate to changes in prediction-market win probabilities?
2. **Market Volatility:** How do game competitiveness and scoring conditions relate to prediction-market volatility?
3. **Price Stabilization:** How do Polymarket prices adjust during the first five minutes following NBA scoring events?
4. **Big Data Systems:** How can Apache Spark support large-scale processing and analysis of NBA play-by-play and prediction-market data?

## Technologies

- Python
- Apache Spark (PySpark)
- Pandas, NumPy, SciPy, Statsmodels
- Apache Parquet and PyArrow
- ESPN NBA API
- Polymarket NBA Dataset
- Matplotlib

## Dataset

The project analyzes NBA games from the 2024–25 and 2025–26 seasons.

| Dataset | Size |
|---|---:|
| ESPN games downloaded | 2,634 |
| ESPN–Polymarket matched games | 2,575 |
| Polymarket price observations | 82.2 million |
| Polymarket trade observations | 7.24 million |
| Scoring events processed | 305,651 |
| Valid scoring events for Q3 | 297,633 |
| Clean scoring runs for Q1 | 54,452 |

The data pipeline matches ESPN games to Polymarket NBA moneyline markets and aligns scoring events with minute-level market prices.

## Data Pipeline

### 1. Data Preparation and Matching

- `11_prepare_full_season_markets.py` — Prepare full-season Polymarket markets.
- `12_fetch_full_season_espn.py` — Download ESPN NBA play-by-play data.
- `13_match_full_season_games.py` — Match ESPN games with Polymarket markets.
- `14_resolve_duplicate_markets.py` — Resolve duplicate market matches.
- `15_validate_full_season_prices.py` — Validate minute-level market prices.
- `16_build_event_price_dataset.py` — Build the scoring-event and price dataset.

### 2. Q1: Scoring Momentum Analysis

- `18_build_clean_momentum.py` — Identify and clean consecutive scoring runs.
- `20_q1_final_regression.py` — Estimate regression models with game-clustered standard errors and robustness checks.

### 3. Q3: Price Stabilization Analysis

- `21_q3_event_study.py` — Analyze price reactions at 1-, 2-, 3-, and 5-minute horizons.
- `22_q3_stabilization_test.py` — Test changes in short-term price movement using game-clustered standard errors.

## Preliminary Results

### Q1: Scoring Momentum

Larger consecutive scoring runs are associated with greater directional changes in Polymarket-implied win probabilities.

| Scoring Run | Average Price Reaction |
|---|---:|
| 4–6 points | 3.23 pp |
| 7–9 points | 8.42 pp |
| 10–14 points | 14.35 pp |
| 15+ points | 21.49 pp |

In a regression excluding runs whose post-event observation occurs after the final recorded scoring play, each additional point in a scoring run is associated with approximately **1.70 percentage points** of additional directional price movement, controlling for selected game conditions.

This is an observational association, not a causal estimate.

### Q3: Price Stabilization

The event study shows that absolute price movements generally become smaller during subsequent observation intervals following scoring events.

For 5,547 isolated scoring events in the first three quarters:

| Interval | Mean Absolute Price Movement |
|---|---:|
| +1 to +2 minutes | 1.122 pp |
| +2 to +3 minutes | 0.634 pp |

The average absolute price movement decreases by **43.49%**, with a statistically significant paired difference using game-clustered standard errors (p < 0.001).

These results suggest short-term price adjustment becomes smaller following scoring events, but do not establish complete information incorporation or market efficiency.

## Results

Analysis outputs are stored in:

- `results/full_season/q1_momentum_clean/` — Scoring momentum summaries, regression results, and robustness checks.
- `results/full_season/q3_price_settling/` — Event study results, statistical tests, and figures.
- `results/figures/` — Additional visualizations.

## Data Sources

**Polymarket NBA Dataset:**  
https://huggingface.co/datasets/epinnock/polymarket-nba

**ESPN NBA API:**  
Used for NBA schedules, game information, and play-by-play events.

Raw and processed datasets are excluded from GitHub due to their size.

## Limitations

- Market prices are observed at minute-level resolution rather than continuously.
- Multiple scoring events and other game developments may affect overlapping price windows.
- Scoring momentum estimates describe associations rather than causal effects.
- Short-term reductions in price movement do not necessarily imply market efficiency.
- Additional robustness checks and systems performance analysis are ongoing.

## Project Status

- Full-season data ingestion and matching: Complete
- Event–price dataset construction: Complete
- Q1 scoring momentum analysis: Complete
- Q3 event study and stabilization tests: Complete
- Additional robustness checks and final reporting: In progress