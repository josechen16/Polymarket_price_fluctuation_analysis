# NBA Polymarket Analytics with Apache Spark

Big Data Systems Final Project

## Overview

This project studies the relationship between NBA game events and
Polymarket prediction-market price movements using Apache Spark.

## Research Questions

1. How do scoring events and score differentials relate to market prices?
2. Are close games associated with higher market volatility?
3. How quickly do prediction-market prices respond to scoring events?

## Technologies

- Python
- Apache Spark (PySpark)
- Apache Parquet
- ESPN NBA API
- Polymarket NBA Dataset

## Pipeline

1. `01_prepare_markets.py` - Prepare NBA moneyline markets
2. `02_fetch_espn.py` - Download ESPN play-by-play
3. `03_prepare_events.py` - Transform NBA events using Spark
4. `04_join_prices.py` - Join NBA events with market prices

## Current Progress

- 16 NBA games matched
- 8,385 play-by-play events processed
- 2,102 scoring events identified
- 26,880 minute-level market price records selected
- 2,102 scoring events matched with before/after prices

## Data

Raw and processed datasets are excluded from GitHub due to their size.

Polymarket dataset:
https://huggingface.co/datasets/epinnock/polymarket-nba

ESPN NBA API is used for game schedules and play-by-play data.

## Status

Data ingestion, preprocessing, and initial event-price joining are complete.

Research analysis and performance benchmarking are in progress.
