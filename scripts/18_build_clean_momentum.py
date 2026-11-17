from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data/processed/full_season/event_price"
OUT = BASE / "results/full_season/q1_momentum_clean"
OUT.mkdir(parents=True, exist_ok=True)

print("=== STEP 10: CLEAN MOMENTUM DATASET ===", flush=True)

events = pd.read_parquet(DATA / "scoring_events.parquet")
events["espn_game_id"] = events["espn_game_id"].astype(str)
events["event_time"] = pd.to_datetime(
    events["event_time"], utc=True
)

events = events.sort_values(
    ["espn_game_id", "sequence_number", "event_time"]
).reset_index(drop=True)

# Identify games with implausibly long scoring-event spans.
spans = events.groupby("espn_game_id")["event_time"].agg(
    ["min", "max"]
)
spans["span_minutes"] = (
    spans["max"] - spans["min"]
).dt.total_seconds() / 60

excluded_games = set(
    spans.index[spans["span_minutes"] > 300]
)

print("Games excluded for >5h span:", len(excluded_games))

events = events[
    ~events["espn_game_id"].isin(excluded_games)
].copy()

# Calculate scoring changes in ESPN sequence order.
events["prev_home_score"] = (
    events.groupby("espn_game_id")["home_score"]
    .shift(fill_value=0)
)
events["prev_away_score"] = (
    events.groupby("espn_game_id")["away_score"]
    .shift(fill_value=0)
)

events["home_added"] = (
    events["home_score"] - events["prev_home_score"]
)
events["away_added"] = (
    events["away_score"] - events["prev_away_score"]
)

home_valid = (
    events["home_added"].between(1, 4)
    & events["away_added"].eq(0)
)

away_valid = (
    events["away_added"].between(1, 4)
    & events["home_added"].eq(0)
)

events["scoring_side"] = np.select(
    [home_valid, away_valid],
    ["home", "away"],
    default="invalid"
)

events["points_added"] = np.select(
    [home_valid, away_valid],
    [events["home_added"], events["away_added"]],
    default=0
)

# Invalid events act as run boundaries.
events["previous_side"] = (
    events.groupby("espn_game_id")["scoring_side"].shift()
)

events["new_run"] = (
    events["scoring_side"].ne(events["previous_side"])
    | events["scoring_side"].eq("invalid")
)

events["run_number"] = (
    events.groupby("espn_game_id")["new_run"].cumsum()
)

valid = events[
    events["scoring_side"].isin(["home", "away"])
].copy()

runs = (
    valid.groupby(
        ["espn_game_id", "run_number"], as_index=False
    )
    .agg(
        season=("season", "first"),
        run_team=("scoring_side", "first"),
        run_points=("points_added", "sum"),
        scoring_plays=("event_id", "count"),
        run_start_time=("event_time", "first"),
        run_end_time=("event_time", "last"),
        start_period=("period", "first"),
        end_period=("period", "last"),
        score_diff_after=("score_diff", "last")
    )
)

runs = runs[runs["run_points"] >= 4].copy()

runs["run_duration_seconds"] = (
    runs["run_end_time"] - runs["run_start_time"]
).dt.total_seconds()

runs["cross_quarter"] = (
    runs["start_period"] != runs["end_period"]
)

# Keep the primary sample within a plausible time range.
runs["valid_duration"] = (
    runs["run_duration_seconds"].between(0, 1200)
)

# Load price-minute dataset.
prices = ds.dataset(
    str(DATA / "price_minutes"),
    format="parquet"
).to_table(
    columns=[
        "espn_game_id",
        "minute_bucket",
        "home_win_probability"
    ]
).to_pandas()

prices["espn_game_id"] = prices["espn_game_id"].astype(str)
prices["minute_bucket"] = pd.to_datetime(
    prices["minute_bucket"], utc=True
)

prices = prices.drop_duplicates(
    ["espn_game_id", "minute_bucket"]
)

runs["pre_minute"] = (
    runs["run_start_time"].dt.floor("min")
    - pd.Timedelta(minutes=1)
)

runs["post_1m_minute"] = (
    runs["run_end_time"].dt.floor("min")
    + pd.Timedelta(minutes=1)
)

runs["post_5m_minute"] = (
    runs["run_end_time"].dt.floor("min")
    + pd.Timedelta(minutes=5)
)

for target, output in [
    ("pre_minute", "price_before"),
    ("post_1m_minute", "price_after_1m"),
    ("post_5m_minute", "price_after_5m")
]:
    temp = prices.rename(columns={
        "minute_bucket": target,
        "home_win_probability": output
    })

    runs = runs.merge(
        temp,
        on=["espn_game_id", target],
        how="left",
        validate="many_to_one"
    )

direction = np.where(
    runs["run_team"] == "home", 1, -1
)

runs["reaction_1m"] = direction * (
    runs["price_after_1m"] - runs["price_before"]
)

runs["reaction_5m"] = direction * (
    runs["price_after_5m"] - runs["price_before"]
)

runs["near_extreme_price"] = (
    runs["price_after_5m"].le(.01)
    | runs["price_after_5m"].ge(.99)
)

runs["run_category"] = pd.cut(
    runs["run_points"],
    bins=[3, 6, 9, 14, np.inf],
    labels=["4-6", "7-9", "10-14", "15+"]
)

runs.to_parquet(
    OUT / "all_cleaned_runs.parquet",
    index=False
)

primary = runs[
    runs["valid_duration"]
    & runs["reaction_5m"].notna()
].copy()

primary.to_csv(
    OUT / "primary_sample.csv",
    index=False
)
primary.to_parquet(
    OUT / "primary_sample.parquet",
    index=False
)

def summarize(data):
    result = (
        data.groupby("run_category", observed=True)
        .agg(
            runs=("reaction_5m", "size"),
            mean_reaction=("reaction_5m", "mean"),
            median_reaction=("reaction_5m", "median")
        )
        .reset_index()
    )
    result["mean_reaction_pp"] = (
        result["mean_reaction"] * 100
    )
    result["median_reaction_pp"] = (
        result["median_reaction"] * 100
    )
    return result

summary = summarize(primary)
summary.to_csv(
    OUT / "clean_momentum_summary.csv",
    index=False
)

non_extreme = primary[
    ~primary["near_extreme_price"]
]
non_extreme_summary = summarize(non_extreme)
non_extreme_summary.to_csv(
    OUT / "non_extreme_summary.csv",
    index=False
)

quarter_summary = (
    primary.groupby("end_period")
    .agg(
        runs=("reaction_5m", "size"),
        mean_reaction=("reaction_5m", "mean")
    )
    .reset_index()
)
quarter_summary["mean_reaction_pp"] = (
    quarter_summary["mean_reaction"] * 100
)
quarter_summary.to_csv(
    OUT / "quarter_summary.csv",
    index=False
)

print("\n=== CLEANING REPORT ===")
print("Original scoring events:", 305651)
print("Remaining scoring events:", len(events))
print("Invalid scoring events:",
      (events["scoring_side"] == "invalid").sum())
print("Runs >=4 points:", len(runs))
print("Primary valid runs:", len(primary))
print("Primary near-extreme prices:",
      primary["near_extreme_price"].sum())
print("Cross-quarter runs:",
      primary["cross_quarter"].sum())

print("\n=== CLEAN MOMENTUM SUMMARY ===")
print(
    summary[
        ["run_category", "runs",
         "mean_reaction_pp", "median_reaction_pp"]
    ].to_string(index=False)
)

print("\n=== NON-EXTREME PRICE SUMMARY ===")
print(
    non_extreme_summary[
        ["run_category", "runs", "mean_reaction_pp"]
    ].to_string(index=False)
)

print("\n=== QUARTER SUMMARY ===")
print(
    quarter_summary[
        ["end_period", "runs", "mean_reaction_pp"]
    ].to_string(index=False)
)

print("\nOutputs saved to:", OUT)
