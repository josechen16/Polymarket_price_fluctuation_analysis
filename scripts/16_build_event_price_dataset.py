from pathlib import Path
import json
import os
import time
import pandas as pd

os.environ["TZ"] = "UTC"
if hasattr(time, "tzset"):
    time.tzset()

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "data/processed/full_season"
PRICE_ROOT = BASE / "data/processed/spark_compatible_prices"
ESPN_ROOT = BASE / "data/raw/nba/full_season/games"
OUT = ROOT / "event_price"
OUT.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("NBA_Event_Price_Join_UTC_Fixed")
    .master("local[*]")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.sql.shuffle.partitions", "24")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("=== STEP 7: EVENT-PRICE JOIN ===", flush=True)

games_pd = pd.read_parquet(
    ROOT / "matched_games_final.parquet"
)

games_pd["espn_game_id"] = (
    games_pd["espn_game_id"].astype(str)
)

games_pd["game_time"] = pd.to_datetime(
    games_pd["game_time"], utc=True
)

games_pd["game_start_epoch"] = (
    games_pd["game_time"].astype("int64") // 10**9
)

game_lookup = {
    str(row["espn_game_id"]): row
    for _, row in games_pd.iterrows()
}

records = []
missing_files = []
missing_wallclock = 0

for game_id, game in game_lookup.items():
    path = ESPN_ROOT / f"{game_id}.json"

    if not path.exists():
        missing_files.append(game_id)
        continue

    with path.open() as f:
        data = json.load(f)

    plays = data.get("plays", [])

    if isinstance(plays, dict):
        plays = plays.get("plays", [])

    if not isinstance(plays, list):
        continue

    for index, play in enumerate(plays):
        if not play.get("scoringPlay"):
            continue

        wallclock = play.get("wallclock")
        event_time = pd.to_datetime(
            wallclock, utc=True, errors="coerce"
        )

        if pd.isna(event_time):
            missing_wallclock += 1
            continue

        home_score = play.get("homeScore")
        away_score = play.get("awayScore")

        if home_score is None or away_score is None:
            continue

        records.append({
            "espn_game_id": game_id,
            "season": str(game["season"]),
            "event_id": str(
                play.get("id") or f"{game_id}_{index}"
            ),
            "sequence_number": int(
                play.get("sequenceNumber") or index
            ),
            "event_time": event_time,
            "event_epoch": int(event_time.timestamp()),
            "period": int(
                (play.get("period") or {}).get("number") or 0
            ),
            "clock": (
                (play.get("clock") or {})
                .get("displayValue", "")
            ),
            "event_description": play.get("text", ""),
            "score_value": int(play.get("scoreValue") or 0),
            "home_score": int(home_score),
            "away_score": int(away_score),
            "score_diff": int(home_score) - int(away_score)
        })

events_pd = pd.DataFrame(records)

if events_pd.empty:
    raise RuntimeError("No scoring events extracted")

events_pd = events_pd.sort_values([
    "espn_game_id", "event_epoch", "sequence_number"
])

events_pd.to_parquet(
    OUT / "scoring_events.parquet",
    index=False
)

print("\n=== ESPN EVENT EXTRACTION ===", flush=True)
print("Matched games:", len(games_pd), flush=True)
print("Scoring events:", len(events_pd), flush=True)
print(
    "Games with scoring events:",
    events_pd["espn_game_id"].nunique(),
    flush=True
)
print("Missing ESPN files:", len(missing_files), flush=True)
print(
    "Missing scoring wallclock:",
    missing_wallclock,
    flush=True
)

# Convert game times using UTC epoch seconds.
game_columns = [
    "espn_game_id",
    "condition_id",
    "home_token_id",
    "away_token_id",
    "game_start_epoch"
]

games_spark_pd = games_pd[game_columns].copy()

games_spark_pd["game_start_epoch"] = (
    games_spark_pd["game_start_epoch"].astype("int64")
)

games = spark.createDataFrame(games_spark_pd)

games = games.withColumn(
    "game_time_utc",
    F.timestamp_seconds(F.col("game_start_epoch"))
)

print("\n=== UTC SANITY CHECK ===", flush=True)

sample = (
    games.filter(F.col("espn_game_id") == "401704628")
    .select("espn_game_id", "game_time_utc")
    .collect()
)

for row in sample:
    print(
        "ESPN game:", row["espn_game_id"],
        "Spark UTC:", row["game_time_utc"],
        flush=True
    )

# Load compatible price files.
prices = spark.read.parquet(
    str(PRICE_ROOT / "prices-2024-25.parquet"),
    str(PRICE_ROOT / "prices-2025-26.parquet")
).select(
    "condition_id",
    "token_id",
    "timestamp",
    "price",
    "fidelity"
)

joined = prices.join(
    F.broadcast(games),
    on="condition_id",
    how="inner"
)

joined = joined.filter(
    F.col("price").between(0.0, 1.0)
)

joined = joined.filter(
    (F.col("token_id") == F.col("home_token_id"))
    |
    (F.col("token_id") == F.col("away_token_id"))
)

# Use actual UTC game start, with a 10-minute buffer.
joined = joined.filter(
    (F.col("timestamp") >=
     F.expr("game_time_utc - INTERVAL 10 MINUTES"))
    &
    (F.col("timestamp") <
     F.expr("game_time_utc + INTERVAL 4 HOURS"))
)

joined = joined.withColumn(
    "home_win_probability",
    F.when(
        F.col("token_id") == F.col("home_token_id"),
        F.col("price")
    ).otherwise(1.0 - F.col("price"))
)

joined = joined.withColumn(
    "minute_bucket",
    F.date_trunc("minute", F.col("timestamp"))
)

window = Window.partitionBy(
    "espn_game_id", "minute_bucket"
).orderBy(
    F.when(
        F.col("fidelity") == "minute", 1
    ).otherwise(0).desc(),
    F.col("timestamp").desc()
)

price_minutes = (
    joined
    .withColumn("rn", F.row_number().over(window))
    .filter(F.col("rn") == 1)
    .select(
        "espn_game_id",
        "minute_bucket",
        "timestamp",
        "home_win_probability"
    )
)

price_path = OUT / "price_minutes"

price_minutes.write.mode("overwrite").parquet(
    str(price_path)
)

# Read materialized prices to avoid rescanning raw data.
price_minutes = spark.read.parquet(str(price_path))

price_count = price_minutes.count()

print("\n=== PRICE DATASET ===", flush=True)
print("Price-minute rows:", price_count, flush=True)

sample_prices = (
    price_minutes
    .filter(F.col("espn_game_id") == "401704628")
    .agg(
        F.count("*").alias("rows"),
        F.min("minute_bucket").alias("first_price"),
        F.max("minute_bucket").alias("last_price")
    )
    .collect()[0]
)

print(
    "Lakers sample:",
    sample_prices.asDict(),
    flush=True
)

if price_count < 100000:
    raise RuntimeError(
        "Price-minute count is unexpectedly low. "
        "Check timestamp alignment before continuing."
    )

# Convert event times using UTC epoch seconds.
events_spark_pd = events_pd.drop(
    columns=["event_time"]
).copy()

events = spark.createDataFrame(events_spark_pd)

events = events.withColumn(
    "event_time",
    F.timestamp_seconds(F.col("event_epoch"))
)

events = events.withColumn(
    "event_minute",
    F.date_trunc("minute", F.col("event_time"))
)

# Fixed-minute event study:
# pre = previous minute
# post = event minute + 1, 2, 3, 5
offsets = [
    (-1, "pre"),
    (1, "post_1m"),
    (2, "post_2m"),
    (3, "post_3m"),
    (5, "post_5m")
]

result = events

for offset, label in offsets:
    p = price_minutes.select(
        F.col("espn_game_id").alias("p_game_id"),
        F.col("minute_bucket").alias("p_minute"),
        F.col("home_win_probability").alias(
            f"price_{label}"
        )
    )

    if offset < 0:
        target = F.expr(
            f"event_minute - INTERVAL {abs(offset)} MINUTES"
        )
    else:
        target = F.expr(
            f"event_minute + INTERVAL {offset} MINUTES"
        )

    result = (
        result.join(
            p,
            (result["espn_game_id"] == p["p_game_id"])
            & (target == p["p_minute"]),
            "left"
        )
        .drop("p_game_id", "p_minute")
    )

for offset, label in offsets:
    if label == "pre":
        continue

    result = result.withColumn(
        f"delta_{label}",
        F.col(f"price_{label}") - F.col("price_pre")
    )

output_path = OUT / "scoring_event_price_join"

result.write.mode("overwrite").parquet(
    str(output_path)
)

result = spark.read.parquet(str(output_path))

summary = result.agg(
    F.count("*").alias("total_events"),
    F.count("price_pre").alias("events_with_pre_price"),
    F.count("price_post_1m").alias("events_with_post_1m"),
    F.count("price_post_5m").alias("events_with_post_5m"),
    F.count("delta_post_5m").alias("complete_5m_pairs")
).collect()[0]

print("\n=== EVENT-PRICE JOIN REPORT ===", flush=True)

for key, value in summary.asDict().items():
    print(f"{key}: {value}", flush=True)

print("\nOutputs:", flush=True)
print(OUT / "scoring_events.parquet")
print(price_path)
print(output_path)

spark.stop()
