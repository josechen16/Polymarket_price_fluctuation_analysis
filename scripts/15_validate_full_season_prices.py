from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "data/processed/full_season"
PRICE_ROOT = BASE / "data/processed/spark_compatible_prices"
RESULTS = BASE / "results/full_season"
RESULTS.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("NBA_Full_Season_Price_Validation")
    .master("local[*]")
    .config("spark.sql.session.timeZone", "UTC")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

games_pd = pd.read_parquet(ROOT / "matched_games_final.parquet")

games_pd["game_time"] = (
    games_pd["game_time"]
    .dt.tz_convert("UTC")
    .dt.tz_localize(None)
)

columns = [
    "espn_game_id", "season", "game_time",
    "condition_id", "home_token_id", "away_token_id"
]

schema = StructType([
    StructField("espn_game_id", StringType()),
    StructField("season", StringType()),
    StructField("game_time", TimestampType()),
    StructField("condition_id", StringType()),
    StructField("home_token_id", StringType()),
    StructField("away_token_id", StringType())
])

games = spark.createDataFrame(games_pd[columns], schema=schema)

prices = spark.read.parquet(
    str(PRICE_ROOT / "prices-2024-25.parquet"),
    str(PRICE_ROOT / "prices-2025-26.parquet")
).select(
    "condition_id", "token_id", "timestamp",
    "price", "fidelity"
)

joined = prices.join(
    F.broadcast(games),
    on="condition_id",
    how="inner"
)

joined = joined.withColumn(
    "token_side",
    F.when(F.col("token_id") == F.col("home_token_id"), "home")
     .when(F.col("token_id") == F.col("away_token_id"), "away")
     .otherwise("unknown")
)

joined = joined.withColumn(
    "valid_price",
    F.col("price").between(0.0, 1.0)
)

joined = joined.withColumn(
    "in_game",
    (F.col("timestamp") >= F.col("game_time")) &
    (F.col("timestamp") < F.expr("game_time + INTERVAL 4 HOURS"))
)

joined = joined.withColumn(
    "minute_bucket",
    F.date_trunc("minute", F.col("timestamp"))
)

joined = joined.withColumn(
    "valid_observation",
    F.col("valid_price") &
    F.col("token_side").isin("home", "away")
)

joined = joined.withColumn(
    "valid_in_game",
    F.col("valid_observation") &
    F.col("in_game")
)

joined = joined.withColumn(
    "valid_minute",
    F.col("valid_in_game") &
    (F.col("fidelity") == "minute")
)

stats = joined.groupBy("espn_game_id").agg(
    F.count("*").alias("total_price_rows"),
    F.sum(F.when(F.col("token_side") == "home", 1).otherwise(0))
        .alias("home_token_rows"),
    F.sum(F.when(F.col("token_side") == "away", 1).otherwise(0))
        .alias("away_token_rows"),
    F.sum(F.when(F.col("token_side") == "unknown", 1).otherwise(0))
        .alias("unknown_token_rows"),
    F.sum(F.when(~F.col("valid_price"), 1).otherwise(0))
        .alias("invalid_price_rows"),
    F.sum(F.when(F.col("valid_in_game"), 1).otherwise(0))
        .alias("in_game_valid_rows"),
    F.countDistinct(
        F.when(F.col("valid_minute"), F.col("minute_bucket"))
    ).alias("minute_coverage"),
    F.min(
        F.when(F.col("valid_in_game"), F.col("timestamp"))
    ).alias("first_in_game_price"),
    F.max(
        F.when(F.col("valid_in_game"), F.col("timestamp"))
    ).alias("last_in_game_price")
)

result = games.join(stats, on="espn_game_id", how="left")

numeric_cols = [
    "total_price_rows", "home_token_rows",
    "away_token_rows", "unknown_token_rows",
    "invalid_price_rows", "in_game_valid_rows",
    "minute_coverage"
]

result = result.fillna(0, subset=numeric_cols)

result = result.withColumn(
    "coverage_pct",
    F.round(F.col("minute_coverage") / 240.0 * 100, 2)
)

result_pd = result.toPandas()
result_pd = result_pd.sort_values(["season", "espn_game_id"])

output_csv = RESULTS / "price_validation_by_game.csv"
result_pd.to_csv(output_csv, index=False)

total = len(result_pd)
with_data = int((result_pd["in_game_valid_rows"] > 0).sum())
at_least_200 = int((result_pd["minute_coverage"] >= 200).sum())
no_data = int((result_pd["in_game_valid_rows"] == 0).sum())

summary = f"""
=== FULL-SEASON PRICE VALIDATION ===

Matched games: {total}
Games with in-game prices: {with_data}
Games without in-game prices: {no_data}
Games with >=200 minute observations: {at_least_200}

Average minute coverage: {result_pd['minute_coverage'].mean():.2f}
Median minute coverage: {result_pd['minute_coverage'].median():.2f}

Total home token rows: {result_pd['home_token_rows'].sum():,}
Total away token rows: {result_pd['away_token_rows'].sum():,}
Total unknown token rows: {result_pd['unknown_token_rows'].sum():,}
Total invalid price rows: {result_pd['invalid_price_rows'].sum():,}

Games by season:
{result_pd.groupby('season').size().to_string()}

Minute coverage distribution:
{result_pd['minute_coverage'].describe().to_string()}
"""

print(summary)

(RESULTS / "price_validation_summary.txt").write_text(summary)

print("Saved:", output_csv)
print("Saved:", RESULTS / "price_validation_summary.txt")

spark.stop()
