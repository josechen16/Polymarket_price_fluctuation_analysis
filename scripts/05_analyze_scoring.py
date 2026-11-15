from pathlib import Path
import csv
import re

from pyspark.sql import SparkSession, functions as F

BASE = Path.home() / "Desktop" / "predict_market_project"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("NBA_Scoring_Impact_Analysis")
    .master("local[*]")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def normalize_team(name):
    if name is None:
        return None
    return re.sub(r"[^a-z0-9]", "", name.lower())


normalize_udf = F.udf(normalize_team, "string")


def save_csv(df, filename):
    rows = df.collect()
    columns = df.columns
    path = RESULTS / filename

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(list(row))

    print("Saved:", path)


reactions = spark.read.parquet(
    str(BASE / "data/processed/nba_event_price_reactions")
)

events = spark.read.parquet(
    str(BASE / "data/processed/nba_events")
)

markets = spark.read.parquet(
    str(BASE / "data/processed/nba_moneyline_markets")
)

print("\n=== LOADING DATA ===")
print("Price reaction rows:", reactions.count())

market_mapping = markets.select(
    "condition_id",
    "token_id",
    F.col("team").alias("token_team")
).dropDuplicates(["condition_id", "token_id"])

event_details = events.select(
    "event_id",
    "play_index",
    "home_team",
    "away_team",
    "home_points",
    "away_points",
    "previous_home_score",
    "previous_away_score"
).dropDuplicates(["event_id", "play_index"])

df = (
    reactions
    .join(
        market_mapping,
        ["condition_id", "token_id"],
        "left"
    )
    .join(
        event_details,
        ["event_id", "play_index"],
        "left"
    )
)

df = (
    df
    .withColumn(
        "scoring_team",
        F.when(
            (F.col("home_points") > 0) &
            (F.col("away_points") == 0),
            F.col("home_team")
        ).when(
            (F.col("away_points") > 0) &
            (F.col("home_points") == 0),
            F.col("away_team")
        )
    )
    .withColumn(
        "token_team_normalized",
        normalize_udf(F.col("token_team"))
    )
    .withColumn(
        "home_team_normalized",
        normalize_udf(F.col("home_team"))
    )
    .withColumn(
        "away_team_normalized",
        normalize_udf(F.col("away_team"))
    )
)

df = df.withColumn(
    "token_side",
    F.when(
        F.col("home_team_normalized").endswith(
            F.col("token_team_normalized")
        ),
        F.lit("home")
    ).when(
        F.col("away_team_normalized").endswith(
            F.col("token_team_normalized")
        ),
        F.lit("away")
    )
)

df = df.withColumn(
    "scoring_side",
    F.when(
        (F.col("home_points") > 0) &
        (F.col("away_points") == 0),
        F.lit("home")
    ).when(
        (F.col("away_points") > 0) &
        (F.col("home_points") == 0),
        F.lit("away")
    )
)

df = df.withColumn(
    "aligned_price_change",
    F.when(
        F.col("token_side") == F.col("scoring_side"),
        F.col("price_change")
    ).when(
        F.col("token_side").isNotNull() &
        F.col("scoring_side").isNotNull(),
        -F.col("price_change")
    )
)

df = (
    df
    .withColumn(
        "absolute_price_change",
        F.abs(F.col("price_change"))
    )
    .withColumn(
        "pre_event_score_diff",
        F.abs(
            F.col("previous_home_score") -
            F.col("previous_away_score")
        )
    )
    .withColumn(
        "game_situation",
        F.when(
            F.col("pre_event_score_diff") <= 3,
            "Very Close (0-3)"
        ).when(
            F.col("pre_event_score_diff") <= 10,
            "Close (4-10)"
        ).otherwise("Large Lead (11+)")
    )
)

print("\n=== TOKEN MAPPING VALIDATION ===")

validation = df.agg(
    F.count("*").alias("total_rows"),
    F.sum(
        F.col("token_side").isNull().cast("int")
    ).alias("unmatched_token_side"),
    F.sum(
        F.col("scoring_side").isNull().cast("int")
    ).alias("unknown_scoring_side"),
    F.sum(
        F.col("aligned_price_change").isNull().cast("int")
    ).alias("missing_aligned_change")
)

validation.show(truncate=False)

invalid = df.filter(
    F.col("aligned_price_change").isNull()
).count()

if invalid > 0:
    spark.stop()
    raise RuntimeError(
        f"Found {invalid} events with invalid team mapping."
    )

print("\n=== DETECTING OVERLAPPING SCORING EVENTS ===")

other_events = (
    events
    .filter(F.col("is_scoring_event") == True)
    .select(
        F.col("event_id").alias("other_event_id"),
        F.col("play_index").alias("other_play_index"),
        F.col("event_time").alias("other_event_time")
    )
)

overlaps = (
    df.alias("r")
    .join(
        other_events.alias("o"),
        (
            (F.col("r.event_id") == F.col("o.other_event_id")) &
            (F.col("r.play_index") != F.col("o.other_play_index")) &
            (
                F.col("o.other_event_time") >=
                F.col("r.time_before")
            ) &
            (
                F.col("o.other_event_time") <=
                F.col("r.time_after")
            )
        ),
        "left"
    )
    .groupBy(
        F.col("r.event_id").alias("event_id"),
        F.col("r.play_index").alias("play_index")
    )
    .agg(
        F.count("other_play_index").alias(
            "other_scoring_events_in_window"
        )
    )
)

df = df.join(
    overlaps,
    ["event_id", "play_index"],
    "left"
)

df = df.withColumn(
    "is_isolated",
    F.col("other_scoring_events_in_window") == 0
)

print("\n=== EVENT ISOLATION ===")

df.groupBy("is_isolated").count().show()

print("\n=== SCORING IMPACT ANALYSIS ===")


def summarize(data):
    return (
        data.groupBy("points_scored")
        .agg(
            F.count("*").alias("event_count"),
            F.countDistinct("event_id").alias("game_count"),
            F.avg("price_change").alias("avg_token_price_change"),
            F.avg("aligned_price_change").alias(
                "avg_scoring_team_price_change"
            ),
            F.avg("absolute_price_change").alias(
                "avg_absolute_price_change"
            ),
            F.stddev("aligned_price_change").alias(
                "stddev_aligned_price_change"
            ),
            F.avg(
                (F.col("aligned_price_change") > 0)
                .cast("double")
            ).alias("fraction_positive")
        )
        .orderBy("points_scored")
    )


all_summary = summarize(df)
isolated_summary = summarize(
    df.filter(F.col("is_isolated"))
)

print("\n=== ALL EVENTS ===")
all_summary.show(truncate=False)

print("\n=== ISOLATED EVENTS ONLY ===")
isolated_summary.show(truncate=False)

save_csv(
    all_summary,
    "scoring_impact_all.csv"
)

save_csv(
    isolated_summary,
    "scoring_impact_isolated.csv"
)

print("\n=== SCORING IMPACT BY PERIOD ===")

period_summary = (
    df.groupBy("period", "points_scored")
    .agg(
        F.count("*").alias("event_count"),
        F.avg("aligned_price_change").alias(
            "avg_aligned_change"
        ),
        F.avg("absolute_price_change").alias(
            "avg_absolute_change"
        )
    )
    .orderBy("period", "points_scored")
)

period_summary.show(30, truncate=False)

save_csv(
    period_summary,
    "scoring_impact_by_period.csv"
)

print("\n=== SCORING IMPACT BY GAME SITUATION ===")

situation_summary = (
    df.groupBy("game_situation", "points_scored")
    .agg(
        F.count("*").alias("event_count"),
        F.avg("aligned_price_change").alias(
            "avg_aligned_change"
        ),
        F.avg("absolute_price_change").alias(
            "avg_absolute_change"
        )
    )
    .orderBy("game_situation", "points_scored")
)

situation_summary.show(20, truncate=False)

save_csv(
    situation_summary,
    "scoring_impact_by_situation.csv"
)

print("\n=== EXPORTING EVENT-LEVEL DATA ===")

export_columns = [
    "event_id",
    "play_index",
    "event_time",
    "period",
    "home_team",
    "away_team",
    "scoring_team",
    "token_team",
    "points_scored",
    "pre_event_score_diff",
    "game_situation",
    "price_before",
    "price_after",
    "price_change",
    "aligned_price_change",
    "absolute_price_change",
    "time_before",
    "time_after",
    "other_scoring_events_in_window",
    "is_isolated"
]

save_csv(
    df.select(*export_columns).orderBy(
        "event_id", "play_index"
    ),
    "event_level_scoring_analysis.csv"
)

print("\nSCORING IMPACT ANALYSIS COMPLETED!")

spark.stop()
