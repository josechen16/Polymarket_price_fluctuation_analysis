
import json
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window


BASE_DIR = Path(__file__).resolve().parent.parent

NBA_DIR = BASE_DIR / "data/raw/nba"
GAMES_DIR = NBA_DIR / "games"

OUTPUT_PATH = str(
    BASE_DIR / "data/processed/nba_events"
)


def parse_espn_time(value):
    if not value:
        return None

    return (
        datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
        .astimezone(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def main():
    spark = (
        SparkSession.builder
        .appName("NBA_Prepare_Events")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    print("\n=== STEP 1: Load Schedule ===")

    with (NBA_DIR / "espn_schedule.json").open() as f:
        schedule = json.load(f)

    game_info = {
        str(game["event_id"]): game
        for game in schedule
    }

    print("Games in schedule:", len(game_info))

    print("\n=== STEP 2: Read ESPN Events ===")

    rows = []

    for path in sorted(GAMES_DIR.glob("*.json")):
        event_id = path.stem

        if event_id not in game_info:
            continue

        game = game_info[event_id]

        with path.open() as f:
            data = json.load(f)

        scheduled_start = parse_espn_time(
            game.get("date")
        )

        for index, play in enumerate(data.get("plays", [])):
            period = play.get("period") or {}

            rows.append({
                "event_id": event_id,
                "game_name": game.get("name"),
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "scheduled_start": scheduled_start,
                "play_index": index,
                "period": period.get("number"),
                "clock": (
                    (play.get("clock") or {})
                    .get("displayValue")
                ),
                "wallclock": play.get("wallclock"),
                "description": play.get("text"),
                "home_score": play.get("homeScore"),
                "away_score": play.get("awayScore")
            })

    if not rows:
        spark.stop()
        raise ValueError(
            "No ESPN play-by-play events found"
        )

    print("Raw events:", len(rows))

    df = spark.createDataFrame(rows)

    print("\n=== STEP 3: Prepare Timestamps ===")

    df = (
        df.withColumn(
            "event_time",
            F.to_timestamp("wallclock")
        )
        .withColumn(
            "scheduled_start_time",
            F.to_timestamp(
                F.col("scheduled_start"),
                "yyyy-MM-dd HH:mm:ss"
            )
        )
    )

    print("\n=== STEP 4: Calculate Scoring Changes ===")

    window = (
        Window.partitionBy("event_id")
        .orderBy("play_index")
    )

    df = (
        df.withColumn(
            "previous_home_score",
            F.lag(
                "home_score", 1, 0
            ).over(window)
        )
        .withColumn(
            "previous_away_score",
            F.lag(
                "away_score", 1, 0
            ).over(window)
        )
        .withColumn(
            "home_points",
            F.col("home_score")
            - F.col("previous_home_score")
        )
        .withColumn(
            "away_points",
            F.col("away_score")
            - F.col("previous_away_score")
        )
        .withColumn(
            "points_scored",
            F.col("home_points")
            + F.col("away_points")
        )
        .withColumn(
            "score_diff",
            F.col("home_score")
            - F.col("away_score")
        )
        .withColumn(
            "is_scoring_event",
            F.col("points_scored") > 0
        )
    )

    print("\n=== STEP 5: Validate Events ===")

    total_events = df.count()

    scoring_events = df.filter(
        F.col("is_scoring_event")
    ).count()

    missing_timestamps = df.filter(
        F.col("event_time").isNull()
    ).count()

    invalid_starts = (
        df.select(
            "event_id",
            "scheduled_start_time"
        )
        .dropDuplicates(["event_id"])
        .filter(
            F.col("scheduled_start_time").isNull()
        )
        .count()
    )

    print("Total events:", total_events)
    print("Scoring events:", scoring_events)
    print("Missing timestamps:", missing_timestamps)
    print("Games with invalid start:", invalid_starts)

    if invalid_starts > 0:
        raise ValueError(
            "ESPN start time conversion failed"
        )

    if missing_timestamps > 0:
        raise ValueError(
            "Some ESPN events have invalid timestamps"
        )

    print("\n=== STEP 6: Save Parquet ===")

    (
        df.repartition(4, "event_id")
        .write
        .mode("overwrite")
        .parquet(OUTPUT_PATH)
    )

    print("Saved to:", OUTPUT_PATH)

    print("\n=== STEP 7: Verify Rockets vs Thunder ===")

    result = spark.read.parquet(OUTPUT_PATH)

    (
        result
        .filter(
            F.col("event_id") == "401809243"
        )
        .select(
            "event_id",
            "home_team",
            "away_team",
            "scheduled_start_time"
        )
        .limit(1)
        .show(truncate=False)
    )

    (
        result
        .filter(
            (F.col("event_id") == "401809243")
            & F.col("is_scoring_event")
        )
        .orderBy("play_index")
        .select(
            "event_time",
            "period",
            "away_score",
            "home_score",
            "points_scored",
            "score_diff",
            "description"
        )
        .show(10, truncate=False)
    )

    spark.stop()

    print("\nEVENT PREPARATION COMPLETED!")


if __name__ == "__main__":
    main()
