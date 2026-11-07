
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data/processed"

EVENTS = str(DATA / "nba_events")
MARKETS = str(DATA / "nba_moneyline_markets")
PRICES = str(DATA / "prices-2025-26.parquet")
OUTPUT = str(DATA / "nba_event_price_reactions")


ALIASES = {
    "Atlanta Hawks": "Hawks",
    "Boston Celtics": "Celtics",
    "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets",
    "Chicago Bulls": "Bulls",
    "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks",
    "Denver Nuggets": "Nuggets",
    "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors",
    "Houston Rockets": "Rockets",
    "Indiana Pacers": "Pacers",
    "LA Clippers": "Clippers",
    "Los Angeles Clippers": "Clippers",
    "Los Angeles Lakers": "Lakers",
    "Memphis Grizzlies": "Grizzlies",
    "Miami Heat": "Heat",
    "Milwaukee Bucks": "Bucks",
    "Minnesota Timberwolves": "Timberwolves",
    "New Orleans Pelicans": "Pelicans",
    "New York Knicks": "Knicks",
    "Oklahoma City Thunder": "Thunder",
    "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers",
    "Phoenix Suns": "Suns",
    "Portland Trail Blazers": "Trail Blazers",
    "Sacramento Kings": "Kings",
    "San Antonio Spurs": "Spurs",
    "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz",
    "Washington Wizards": "Wizards"
}


def main():
    spark = (
        SparkSession.builder
        .appName("NBA_Event_Price_Join")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    print("\n=== STEP 1: Load Data ===")

    events = spark.read.parquet(EVENTS)
    markets = spark.read.parquet(MARKETS)

    print("ESPN events:", events.count())
    print("Market token rows:", markets.count())

    print("\n=== STEP 2: Normalize Teams and Times ===")

    mapping = F.create_map(
        *[
            x
            for key, value in ALIASES.items()
            for x in (F.lit(key), F.lit(value))
        ]
    )

    games = (
        events
        .select(
            "event_id",
            "home_team",
            "away_team",
            "scheduled_start_time"
        )
        .dropDuplicates(["event_id"])
        .withColumn(
            "home_short",
            mapping[F.col("home_team")]
        )
        .withColumn(
            "away_short",
            mapping[F.col("away_team")]
        )
    )

    # Polymarket uses strings such as:
    # 2025-10-21 23:30:00+00
    # Normalize the UTC suffix before parsing.
    markets = markets.withColumn(
        "market_start",
        F.to_timestamp(
            F.regexp_replace(
                F.col("game_start_time"),
                r"\+00$",
                ""
            ),
            "yyyy-MM-dd HH:mm:ss"
        )
    )

    invalid_espn = games.filter(
        F.col("scheduled_start_time").isNull()
    ).count()

    invalid_market = markets.filter(
        F.col("market_start").isNull()
    ).count()

    print("ESPN games with invalid start:", invalid_espn)
    print("Market tokens with invalid start:", invalid_market)

    if invalid_espn > 0:
        raise ValueError(
            "ESPN start timestamps are NULL. "
            "Check 03_prepare_events.py."
        )

    print("\n=== STEP 3: Match Games ===")

    market_games = (
        markets
        .groupBy(
            "condition_id",
            "event_title",
            "market_start"
        )
        .agg(
            F.collect_set("team").alias("teams")
        )
    )

    matched = (
        games.alias("g")
        .join(
            F.broadcast(market_games).alias("m"),
            (
                F.array_contains(
                    F.col("m.teams"),
                    F.col("g.home_short")
                )
                & F.array_contains(
                    F.col("m.teams"),
                    F.col("g.away_short")
                )
                & (
                    F.abs(
                        F.unix_timestamp(
                            F.col("g.scheduled_start_time")
                        )
                        -
                        F.unix_timestamp(
                            F.col("m.market_start")
                        )
                    ) <= 7200
                )
            ),
            "inner"
        )
        .select(
            F.col("g.event_id"),
            F.col("g.home_short"),
            F.col("g.away_short"),
            F.col("m.condition_id"),
            F.col("m.event_title")
        )
        .dropDuplicates(["event_id", "condition_id"])
    )

    matched_count = matched.count()

    print("Matched games:", matched_count)

    matched.show(20, truncate=False)

    if matched_count == 0:
        raise ValueError(
            "No ESPN games matched Polymarket markets."
        )

    print("\n=== STEP 4: Select Minute Prices ===")

    prices = (
        spark.read.parquet(PRICES)
        .filter(F.col("fidelity") == "minute")
    )

    selected_prices = (
        prices
        .join(
            F.broadcast(
                matched.select("condition_id").distinct()
            ),
            "condition_id",
            "inner"
        )
        .select(
            "condition_id",
            "token_id",
            F.col("timestamp").alias("price_time"),
            "price"
        )
    )

    price_count = selected_prices.count()
    print("Selected minute prices:", price_count)

    if price_count == 0:
        raise ValueError("No minute prices found.")

    print("\n=== STEP 5: Prepare Scoring Events ===")

    scoring = (
        events
        .filter(F.col("is_scoring_event"))
        .join(
            F.broadcast(matched),
            "event_id",
            "inner"
        )
        .select(
            "event_id",
            "play_index",
            "condition_id",
            "event_time",
            "period",
            "home_score",
            "away_score",
            "score_diff",
            "points_scored",
            "description"
        )
    )

    print("Matched scoring events:", scoring.count())

    print("\n=== STEP 6: Event-Price Join ===")

    candidates = (
        scoring.alias("e")
        .join(
            selected_prices.alias("p"),
            (
                (
                    F.col("e.condition_id")
                    == F.col("p.condition_id")
                )
                &
                (
                    F.abs(
                        F.unix_timestamp("p.price_time")
                        -
                        F.unix_timestamp("e.event_time")
                    ) <= 120
                )
            ),
            "inner"
        )
        .select(
            F.col("e.*"),
            F.col("p.token_id"),
            F.col("p.price_time"),
            F.col("p.price")
        )
    )

    keys = [
        "event_id",
        "play_index",
        "token_id"
    ]

    before_window = (
        Window.partitionBy(*keys)
        .orderBy(F.col("price_time").desc())
    )

    after_window = (
        Window.partitionBy(*keys)
        .orderBy(F.col("price_time").asc())
    )

    before = (
        candidates
        .filter(F.col("price_time") < F.col("event_time"))
        .withColumn(
            "rn",
            F.row_number().over(before_window)
        )
        .filter(F.col("rn") == 1)
        .select(
            *keys,
            F.col("price").alias("price_before"),
            F.col("price_time").alias("time_before")
        )
    )

    after = (
        candidates
        .filter(F.col("price_time") > F.col("event_time"))
        .withColumn(
            "rn",
            F.row_number().over(after_window)
        )
        .filter(F.col("rn") == 1)
        .select(
            *keys,
            F.col("price").alias("price_after"),
            F.col("price_time").alias("time_after")
        )
    )

    reactions = (
        scoring
        .join(before, ["event_id", "play_index"], "inner")
        .join(after, keys, "inner")
        .withColumn(
            "price_change",
            F.col("price_after") - F.col("price_before")
        )
    )

    reaction_count = reactions.count()

    print("Price reactions:", reaction_count)

    if reaction_count == 0:
        raise ValueError(
            "No scoring events have before/after prices."
        )

    print("\n=== STEP 7: Save Parquet ===")

    (
        reactions
        .repartition(4, "event_id")
        .write
        .mode("overwrite")
        .parquet(OUTPUT)
    )

    print("Saved:", OUTPUT)

    print("\n=== STEP 8: Preview ===")

    (
        reactions
        .orderBy("event_id", "play_index")
        .select(
            "event_id",
            "play_index",
            "event_time",
            "points_scored",
            "score_diff",
            "price_before",
            "price_after",
            "price_change"
        )
        .show(15, truncate=False)
    )

    spark.stop()

    print("\nEVENT-PRICE JOIN COMPLETED!")


if __name__ == "__main__":
    main()
