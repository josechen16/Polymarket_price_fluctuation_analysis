
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = str(
    BASE_DIR / "data/raw/polymarket/data/markets.parquet"
)

OUTPUT_PATH = str(
    BASE_DIR / "data/processed/nba_moneyline_markets"
)


def main():
    spark = (
        SparkSession.builder
        .appName("NBA_Prepare_Markets")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    print("\n=== STEP 1: Read Markets ===")

    markets = spark.read.parquet(INPUT_PATH)

    print("Total markets:", markets.count())

    print("\n=== STEP 2: Filter NBA Moneyline ===")

    moneyline = markets.filter(
        (F.col("season") == "2025-26")
        & (F.col("market_type") == "moneyline")
        & F.col("game_start_time").isNotNull()
    )

    print("Moneyline markets:", moneyline.count())

    print("\n=== STEP 3: Map Teams to Tokens ===")

    # Keep the position of each outcome aligned with its token ID.
    mapped = moneyline.select(
        "condition_id",
        "event_id",
        "event_title",
        "game_start_time",
        "season",
        "trades_coverage",
        F.posexplode("outcomes").alias("outcome_index", "team"),
        "token_ids"
    )

    mapped = mapped.withColumn(
        "token_id",
        F.element_at(
            F.col("token_ids"),
            F.col("outcome_index") + F.lit(1)
        )
    )

    mapped = mapped.select(
        "condition_id",
        "event_id",
        "event_title",
        "game_start_time",
        "season",
        "trades_coverage",
        "outcome_index",
        "team",
        "token_id"
    )

    print("Mapped token rows:", mapped.count())

    print("\n=== STEP 4: Validate Token Mapping ===")

    invalid = mapped.filter(
        F.col("team").isNull()
        | F.col("token_id").isNull()
    ).count()

    print("Invalid token mappings:", invalid)

    if invalid > 0:
        spark.stop()
        raise ValueError("Found invalid token mappings")

    print("\n=== STEP 5: Save Parquet ===")

    (
        mapped
        .repartition(4)
        .write
        .mode("overwrite")
        .parquet(OUTPUT_PATH)
    )

    print("Saved to:", OUTPUT_PATH)

    print("\n=== STEP 6: Verify Output ===")

    result = spark.read.parquet(OUTPUT_PATH)

    print("Saved rows:", result.count())

    result.orderBy("game_start_time", "outcome_index").show(
        10,
        truncate=False
    )

    spark.stop()

    print("\nMARKET PREPARATION COMPLETED!")


if __name__ == "__main__":
    main()
