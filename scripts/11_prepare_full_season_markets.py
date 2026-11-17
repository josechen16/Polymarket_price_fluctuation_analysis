from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE = Path(__file__).resolve().parent.parent
INPUT = str(BASE / "data/raw/polymarket/data/markets.parquet")
OUTPUT = str(BASE / "data/processed/full_season/nba_moneyline_markets")

def main():
    spark = (
        SparkSession.builder
        .appName("NBA_Full_Season_Markets")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    markets = spark.read.parquet(INPUT)
    print("Total raw markets:", markets.count())

    moneyline = markets.filter(
        F.col("season").isin("2024-25", "2025-26")
        & (F.col("market_type") == "moneyline")
        & F.col("game_start_time").isNotNull()
    )

    print("\nMarkets by season:")
    moneyline.groupBy("season").count().orderBy("season").show()

    mapped = moneyline.select(
        "condition_id",
        "event_id",
        "event_title",
        "game_start_time",
        "season",
        "trades_coverage",
        F.posexplode("outcomes").alias("outcome_index", "team"),
        "token_ids"
    ).withColumn(
        "token_id",
        F.element_at("token_ids", F.col("outcome_index") + 1)
    ).drop("token_ids")

    invalid = mapped.filter(
        F.col("team").isNull()
        | F.col("token_id").isNull()
        | F.col("condition_id").isNull()
    ).count()

    if invalid:
        raise ValueError(f"Invalid token mappings: {invalid}")

    duplicates = (
        mapped.groupBy("condition_id", "outcome_index")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    if duplicates:
        raise ValueError(f"Duplicate mappings: {duplicates}")

    print("\nToken rows by season:")
    mapped.groupBy("season").agg(
        F.count("*").alias("token_rows"),
        F.countDistinct("condition_id").alias("markets")
    ).orderBy("season").show()

    (
        mapped.repartition(4)
        .write.mode("overwrite")
        .parquet(OUTPUT)
    )

    print("Saved:", OUTPUT)
    spark.stop()

if __name__ == "__main__":
    main()
