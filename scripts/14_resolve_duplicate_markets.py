from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "data/processed/full_season"
PRICE_ROOT = BASE / "data/raw/polymarket/data/prices"

candidates = pd.read_csv(
    ROOT / "game_matching_candidates.csv",
    dtype=str
)

matched = pd.read_csv(
    ROOT / "matched_games.csv",
    dtype=str
)

candidates["time_difference_hours"] = pd.to_numeric(
    candidates["time_difference_hours"]
)

counts = candidates.groupby("espn_game_id").size()
duplicate_ids = set(counts[counts > 1].index)

duplicates = candidates[
    candidates["espn_game_id"].isin(duplicate_ids)
].copy()

duplicates["game_time"] = pd.to_datetime(
    duplicates["game_time"], utc=True
)

duplicates["window_end"] = (
    duplicates["game_time"] + pd.Timedelta(hours=4)
)

target_ids = pa.array(
    duplicates["condition_id"].unique().tolist(),
    type=pa.string()
)

print("Duplicate games:", len(duplicate_ids))
print("Candidate markets:", len(duplicates))

frames = []

for season in ["2024-25", "2025-26"]:
    path = PRICE_ROOT / f"prices-{season}.parquet"
    pf = pq.ParquetFile(path)

    print(
        f"\nScanning {season}: "
        f"{pf.metadata.num_rows:,} rows",
        flush=True
    )

    for batch in pf.iter_batches(
        batch_size=250000,
        columns=[
            "condition_id", "token_id",
            "timestamp", "price", "fidelity"
        ]
    ):
        mask = pc.is_in(
            batch.column("condition_id"),
            value_set=target_ids
        )

        filtered = batch.filter(mask)

        if filtered.num_rows:
            frames.append(filtered.to_pandas())

if not frames:
    raise RuntimeError(
        "No price records found for duplicate markets"
    )

prices = pd.concat(frames, ignore_index=True)
del frames

prices["timestamp"] = pd.to_datetime(
    prices["timestamp"], utc=True
)

prices["token_id"] = prices["token_id"].astype(str)
prices["condition_id"] = prices["condition_id"].astype(str)

print("\nMatching price rows:", len(prices))

# Join price observations to candidate markets.
joined = prices.merge(
    duplicates[[
        "condition_id",
        "espn_game_id",
        "home_token_id",
        "away_token_id",
        "game_time",
        "window_end"
    ]],
    on="condition_id",
    how="inner",
    validate="many_to_one"
)

joined["is_home"] = (
    joined["token_id"] == joined["home_token_id"]
)

joined["is_away"] = (
    joined["token_id"] == joined["away_token_id"]
)

joined["valid_token"] = (
    joined["is_home"] | joined["is_away"]
)

joined["valid_price"] = (
    joined["price"].between(0, 1)
)

joined["in_game"] = (
    (joined["timestamp"] >= joined["game_time"])
    & (joined["timestamp"] <= joined["window_end"])
)

# Convert either token into home-win probability.
joined["home_win_probability"] = joined["price"].where(
    joined["is_home"],
    1.0 - joined["price"]
)

joined.loc[
    ~joined["valid_token"],
    "home_win_probability"
] = float("nan")

valid = joined[
    joined["valid_token"] & joined["valid_price"]
].copy()

valid["minute_bucket"] = valid["timestamp"].dt.floor("min")

# Prefer minute fidelity and unique minute observations.
minute = valid[
    valid["in_game"] & (valid["fidelity"] == "minute")
]

minute_counts = minute.groupby("condition_id")[
    "minute_bucket"
].nunique()

game_counts = valid[
    valid["in_game"]
].groupby("condition_id")[
    "timestamp"
].nunique()

total_counts = valid.groupby("condition_id").size()

home_counts = valid[
    valid["is_home"]
].groupby("condition_id").size()

away_counts = valid[
    valid["is_away"]
].groupby("condition_id").size()

unknown_counts = joined[
    ~joined["valid_token"]
].groupby("condition_id").size()

duplicates["minute_observations"] = (
    duplicates["condition_id"]
    .map(minute_counts).fillna(0).astype(int)
)

duplicates["in_game_observations"] = (
    duplicates["condition_id"]
    .map(game_counts).fillna(0).astype(int)
)

duplicates["total_observations"] = (
    duplicates["condition_id"]
    .map(total_counts).fillna(0).astype(int)
)

duplicates["home_token_rows"] = (
    duplicates["condition_id"]
    .map(home_counts).fillna(0).astype(int)
)

duplicates["away_token_rows"] = (
    duplicates["condition_id"]
    .map(away_counts).fillna(0).astype(int)
)

duplicates["unknown_token_rows"] = (
    duplicates["condition_id"]
    .map(unknown_counts).fillna(0).astype(int)
)

resolved = []
unresolved = []

for game_id, group in duplicates.groupby("espn_game_id"):
    ranked = group.sort_values(
        [
            "minute_observations",
            "in_game_observations",
            "total_observations"
        ],
        ascending=False
    )

    best = ranked.iloc[0]

    if best["in_game_observations"] == 0:
        unresolved.append(group)
        continue

    if len(ranked) > 1:
        second = ranked.iloc[1]

        best_score = (
            best["minute_observations"],
            best["in_game_observations"],
            best["total_observations"]
        )

        second_score = (
            second["minute_observations"],
            second["in_game_observations"],
            second["total_observations"]
        )

        if best_score == second_score:
            unresolved.append(group)
            continue

    resolved.append(best)

resolved_df = pd.DataFrame(resolved)

if unresolved:
    unresolved_df = pd.concat(
        unresolved, ignore_index=True
    )
else:
    unresolved_df = duplicates.iloc[0:0].copy()

original_columns = matched.columns.tolist()

if not resolved_df.empty:
    final = pd.concat(
        [
            matched,
            resolved_df[original_columns]
        ],
        ignore_index=True
    )
else:
    final = matched.copy()

assert final["espn_game_id"].is_unique
assert final["condition_id"].is_unique

final.to_csv(
    ROOT / "matched_games_final.csv",
    index=False
)

final.to_parquet(
    ROOT / "matched_games_final.parquet",
    index=False
)

duplicates.to_csv(
    ROOT / "duplicate_market_price_coverage.csv",
    index=False
)

unresolved_df.to_csv(
    ROOT / "unresolved_duplicate_markets.csv",
    index=False
)

print("\n=== FINAL MARKET RESOLUTION REPORT ===")
print("Original matched games:", len(matched))
print("Duplicate games:", len(duplicate_ids))
print("Resolved duplicate games:", len(resolved_df))
print(
    "Unresolved duplicate games:",
    unresolved_df["espn_game_id"].nunique()
)
print("Final matched games:", len(final))

print("\n=== TOKEN VALIDATION ===")
print(
    "Home token price rows:",
    int(joined["is_home"].sum())
)
print(
    "Away token price rows:",
    int(joined["is_away"].sum())
)
print(
    "Unknown token price rows:",
    int((~joined["valid_token"]).sum())
)

print("\n=== FINAL MATCHES BY SEASON ===")
print(final.groupby("season").size().to_string())

print("\n=== DUPLICATE MARKET COVERAGE ===")
print(
    duplicates[[
        "minute_observations",
        "in_game_observations",
        "total_observations"
    ]].describe().to_string()
)

print("\nSaved:", ROOT / "matched_games_final.parquet")
