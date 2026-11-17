from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data/processed/full_season/event_price"
OUT = BASE / "results/full_season/q3_price_settling"
OUT.mkdir(parents=True, exist_ok=True)

print("=== Q3 EVENT STUDY ===", flush=True)

price_cols = [
    "price_pre", "price_post_1m",
    "price_post_2m", "price_post_3m",
    "price_post_5m"
]

df = ds.dataset(
    str(DATA / "scoring_event_price_join"),
    format="parquet"
).to_table().to_pandas()

df["espn_game_id"] = df["espn_game_id"].astype(str)
df["event_time"] = pd.to_datetime(
    df["event_epoch"], unit="s", utc=True
)

df = df.sort_values(
    ["espn_game_id", "sequence_number"]
).reset_index(drop=True)

# Identify abnormal games using scoring-event timestamps.
spans = df.groupby("espn_game_id")["event_time"].agg(
    ["min", "max"]
)
spans["span_minutes"] = (
    spans["max"] - spans["min"]
).dt.total_seconds() / 60

bad_games = set(
    spans.index[spans["span_minutes"] > 300]
)

# Determine which team scored using score changes.
df["previous_home"] = (
    df.groupby("espn_game_id")["home_score"]
    .shift(fill_value=0)
)
df["previous_away"] = (
    df.groupby("espn_game_id")["away_score"]
    .shift(fill_value=0)
)

df["home_added"] = (
    df["home_score"] - df["previous_home"]
)
df["away_added"] = (
    df["away_score"] - df["previous_away"]
)

home_scored = (
    df["home_added"].between(1, 4)
    & df["away_added"].eq(0)
)
away_scored = (
    df["away_added"].between(1, 4)
    & df["home_added"].eq(0)
)

df["direction"] = np.select(
    [home_scored, away_scored],
    [1, -1],
    default=0
)

df["points_added"] = np.select(
    [home_scored, away_scored],
    [df["home_added"], df["away_added"]],
    default=0
)

# Next scoring event in official sequence order.
df["next_event_time"] = (
    df.groupby("espn_game_id")["event_time"].shift(-1)
)

df["seconds_to_next_score"] = (
    df["next_event_time"] - df["event_time"]
).dt.total_seconds()

df["no_new_score_5m"] = (
    df["seconds_to_next_score"] > 300
)

# Exclude events without a next scoring event from this
# classification, because they may be near the game end.
df["has_next_score"] = (
    df["next_event_time"].notna()
)

df["valid_event"] = (
    df["direction"].ne(0)
    & ~df["espn_game_id"].isin(bad_games)
    & df[price_cols].notna().all(axis=1)
    & df[price_cols].ge(0).all(axis=1)
    & df[price_cols].le(1).all(axis=1)
)

# Directional cumulative reactions.
for minute in [1, 2, 3, 5]:
    df[f"reaction_{minute}m_pp"] = (
        df["direction"]
        * (df[f"price_post_{minute}m"] - df["price_pre"])
        * 100
    )

# Absolute price movement between observations.
pairs = [
    ("price_pre", "price_post_1m", "move_0_1m_pp"),
    ("price_post_1m", "price_post_2m", "move_1_2m_pp"),
    ("price_post_2m", "price_post_3m", "move_2_3m_pp"),
    ("price_post_3m", "price_post_5m", "move_3_5m_pp")
]

for before, after, name in pairs:
    df[name] = (
        df[after] - df[before]
    ).abs() * 100

# The final interval covers two minutes; normalize per minute.
df["move_3_5_per_min_pp"] = (
    df["move_3_5m_pp"] / 2
)

all_events = df[df["valid_event"]].copy()

isolated = all_events[
    all_events["has_next_score"]
    & all_events["no_new_score_5m"]
].copy()

# Conservative subset: next score exists and is >5m away.
# Thus the observation is before another known scoring play.
clean_isolated = isolated[
    isolated["seconds_to_next_score"].between(
        301, 3600
    )
].copy()

samples = {
    "all_events": all_events,
    "no_new_score_5m": isolated,
    "clean_no_new_score": clean_isolated
}

def event_summary(name, data):
    rows = []
    for minute in [1, 2, 3, 5]:
        col = f"reaction_{minute}m_pp"
        rows.append({
            "sample": name,
            "minute": minute,
            "n_events": len(data),
            "mean_reaction_pp": data[col].mean(),
            "median_reaction_pp": data[col].median(),
            "mean_abs_reaction_pp": data[col].abs().mean()
        })
    return pd.DataFrame(rows)

summary = pd.concat(
    [event_summary(name, data)
     for name, data in samples.items()],
    ignore_index=True
)

summary.to_csv(
    OUT / "event_study_summary.csv",
    index=False
)

movement_cols = [
    "move_0_1m_pp",
    "move_1_2m_pp",
    "move_2_3m_pp",
    "move_3_5_per_min_pp"
]

movement = pd.DataFrame([
    {
        "sample": name,
        "n_events": len(data),
        **{
            col: data[col].mean()
            for col in movement_cols
        }
    }
    for name, data in samples.items()
])

movement.to_csv(
    OUT / "incremental_movement.csv",
    index=False
)

# Save event-level observations for future regression.
keep = [
    "espn_game_id", "season", "event_id",
    "sequence_number", "event_time",
    "period", "points_added", "direction",
    "seconds_to_next_score", "no_new_score_5m",
    *price_cols,
    "reaction_1m_pp", "reaction_2m_pp",
    "reaction_3m_pp", "reaction_5m_pp",
    *movement_cols
]

all_events[keep].to_parquet(
    OUT / "q3_event_level.parquet",
    index=False
)

# Figure 1: Directional cumulative reaction.
plt.figure(figsize=(9, 5))

for name, data in samples.items():
    means = [
        data[f"reaction_{m}m_pp"].mean()
        for m in [1, 2, 3, 5]
    ]
    plt.plot(
        [1, 2, 3, 5],
        means,
        marker="o",
        label=name.replace("_", " ")
    )

plt.xlabel("Minutes after scoring event")
plt.ylabel("Mean directional reaction (pp)")
plt.title("Polymarket Price Reaction After NBA Scores")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUT / "q3_cumulative_reaction.png",
    dpi=200
)
plt.close()

# Figure 2: Incremental absolute movement.
plt.figure(figsize=(9, 5))

for name, data in samples.items():
    means = [
        data[col].mean()
        for col in movement_cols
    ]
    plt.plot(
        [1, 2, 3, 4],
        means,
        marker="o",
        label=name.replace("_", " ")
    )

plt.xticks(
    [1, 2, 3, 4],
    ["0-1m", "1-2m", "2-3m", "3-5m / min"]
)
plt.ylabel("Mean absolute movement (pp per minute)")
plt.title("Incremental Polymarket Price Movement")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUT / "q3_incremental_movement.png",
    dpi=200
)
plt.close()

print("\n=== DATA QUALITY ===")
print("Original events:", len(df))
print("Excluded abnormal games:", len(bad_games))
print("Valid events:", len(all_events))
print("No new score within 5m:", len(isolated))
print("Clean isolated events:", len(clean_isolated))

print("\n=== EVENT STUDY SUMMARY ===")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n=== INCREMENTAL MOVEMENT ===")
print(
    movement.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nOutputs saved to:", OUT)
