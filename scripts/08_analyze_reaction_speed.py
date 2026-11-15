from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path.home() / "Desktop" / "predict_market_project"
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    RESULTS / "event_level_scoring_analysis.csv",
    parse_dates=["event_time", "time_before", "time_after"]
)

df["is_isolated"] = (
    df["is_isolated"].astype(str).str.lower() == "true"
)

df["seconds_to_next_observation"] = (
    df["time_after"] - df["event_time"]
).dt.total_seconds()

df["observation_window_seconds"] = (
    df["time_after"] - df["time_before"]
).dt.total_seconds()

df["price_changed"] = (
    df["price_change"].abs() > 1e-10
)

df["aligned_positive"] = (
    df["aligned_price_change"] > 1e-10
)

df["absolute_change_pp"] = (
    df["price_change"].abs() * 100
)

assert df["seconds_to_next_observation"].ge(0).all()
assert df["observation_window_seconds"].gt(0).all()

def summarize(data, group_columns):
    return (
        data.groupby(group_columns, observed=True)
        .agg(
            event_count=("event_id", "size"),
            game_count=("event_id", "nunique"),
            changed_count=("price_changed", "sum"),
            fraction_price_changed=("price_changed", "mean"),
            fraction_aligned_positive=("aligned_positive", "mean"),
            mean_seconds_to_next_observation=(
                "seconds_to_next_observation", "mean"
            ),
            median_seconds_to_next_observation=(
                "seconds_to_next_observation", "median"
            ),
            mean_absolute_change_pp=(
                "absolute_change_pp", "mean"
            )
        )
        .reset_index()
    )

print("\n=== FIRST OBSERVATION RESPONSE: ALL EVENTS ===")

all_summary = summarize(df, ["points_scored"])
print(all_summary.to_string(index=False))
all_summary.to_csv(
    RESULTS / "reaction_detection_all.csv",
    index=False
)

print("\n=== FIRST OBSERVATION RESPONSE: ISOLATED ===")

isolated = df[df["is_isolated"]].copy()

isolated_summary = summarize(
    isolated, ["points_scored"]
)

print(isolated_summary.to_string(index=False))
isolated_summary.to_csv(
    RESULTS / "reaction_detection_isolated.csv",
    index=False
)

print("\n=== OBSERVATION DELAY DISTRIBUTION ===")

bins = [0, 15, 30, 45, 60, float("inf")]
labels = [
    "0-15 sec",
    "15-30 sec",
    "30-45 sec",
    "45-60 sec",
    "60+ sec"
]

df["delay_bucket"] = pd.cut(
    df["seconds_to_next_observation"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

delay_summary = summarize(df, ["delay_bucket"])

print(delay_summary.to_string(index=False))

delay_summary.to_csv(
    RESULTS / "reaction_observation_delay.csv",
    index=False
)

# Figure 5: price-change detection
plt.figure(figsize=(8, 5))

x = np.arange(3)
width = 0.35

plt.bar(
    x - width / 2,
    all_summary["fraction_price_changed"] * 100,
    width,
    label="All Events"
)

plt.bar(
    x + width / 2,
    isolated_summary["fraction_price_changed"] * 100,
    width,
    label="Isolated Events"
)

plt.xticks(
    x, ["Free Throw", "2-Point", "3-Point"]
)

plt.ylabel("Price Change Detected (%)")
plt.title("Price Changes at First Post-Event Observation")
plt.ylim(0, 105)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES / "05_first_observation_detection.png",
    dpi=200
)
plt.close()

# Figure 6: observation delay distribution
plt.figure(figsize=(8, 5))

plt.bar(
    delay_summary["delay_bucket"].astype(str),
    delay_summary["event_count"]
)

plt.xlabel("Time Until Next Recorded Price")
plt.ylabel("Number of Scoring Events")
plt.title("Post-Event Price Observation Delay")
plt.tight_layout()

plt.savefig(
    FIGURES / "06_observation_delay.png",
    dpi=200
)
plt.close()

print("\n=== OVERALL SUMMARY ===")

print("Events:", len(df))
print("Games:", df["event_id"].nunique())
print(
    "Price change detected:",
    round(df["price_changed"].mean() * 100, 2),
    "%"
)
print(
    "Mean observation delay:",
    round(
        df["seconds_to_next_observation"].mean(), 2
    ),
    "seconds"
)

print("\nSaved 3 CSV files and 2 PNG figures.")

print(
    "\nIMPORTANT: These results measure recorded "
    "price changes at the first post-event observation. "
    "They do not identify the actual market reaction "
    "time or establish causality."
)
