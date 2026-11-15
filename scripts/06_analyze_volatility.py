from pathlib import Path
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

df["absolute_change_pp"] = (
    df["absolute_price_change"] * 100
)

df["aligned_change_pp"] = (
    df["aligned_price_change"] * 100
)

df["seconds_to_next_observation"] = (
    df["time_after"] - df["event_time"]
).dt.total_seconds()

df["observation_window_seconds"] = (
    df["time_after"] - df["time_before"]
).dt.total_seconds()

df["game_phase"] = df["period"].apply(
    lambda p: "Early (Q1-Q2)" if p <= 2
    else "Late (Q3-Q4)" if p <= 4
    else "Overtime"
)

df["score_category"] = pd.cut(
    df["pre_event_score_diff"],
    bins=[-1, 3, 10, float("inf")],
    labels=["0-3", "4-10", "11+"]
)

def aggregate(data, columns):
    return (
        data.groupby(columns, observed=True)
        .agg(
            event_count=("absolute_change_pp", "size"),
            game_count=("event_id", "nunique"),
            mean_absolute_change_pp=(
                "absolute_change_pp", "mean"
            ),
            median_absolute_change_pp=(
                "absolute_change_pp", "median"
            ),
            std_absolute_change_pp=(
                "absolute_change_pp", "std"
            ),
            mean_aligned_change_pp=(
                "aligned_change_pp", "mean"
            )
        )
        .reset_index()
    )

print("\n=== DATA VALIDATION ===")
print("Total events:", len(df))
print("Games:", df["event_id"].nunique())
print("Isolated events:", df["is_isolated"].sum())

assert len(df) == 2102, "Unexpected event count"
assert df["event_id"].nunique() == 16
assert df["absolute_change_pp"].notna().all()
assert (df["observation_window_seconds"] > 0).all()
assert (df["seconds_to_next_observation"] >= 0).all()

print("\n=== RQ2: VOLATILITY BY SCORE DIFFERENTIAL ===")

score_summary = aggregate(df, ["score_category"])
print(score_summary.to_string(index=False))
score_summary.to_csv(
    RESULTS / "volatility_by_score.csv", index=False
)

print("\n=== RQ2: VOLATILITY BY PERIOD ===")

period_summary = aggregate(df, ["period"])
print(period_summary.to_string(index=False))
period_summary.to_csv(
    RESULTS / "volatility_by_period.csv", index=False
)

print("\n=== RQ2: SCORE DIFFERENTIAL AND GAME PHASE ===")

phase_summary = aggregate(
    df[df["period"] <= 4],
    ["game_phase", "score_category"]
)
print(phase_summary.to_string(index=False))
phase_summary.to_csv(
    RESULTS / "volatility_by_phase_and_score.csv",
    index=False
)

print("\n=== RQ3: OBSERVATION TIMING ===")

timing = (
    df.groupby("points_scored")
    .agg(
        event_count=("event_id", "size"),
        mean_seconds_to_next_observation=(
            "seconds_to_next_observation", "mean"
        ),
        median_seconds_to_next_observation=(
            "seconds_to_next_observation", "median"
        ),
        mean_window_seconds=(
            "observation_window_seconds", "mean"
        )
    )
    .reset_index()
)

print(timing.to_string(index=False))
timing.to_csv(
    RESULTS / "price_observation_timing.csv",
    index=False
)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "figure.dpi": 130
})

def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=200)
    plt.close()
    print("Saved figure:", filename)

# Figure 1: Scoring impact
impact = (
    df.groupby("points_scored")["aligned_change_pp"]
    .mean()
)

isolated = (
    df[df["is_isolated"]]
    .groupby("points_scored")["aligned_change_pp"]
    .mean()
)

x = list(range(3))
labels = ["Free Throw", "2-Point", "3-Point"]

plt.figure(figsize=(8, 5))
plt.bar(
    [i - 0.2 for i in x],
    [impact.get(i, 0) for i in [1, 2, 3]],
    width=0.4,
    label="All Events"
)
plt.bar(
    [i + 0.2 for i in x],
    [isolated.get(i, 0) for i in [1, 2, 3]],
    width=0.4,
    label="Isolated Events"
)
plt.xticks(x, labels)
plt.ylabel("Mean Aligned Price Change (pp)")
plt.title("Scoring Event Impact on Market Prices")
plt.legend()
save_figure("01_scoring_impact.png")

# Figure 2: Score differential
plt.figure(figsize=(8, 5))
plt.bar(
    score_summary["score_category"].astype(str),
    score_summary["mean_absolute_change_pp"]
)
plt.xlabel("Pre-Event Score Differential")
plt.ylabel("Mean Absolute Price Change (pp)")
plt.title("Market Price Movement by Score Differential")
save_figure("02_volatility_by_score.png")

# Figure 3: Quarter
regular = period_summary[
    period_summary["period"] <= 4
]

plt.figure(figsize=(8, 5))
plt.plot(
    regular["period"],
    regular["mean_absolute_change_pp"],
    marker="o",
    linewidth=2
)
plt.xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
plt.ylabel("Mean Absolute Price Change (pp)")
plt.title("Market Price Movement Across Quarters")
plt.grid(alpha=0.25)
save_figure("03_volatility_by_quarter.png")

# Figure 4: Game phase x score differential
pivot = phase_summary.pivot(
    index="score_category",
    columns="game_phase",
    values="mean_absolute_change_pp"
)

pivot.plot(kind="bar", figsize=(9, 5))
plt.xlabel("Pre-Event Score Differential")
plt.ylabel("Mean Absolute Price Change (pp)")
plt.title("Close Games vs. Early and Late Game")
plt.xticks(rotation=0)
plt.legend(title="Game Phase")
save_figure("04_phase_score_comparison.png")

print("\n=== COMPLETED ===")
print("CSV files saved to:", RESULTS)
print("Figures saved to:", FIGURES)
print(
    "\nNOTE: Time to next price observation is NOT "
    "the actual market reaction latency."
)
