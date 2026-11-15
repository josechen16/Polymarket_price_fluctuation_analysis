from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / "Desktop" / "predict_market_project"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(
    RESULTS / "event_level_scoring_analysis.csv"
)

df["is_isolated"] = (
    df["is_isolated"].astype(str).str.lower() == "true"
)

df["aligned_pp"] = df["aligned_price_change"] * 100
df["absolute_pp"] = df["absolute_price_change"] * 100

df["phase"] = np.where(
    df["period"] <= 2, "Early",
    np.where(df["period"] <= 4, "Late", "Overtime")
)

df["close"] = df["pre_event_score_diff"] <= 3
df["large_lead"] = df["pre_event_score_diff"] >= 11

rng = np.random.default_rng(42)
B = 3000

games = sorted(df["event_id"].unique())
game_groups = {
    g: df[df["event_id"] == g]
    for g in games
}

def mean_difference(data, group_a, group_b, column):
    a = data.loc[group_a(data), column]
    b = data.loc[group_b(data), column]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    return float(a.mean() - b.mean())


tests = [
    (
        "Three-point vs two-point (all)",
        df,
        lambda d: d["points_scored"] == 3,
        lambda d: d["points_scored"] == 2,
        "aligned_pp"
    ),
    (
        "Three-point vs two-point (isolated)",
        df[df["is_isolated"]],
        lambda d: d["points_scored"] == 3,
        lambda d: d["points_scored"] == 2,
        "aligned_pp"
    ),
    (
        "Close vs large lead",
        df,
        lambda d: d["close"],
        lambda d: d["large_lead"],
        "absolute_pp"
    ),
    (
        "Late close vs early close",
        df,
        lambda d: d["close"] & (d["phase"] == "Late"),
        lambda d: d["close"] & (d["phase"] == "Early"),
        "absolute_pp"
    )
]

results = []

for name, subset, group_a, group_b, column in tests:
    observed = mean_difference(
        subset, group_a, group_b, column
    )

    subset_groups = {
        g: subset[subset["event_id"] == g]
        for g in games
    }

    estimates = []

    for _ in range(B):
        sampled_games = rng.choice(
            games, size=len(games), replace=True
        )

        sampled = pd.concat(
            [subset_groups[g] for g in sampled_games],
            ignore_index=True
        )

        estimate = mean_difference(
            sampled, group_a, group_b, column
        )

        if np.isfinite(estimate):
            estimates.append(estimate)

    estimates = np.asarray(estimates)

    if len(estimates) < B * 0.95:
        raise RuntimeError(
            f"Too many invalid bootstrap samples: {name}"
        )

    lower, upper = np.percentile(
        estimates, [2.5, 97.5]
    )

    results.append({
        "comparison": name,
        "observed_difference_pp": observed,
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "ci_excludes_zero": bool(
            lower > 0 or upper < 0
        ),
        "bootstrap_samples": len(estimates),
        "games": len(games)
    })

output = pd.DataFrame(results)

print("\n=== GAME-CLUSTERED BOOTSTRAP RESULTS ===")
print(output.to_string(index=False))

output.to_csv(
    RESULTS / "statistical_validation.csv",
    index=False
)

print("\nSaved: results/statistical_validation.csv")
print(
    "\nNote: Bootstrap intervals quantify sampling "
    "uncertainty; they do not establish causality."
)
