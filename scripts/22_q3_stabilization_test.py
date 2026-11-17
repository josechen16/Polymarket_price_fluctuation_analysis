from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "results/full_season/q3_price_settling"
ROOT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "q3_event_level.parquet")

df["espn_game_id"] = df["espn_game_id"].astype(str)

df["isolated_5m"] = (
    df["seconds_to_next_score"].notna()
    & df["seconds_to_next_score"].gt(300)
    & df["seconds_to_next_score"].le(3600)
)

samples = {
    "all_events": df,
    "isolated_5m": df[df["isolated_5m"]],
    "isolated_first_three_quarters": df[
        df["isolated_5m"]
        & df["period"].between(1, 3)
    ]
}

rng = np.random.default_rng(42)
results = []

for name, sample in samples.items():
    sample = sample.dropna(
        subset=["move_1_2m_pp", "move_2_3m_pp"]
    ).copy()

    sample["difference"] = (
        sample["move_1_2m_pp"]
        - sample["move_2_3m_pp"]
    )

    # Cluster-robust inference for the mean paired difference.
    game_stats = sample.groupby("espn_game_id").agg(
        total_difference=("difference", "sum"),
        n=("difference", "size")
    )

    n = len(sample)
    g = len(game_stats)
    estimate = sample["difference"].mean()

    if g < 2:
        raise ValueError("Need at least two games")

    centered = (
        game_stats["total_difference"].to_numpy()
        - estimate * game_stats["n"].to_numpy()
    )

    se = np.sqrt(
        (g / (g - 1))
        * np.sum(centered ** 2)
        / (n ** 2)
    )

    t_stat = estimate / se if se > 0 else np.nan
    p_two_sided = (
        2 * stats.t.sf(abs(t_stat), df=g - 1)
        if se > 0 else np.nan
    )

    critical = stats.t.ppf(0.975, df=g - 1)
    ci_low = estimate - critical * se
    ci_high = estimate + critical * se

    first = sample["move_1_2m_pp"].mean()
    second = sample["move_2_3m_pp"].mean()

    results.append({
        "sample": name,
        "n_events": n,
        "n_games": g,
        "mean_move_1_2m_pp": first,
        "mean_move_2_3m_pp": second,
        "mean_difference_pp": estimate,
        "relative_decline_pct": (
            (first - second) / first * 100
            if first > 0 else np.nan
        ),
        "cluster_se": se,
        "t_stat": t_stat,
        "p_value_two_sided": p_two_sided,
        "ci95_lower": ci_low,
        "ci95_upper": ci_high,
        "fraction_events_declining": (
            sample["difference"] > 0
        ).mean(),
        "fraction_events_equal": (
            sample["difference"] == 0
        ).mean()
    })

result = pd.DataFrame(results)
result.to_csv(
    ROOT / "stabilization_test.csv",
    index=False
)

print("=== Q3 STABILIZATION TEST ===")
print(
    result.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

print("\n=== INTERPRETATION ===")
for _, row in result.iterrows():
    print(f"\nSample: {row['sample']}")
    print(f"Events: {row['n_events']}")
    print(f"Games: {row['n_games']}")
    print(
        "Mean reduction:",
        round(row["mean_difference_pp"], 4),
        "percentage points"
    )
    print(
        "Relative decline:",
        round(row["relative_decline_pct"], 2),
        "%"
    )
    print(
        "95% CI:",
        f"[{row['ci95_lower']:.4f}, "
        f"{row['ci95_upper']:.4f}]"
    )
    print(
        "Two-sided p-value:",
        f"{row['p_value_two_sided']:.6g}"
    )
    print(
        "Fraction with declining movement:",
        f"{row['fraction_events_declining']:.2%}"
    )
    print(
        "Fraction with equal movement:",
        f"{row['fraction_events_equal']:.2%}"
    )

print("\nSaved to:", ROOT)
