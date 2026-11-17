from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "results/full_season/q1_momentum_clean"
EVENTS = BASE / "data/processed/full_season/event_price/scoring_events.parquet"

df = pd.read_parquet(ROOT / "primary_sample.parquet")
events = pd.read_parquet(EVENTS)

df["espn_game_id"] = df["espn_game_id"].astype(str)
events["espn_game_id"] = events["espn_game_id"].astype(str)
events["event_time"] = pd.to_datetime(
    events["event_time"], utc=True
)

last_score = (
    events.groupby("espn_game_id")["event_time"]
    .max()
    .rename("last_scoring_time")
    .reset_index()
)

df = df.merge(
    last_score,
    on="espn_game_id",
    how="left",
    validate="many_to_one"
)

df["post_5m_minute"] = pd.to_datetime(
    df["post_5m_minute"], utc=True
)

df["after_last_score"] = (
    df["post_5m_minute"] > df["last_scoring_time"]
)

direction = np.where(
    df["run_team"] == "home", 1, -1
)

df["score_diff_before"] = (
    direction * df["score_diff_after"]
    - df["run_points"]
)

df["run_team_pre_prob"] = np.where(
    df["run_team"] == "home",
    df["price_before"],
    1 - df["price_before"]
)

df["reaction_pp"] = df["reaction_5m"] * 100
df["duration_min"] = df["run_duration_seconds"] / 60

df["quarter"] = df["end_period"].astype(str)
df["season"] = df["season"].astype(str)

formula = (
    "reaction_pp ~ run_points + duration_min "
    "+ score_diff_before + run_team_pre_prob "
    "+ C(quarter) + C(season)"
)

def fit(name, data):
    data = data.replace([np.inf, -np.inf], np.nan)
    required = [
        "reaction_pp", "run_points", "duration_min",
        "score_diff_before", "run_team_pre_prob",
        "quarter", "season", "espn_game_id"
    ]
    data = data.dropna(subset=required).copy()

    model = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["espn_game_id"]}
    )

    (ROOT / f"final_regression_{name}.txt").write_text(
        model.summary().as_text()
    )

    ci = model.conf_int().loc["run_points"]

    return {
        "model": name,
        "n_runs": len(data),
        "n_games": data["espn_game_id"].nunique(),
        "beta_pp_per_point": model.params["run_points"],
        "standard_error": model.bse["run_points"],
        "p_value": model.pvalues["run_points"],
        "ci95_lower": ci.iloc[0],
        "ci95_upper": ci.iloc[1],
        "r_squared": model.rsquared
    }

before_last_score = df[~df["after_last_score"]]

samples = {
    "clean_all": df,
    "before_last_score": before_last_score,
    "before_last_score_non_extreme":
        before_last_score[
            ~before_last_score["near_extreme_price"]
        ],
    "first_three_quarters":
        before_last_score[
            before_last_score["end_period"].between(1, 3)
        ],
    "within_quarter":
        before_last_score[
            ~before_last_score["cross_quarter"]
        ]
}

results = pd.DataFrame([
    fit(name, data)
    for name, data in samples.items()
])

results.to_csv(
    ROOT / "final_regression_comparison.csv",
    index=False
)

print("=== FINAL Q1 REGRESSION ===")
print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\n=== SAMPLE CHECK ===")
print("Total cleaned runs:", len(df))
print("Runs before last scoring play:", len(before_last_score))
print("Games:", before_last_score["espn_game_id"].nunique())

print("\nSaved to:", ROOT)
