import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "data/processed/full_season"
NBA = BASE / "data/raw/nba/full_season"

MARKETS = ROOT / "nba_moneyline_markets"
SCHEDULE = NBA / "espn_schedule.json"
GAMES = NBA / "games"

ROOT.mkdir(parents=True, exist_ok=True)

TEAM_NAMES = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

ALIASES = {
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "PHO": "PHX",
    "Twolves": "MIN",
    "T-Wolves": "MIN",
    "Wolves": "MIN",
    "Sixers": "PHI",
    "Cavs": "CLE",
    "Mavs": "DAL",
    "Pels": "NOP",
    "Dubs": "GSW",
    "Trail Blazers": "POR",
}

def clean(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

LOOKUP = {}

for code, name in TEAM_NAMES.items():
    LOOKUP[clean(code)] = code
    LOOKUP[clean(name)] = code
    LOOKUP[clean(name.split()[-1])] = code

for alias, code in ALIASES.items():
    LOOKUP[clean(alias)] = code

def team_code(value):
    return LOOKUP.get(clean(value))

def utc(value):
    return pd.to_datetime(value, utc=True, errors="coerce")

def main():
    print("Loading markets...", flush=True)

    raw = pd.read_parquet(MARKETS)
    raw["game_start_time"] = utc(raw["game_start_time"])

    market_records = []
    unknown = set()

    for condition_id, group in raw.groupby("condition_id"):
        if len(group) != 2:
            continue

        group = group.sort_values("outcome_index")
        a, b = group.iloc[0], group.iloc[1]

        ac = team_code(a["team"])
        bc = team_code(b["team"])

        if not ac:
            unknown.add(str(a["team"]))
        if not bc:
            unknown.add(str(b["team"]))

        if not ac or not bc or ac == bc:
            continue

        market_records.append({
            "condition_id": str(condition_id),
            "polymarket_event_id": str(a["event_id"]),
            "season": str(a["season"]),
            "market_title": str(a["event_title"]),
            "market_time": a["game_start_time"],
            "team_a": ac,
            "team_b": bc,
            "token_a": str(a["token_id"]),
            "token_b": str(b["token_id"]),
        })

    print("Valid markets:", len(market_records))
    print("Unknown outcomes:", sorted(unknown))

    schedule = json.loads(SCHEDULE.read_text())
    espn_records = []
    excluded = []

    for game in schedule:
        game_id = str(game["event_id"])
        path = GAMES / f"{game_id}.json"

        if not path.exists():
            continue

        home = team_code(game.get("home_team"))
        away = team_code(game.get("away_team"))

        if not home or not away:
            excluded.append(game)
            continue

        espn_records.append({
            "espn_game_id": game_id,
            "season": str(game["season"]),
            "game_time": utc(game.get("date")),
            "home_team": home,
            "away_team": away,
            "game_name": game.get("name"),
        })

    print("ESPN games:", len(espn_records))
    print("Excluded non-NBA-team games:", len(excluded))

    market_index = {}

    for market in market_records:
        key = (
            market["season"],
            tuple(sorted([market["team_a"], market["team_b"]]))
        )
        market_index.setdefault(key, []).append(market)

    candidates = []

    for game in espn_records:
        key = (
            game["season"],
            tuple(sorted([
                game["home_team"], game["away_team"]
            ]))
        )

        for market in market_index.get(key, []):
            if pd.isna(game["game_time"]) or pd.isna(
                market["market_time"]
            ):
                continue

            hours = abs(
                (game["game_time"] - market["market_time"])
                .total_seconds()
            ) / 3600

            if hours > 2:
                continue

            home_token = (
                market["token_a"]
                if market["team_a"] == game["home_team"]
                else market["token_b"]
            )
            away_token = (
                market["token_a"]
                if market["team_a"] == game["away_team"]
                else market["token_b"]
            )

            candidates.append({
                **game,
                "condition_id": market["condition_id"],
                "polymarket_event_id":
                    market["polymarket_event_id"],
                "market_title": market["market_title"],
                "market_time": market["market_time"],
                "home_token_id": home_token,
                "away_token_id": away_token,
                "time_difference_hours": hours,
            })

    if not candidates:
        raise RuntimeError("No candidates found")

    df = pd.DataFrame(candidates)

    df.to_csv(
        ROOT / "game_matching_candidates.csv",
        index=False
    )

    # Each game must have a unique closest market.
    game_min = df.groupby("espn_game_id")[
        "time_difference_hours"
    ].transform("min")

    # Each market must have a unique closest game.
    market_min = df.groupby("condition_id")[
        "time_difference_hours"
    ].transform("min")

    best = df[
        (df["time_difference_hours"] == game_min)
        & (df["time_difference_hours"] == market_min)
    ].copy()

    game_count = best.groupby("espn_game_id")[
        "condition_id"
    ].transform("count")

    market_count = best.groupby("condition_id")[
        "espn_game_id"
    ].transform("count")

    matched = best[
        (game_count == 1) & (market_count == 1)
    ].copy()

    ambiguous = df[
        ~df.index.isin(matched.index)
    ].copy()

    assert matched["espn_game_id"].is_unique
    assert matched["condition_id"].is_unique
    assert (
        matched["home_token_id"] != matched["away_token_id"]
    ).all()

    matched.to_csv(
        ROOT / "matched_games.csv", index=False
    )
    matched.to_parquet(
        ROOT / "matched_games.parquet", index=False
    )
    ambiguous.to_csv(
        ROOT / "ambiguous_matches.csv", index=False
    )

    matched_games = set(matched["espn_game_id"])
    matched_markets = set(matched["condition_id"])

    unmatched_espn = pd.DataFrame([
        g for g in espn_records
        if g["espn_game_id"] not in matched_games
    ])

    unmatched_markets = pd.DataFrame([
        m for m in market_records
        if m["condition_id"] not in matched_markets
    ])

    unmatched_espn.to_csv(
        ROOT / "unmatched_espn_games.csv", index=False
    )
    unmatched_markets.to_csv(
        ROOT / "unmatched_polymarket_markets.csv",
        index=False
    )

    print("\n=== IMPROVED GAME MATCHING REPORT ===")
    print("ESPN games:", len(espn_records))
    print("Valid markets:", len(market_records))
    print("Candidates within 2 hours:", len(df))
    print("Unique matched games:", len(matched))
    print(
        "Unmatched ESPN games:",
        len(unmatched_espn)
    )
    print(
        "Unmatched markets:",
        len(unmatched_markets)
    )
    print(
        "Remaining ambiguous candidate rows:",
        len(ambiguous)
    )

    print("\nMatched by season:")
    if not matched.empty:
        print(matched.groupby("season").size())

    print("\nTime differences:")
    if not matched.empty:
        print(
            matched["time_difference_hours"]
            .describe()
        )

    print("\nSaved:", ROOT)

if __name__ == "__main__":
    main()
