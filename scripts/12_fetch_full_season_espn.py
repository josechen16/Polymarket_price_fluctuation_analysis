import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / "data/raw/nba/full_season"
SCHEDULE_DIR = ROOT / "schedule_daily"
GAMES_DIR = ROOT / "games"

API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

SEASONS = {
    "2024-25": (date(2024, 10, 1), date(2025, 7, 1)),
    "2025-26": (date(2025, 10, 1), date(2026, 7, 1)),
}

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as f:
        json.dump(data, f)
    os.replace(temp, path)

def load_json(path):
    with path.open() as f:
        return json.load(f)

def fetch_json(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(
                request, timeout=15
            ) as response:
                return json.load(response)

        except urllib.error.HTTPError as error:
            if 400 <= error.code < 500:
                raise
            if attempt == 2:
                raise

        except Exception:
            if attempt == 2:
                raise

        time.sleep(2 * (attempt + 1))

def get_schedule(season, start, end, delay, max_days):
    directory = SCHEDULE_DIR / season
    directory.mkdir(parents=True, exist_ok=True)

    days = (end - start).days + 1
    if max_days:
        days = min(days, max_days)

    games = {}
    failures = []

    for i in range(days):
        current = start + timedelta(days=i)
        day = current.strftime("%Y%m%d")
        path = directory / f"{day}.json"

        print(
            f"[Schedule {i+1}/{days}] {day}",
            flush=True
        )

        try:
            if path.exists():
                data = load_json(path)
                print("  Cached", flush=True)
            else:
                url = f"{API}/scoreboard?dates={day}&limit=100"
                data = fetch_json(url)
                save_json(path, data)
                time.sleep(delay)

            events = data.get("events", [])
            print("  Events:", len(events), flush=True)

            for event in events:
                event_id = str(event.get("id", ""))
                if not event_id:
                    continue

                season_type = event.get("season", {}).get("type")
                status = event.get("status", {}).get("type", {})
                completed = status.get("completed", False)

                if season_type not in (2, 3) or not completed:
                    continue

                competition = (
                    event.get("competitions") or [{}]
                )[0]

                teams = {
                    c.get("homeAway"): c.get(
                        "team", {}
                    ).get("displayName")
                    for c in competition.get(
                        "competitors", []
                    )
                }

                games[event_id] = {
                    "event_id": event_id,
                    "season": season,
                    "date": event.get("date"),
                    "name": event.get("name"),
                    "home_team": teams.get("home"),
                    "away_team": teams.get("away"),
                    "season_type": season_type,
                }

        except Exception as error:
            failures.append({
                "date": day,
                "error": str(error)
            })
            print("  FAILED:", error, flush=True)

    return games, failures

def fetch_games(games, delay, limit):
    GAMES_DIR.mkdir(parents=True, exist_ok=True)

    items = sorted(
        games.items(),
        key=lambda item: item[1].get("date") or ""
    )

    if limit:
        items = items[:limit]

    result = {
        "downloaded": 0,
        "cached": 0,
        "missing_plays": 0,
        "failures": []
    }

    for i, (event_id, game) in enumerate(items, 1):
        path = GAMES_DIR / f"{event_id}.json"

        if path.exists():
            try:
                if load_json(path).get("plays"):
                    result["cached"] += 1
                    print(
                        f"[Game {i}/{len(items)}] Cached {event_id}",
                        flush=True
                    )
                    continue
            except Exception:
                pass

        print(
            f"[Game {i}/{len(items)}] {game['name']}",
            flush=True
        )

        try:
            data = fetch_json(
                f"{API}/summary?event={event_id}"
            )

            plays = data.get("plays", [])

            if not plays:
                result["missing_plays"] += 1
                result["failures"].append({
                    "event_id": event_id,
                    "reason": "no_plays"
                })
                print("  No plays", flush=True)
                continue

            save_json(path, data)
            result["downloaded"] += 1
            print("  Plays:", len(plays), flush=True)

        except Exception as error:
            result["failures"].append({
                "event_id": event_id,
                "reason": str(error)
            })
            print("  FAILED:", error, flush=True)

        time.sleep(delay)

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--season",
        choices=["2024-25", "2025-26", "both"],
        default="both"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--schedule-days", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    selected = (
        list(SEASONS)
        if args.season == "both"
        else [args.season]
    )

    all_games = {}
    schedule_failures = []

    for season in selected:
        start, end = SEASONS[season]

        print("\nFetching schedule:", season, flush=True)

        games, failures = get_schedule(
            season,
            start,
            end,
            args.delay,
            args.schedule_days
        )

        print(
            "Completed games found:",
            len(games),
            flush=True
        )

        all_games.update(games)
        schedule_failures.extend(failures)

    schedule_file = (
        ROOT / "espn_schedule_test.json"
        if args.schedule_days
        else ROOT / "espn_schedule.json"
    )

    save_json(schedule_file, list(all_games.values()))

    result = fetch_games(
        all_games,
        args.delay,
        args.limit
    )

    result["scheduled_games"] = len(all_games)
    result["schedule_failures"] = schedule_failures

    report_file = (
        ROOT / "download_report_test.json"
        if args.schedule_days
        else ROOT / "download_report.json"
    )

    save_json(report_file, result)

    print("\n=== DOWNLOAD REPORT ===")
    print("Scheduled games:", len(all_games))
    print("New downloads:", result["downloaded"])
    print("Cached games:", result["cached"])
    print("Missing plays:", result["missing_plays"])
    print("Game failures:", len(result["failures"]))
    print("Schedule failures:", len(schedule_failures))
    print("Report:", report_file)

if __name__ == "__main__":
    main()
